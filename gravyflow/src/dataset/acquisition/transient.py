"""
Transient mode data acquisition.

This module provides TransientDataObtainer for acquiring data around
specific transient events (GW events, glitches) in TRANSIENT mode.

Every example is read through the store (``store.read``), as noise is: on each
detector, one view of the source's tiles over its offsource and padded onsource,
resampled by the one resampler. A record is planned only where the timeline holds its
span on every detector it is read on at its own time; a record whose strain the source
then does not deliver is refused by name (``TransientUnavailable``), never skipped,
zero-filled or stored as absent.
"""

import dataclasses
import itertools
import logging
from collections import Counter
from copy import deepcopy
from typing import Any, Callable, List, Dict, Iterator, NamedTuple, Optional, Sequence, Union, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features.event import EventConfidence
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, Coincidence,
    AcquisitionMode, SamplingMode, ObservingRun, label_kinds
)
from gravyflow.src.utils.numerics import ensure_list
from gravyflow.src.utils.gps import as_gps_times
from gravyflow.src.utils.tensor import centre_index
from gravyflow.src.dataset.features.transient_index import TransientIndex, transient_footprint
from .transient_segment import TransientSegment
from .gwosc_data import within_segments
from .sources import DataSource
from gravyflow.src.dataset.store import read as store_read
from gravyflow.src.dataset.store.pools import thread_pool
from gravyflow.src.dataset.store.read import TimelineContradiction
from .augmentations import augmented
from .partners import PARTNER_GRID_HERTZ, partner_time
from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.dataset.splits import SplitSpec
from gravyflow.src.rng import BatchStreams, SeedTree


#: The most threads that read one batch's examples at once, so a cold batch fetches its
#: tiles together; the store's single-flight table fetches a shared tile once, and its
#: memory accountant bounds what the fetches hold.
READ_THREADS = 8


class DetectorRead(NamedTuple):
    """
    Where an example of one record reads one detector: the centre of its window,
    and whether it is a partner (``partners``), read away from the transient, or
    read at the transient's time. A detector is labelled with the transient only
    where the record was seen in it (``seen_in``); every other is noise.
    """

    ifo: "gf.IFO"
    centre: float
    partner: bool


class TransientUnavailable(TimelineContradiction):
    """
    A planned transient example that cannot be read: the timeline holds its span, and
    the source does not deliver it (its fetch failed after its retries, or the samples
    are not held: non-finite, or not in the source's file). It names the record and the
    read, with all a ``TimelineContradiction`` names: the detector, the interval, the
    source, the cause and the ``exclusions`` entry that plans around it. A record is
    never skipped, zero-filled or stored as absent in its place.
    """

    def __init__(self, record: TransientSegment, read: "DetectorRead", contradiction: TimelineContradiction):
        self.record, self.read = record, read
        kind = getattr(record.kind, "name", str(record.kind))
        what = f"{'event' if record.is_event else 'glitch'} {record.name or kind}"
        where = f", read at its partner time {read.centre:.4f}" if read.partner else ""
        super().__init__(
            contradiction.source, contradiction.detector, contradiction.interval,
            f"the {what} at GPS {record.transient_gps_time:.4f}{where}: {contradiction.reason}",
            contradiction.exclusion,
        )


class TransientDataObtainer(BaseDataObtainer):
    """
    Data obtainer for TRANSIENT mode acquisition.
    
    Acquires data windows centered around specific events (GW mergers, glitches),
    suitable for event classification or detection training.
    """
    
    def __init__(
            self,
            data_quality: DataQuality,
            data_labels: Union[DataLabel, List[DataLabel]],
            observing_runs: Union[ObservingRun, List[ObservingRun]] = None,
            segment_order: SegmentOrder = SegmentOrder.RANDOM,
            max_segment_duration_seconds: float = 2048.0,
            saturation: float = 8.0,
            overrides: dict = None,
            event_types: List[EventConfidence] = None,
            logging_level: int = logging.WARNING,
            augmentations: List = None,  # Training only; None is none
            # Class balancing
            balanced_glitch_types: bool = False,
            # Specific Names
            event_names: List[str] = None,
            split_spec: Optional[SplitSpec] = None,
            train_on_real_events: bool = False,
            source: Optional[DataSource] = None,
            coincidence: Coincidence = Coincidence.SEPARATED,
            exclusions=(),
        ):
        """
        Args (split and source):
            split_spec: How GPS time is split (``gf.SplitSpec``); None is the default.
            train_on_real_events: Real events are evaluation data, all in ``test``;
                True puts each in its footprint's split instead, training on some.
                An event is read at one time on every detector that observed it, so
                training on events needs the coincident opt-in.
            coincidence: ``gf.Coincidence.COINCIDENT`` reads every requested
                detector at the transient's time; by default a detector the
                transient was not seen in is read at a partner time.
            source: Where strain comes from (``gf.DataSource``); None takes
                ``gf.Defaults.source`` (``BaseDataObtainer``).
            exclusions: ``(detector, (start, end), reason)`` entries taken from the
                readable time (``BaseDataObtainer``).
        """
        if train_on_real_events and Coincidence(coincidence) != Coincidence.COINCIDENT:
            raise ValueError(
                "Training on real events puts coincident data in training: pass "
                "coincidence=Coincidence.COINCIDENT with train_on_real_events=True, so its "
                "batches carry the COINCIDENT flag."
            )
        
        # Default observing runs matching BaseDataObtainer behavior
        if observing_runs is None:
            observing_runs = [ObservingRun.O1, ObservingRun.O2, ObservingRun.O3]
        if not isinstance(observing_runs, list):
            observing_runs = [observing_runs]
            
        self.observing_runs = observing_runs
        self.event_names = event_names

        super().__init__(
            data_quality=data_quality,
            data_labels=data_labels,
            observing_runs=observing_runs,
            segment_order=segment_order,
            max_segment_duration_seconds=max_segment_duration_seconds,
            saturation=saturation,
            overrides=overrides,
            logging_level=logging_level,
            augmentations=augmentations,
            split_spec=split_spec,
            source=source,
            coincidence=coincidence,
            exclusions=exclusions,
        )

        self.acquisition_mode = AcquisitionMode.TRANSIENT
        self.train_on_real_events = bool(train_on_real_events)
        
        # Event type filtering
        if event_types is None:
            event_types = [EventConfidence.CONFIDENT]
        self.event_types = event_types
        
        # Class balancing
        self.balanced_glitch_types = balanced_glitch_types

    def _served_state(self) -> Dict[str, Any]:
        return {
            **super()._served_state(),
            # TransientIndex - canonical source of truth for transient examples
            "_feature_index": None,
            # The split's records, in serving order, and each record's reads, one
            # per requested detector in order (_plan_reads)
            "transient_segments": [],
            "transient_reads": [],
            # Records not planned, by reason: a census of the split, made as it is
            # planned (_passing_data_quality, _plan_reads)
            "_rejections": Counter(),
        }

    #: Why a record is not planned: its span leaves the readable data of a detector
    #: it is read on at its own time (an event: of every requested detector); a
    #: detector it was not seen in has no partner time; or, under the coincident
    #: opt-in, its window on a detector it was not seen in meets a catalogued glitch
    #: there. A planned record whose strain the source does not deliver is refused
    #: (``TransientUnavailable``), never counted here.
    REJECTED_BY_DATA_QUALITY = "data quality"
    REJECTED_FOR_NO_PARTNER = "no partner"
    REJECTED_FOR_OTHER_TRANSIENT = "transient in another detector"

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """
        ``rejections``: the records of the split not planned, by reason
        (``REJECTED_BY_DATA_QUALITY``, ``REJECTED_FOR_NO_PARTNER``,
        ``REJECTED_FOR_OTHER_TRANSIENT``), each record counted once; ``dropped_segments``
        is their total.
        """
        return {
            "dropped_segments": sum(self._rejections.values()),
            "rejections": dict(self._rejections),
        }

    def reset_acquisition_stats(self) -> None:
        self._rejections = Counter()

    def _own_detectors(self, record: TransientSegment, ifos: List[gf.IFO]) -> List[gf.IFO]:
        """
        The requested detectors ``record`` is read on at its own time: those it was
        seen in (an event's from data quality, ``_passing_data_quality``), or every
        one under the coincident opt-in. The others are read at partner times
        (``_plan_reads``).
        """
        if self.coincidence == Coincidence.COINCIDENT:
            return list(ifos)
        return [ifo for ifo in ifos if ifo in record.seen_in]

    def _count_rejections(self, reason: str, rejected: int, total: int, why: str) -> None:
        """Record ``rejected`` of ``total`` records as the split's ``reason`` count."""
        self._rejections.pop(reason, None)
        if rejected:
            self._rejections[reason] = rejected
            logger.info(f"{rejected} of {total} transients rejected: {why}.")

    def _passing_data_quality(
        self, segments: List[TransientSegment], ifos: List[gf.IFO],
        read_span: Optional[Tuple[float, float]] = None,
    ) -> List[TransientSegment]:
        """
        The records whose read (``read_span`` about the transient, ``_read_span``; the
        record's own span when None) lies, on every detector read at their own time
        (``_own_detectors``), inside one readable segment (``get_all_segment_times``:
        the obtainer's GWOSC flags, less its ``exclusions``, so by default no hardware
        injection either). The others are rejected at plan time, and their number is
        the split's ``REJECTED_BY_DATA_QUALITY`` count. A partner's data quality is its
        own (``_partner_segments``).

        An event's detectors are found here: the catalogue does not say which
        detectors observed it, so it is seen in each requested detector whose usable
        data holds its span (its ``seen_in``, returned), and an event seen in none is
        rejected.
        """
        records, keep = list(segments), np.ones(len(segments), dtype=bool)
        if records:
            spans = np.array([_span_about(s, s.transient_gps_time, read_span) for s in records], dtype=np.float64)
            usable = {ifo: within_segments(spans, self.get_all_segment_times(ifo)) for ifo in ifos}
            for k, record in enumerate(records):
                if record.is_event:
                    records[k] = dataclasses.replace(record, seen_in=[ifo for ifo in ifos if usable[ifo][k]])
                keep[k] = any(ifo in records[k].seen_in for ifo in ifos) and all(
                    usable[ifo][k] for ifo in self._own_detectors(records[k], ifos))
        self._count_rejections(
            self.REJECTED_BY_DATA_QUALITY, int(np.sum(~keep)), len(records),
            f"their windows leave the usable data of {[ifo.name for ifo in ifos]}",
        )
        return list(itertools.compress(records, keep))

    def _partner_segments(self, ifo: gf.IFO, group_name: str) -> np.ndarray:
        """
        Where a partner of ``ifo`` may read: its usable data in split
        ``group_name`` (``SplitSpec.admissible``), less every event's footprint and
        guard and every one of its catalogued glitches, as noise is vetoed.
        """
        vetoes = np.concatenate([
            np.reshape(self.get_event_veto_segments(), (-1, 2)),
            np.reshape(self.get_glitch_veto_segments(ifo), (-1, 2)),
        ])
        admissible = self.split_spec.admissible(self.get_all_segment_times(ifo), group_name)
        return self.veto_time_segments(admissible, vetoes)

    def _partner_blocks(self, record: TransientSegment, group_name: str) -> Iterator[Tuple[float, float]]:
        """
        The blocks, as GPS intervals, a partner of ``record`` may lie in, in the
        order they are tried. A glitch's is its own block, which is in its split.
        An event is in ``test`` whatever its block (unless trained on), so its
        partner lies in the nearest block of split ``group_name`` in the event's
        observing run: its own block when that is one, else the next out, earlier
        first at equal distance. An event outside every run has none.
        """
        spec = self.split_spec
        own = spec.block_of(record.transient_gps_time)
        if not record.is_event:
            yield own * spec.block_seconds, (own + 1) * spec.block_seconds
            return
        t = float(record.transient_gps_time)
        runs = [(start, end) for start, end in zip(self.start_gps_times, self.end_gps_times) if start <= t < end]
        if not runs:
            return
        first, last = spec.block_of(runs[0][0]), spec.block_of(runs[0][1])
        for distance in range(max(own - first, last - own) + 1):
            for block in sorted({own - distance, own + distance}):
                if first <= block <= last and spec.block_split(block) == group_name:
                    yield block * spec.block_seconds, (block + 1) * spec.block_seconds

    def _plan_reads(
        self, segments: List[TransientSegment], ifos: List[gf.IFO], group_name: str,
        read_span: Optional[Tuple[float, float]] = None,
    ) -> Tuple[List[TransientSegment], List[Tuple[DetectorRead, ...]]]:
        """
        Each record's reads, one per requested detector in order: its own detectors
        at its own time, every other at a partner time (``partners.partner_time``)
        whose read (``read_span``; the record's span when None) lies in that
        detector's partner segments and the first of the record's partner
        blocks that holds one (``_partner_blocks``), its centre at
        least twice the record's footprint from the transient, keyed by the split's
        fingerprint, the transient's time and the detector. The records with no
        partner time on a detector they need are rejected, and their number is the
        split's ``REJECTED_FOR_NO_PARTNER`` count.

        Under the coincident opt-in every detector is read at the record's time, and
        one the record was not seen in is labelled noise: a record whose read there
        meets one of that detector's catalogued glitches (``get_glitch_veto_segments``)
        is rejected, and their number is the ``REJECTED_FOR_OTHER_TRANSIENT`` count.
        """
        spec = self.split_spec
        fingerprint = spec.fingerprint()
        allowed: Dict[gf.IFO, np.ndarray] = {}
        glitches: Dict[gf.IFO, np.ndarray] = {}
        rejected: Counter = Counter()
        planned, plans = [], []
        for record in segments:
            t = float(record.transient_gps_time)
            own = self._own_detectors(record, ifos)
            footprint_start, footprint_end = transient_footprint(record)
            read_start, read_end = _span_about(record, t, read_span)
            reads = []
            for ifo in ifos:
                if ifo in own:
                    if ifo not in record.seen_in:
                        if ifo not in glitches:
                            glitches[ifo] = np.reshape(self.get_glitch_veto_segments(ifo), (-1, 2))
                        if np.any((glitches[ifo][:, 0] < read_end) & (glitches[ifo][:, 1] > read_start)):
                            rejected[self.REJECTED_FOR_OTHER_TRANSIENT] += 1
                            break
                    reads.append(DetectorRead(ifo, t, False))
                    continue
                if ifo not in allowed:
                    allowed[ifo] = self._partner_segments(ifo, group_name)
                centre = next((found for found in (
                    partner_time(
                        allowed[ifo], t,
                        before=t - read_start, after=read_end - t,
                        block=block,
                        separation=2.0 * (footprint_end - footprint_start),
                        key=(fingerprint, repr(t), ifo.name),
                        sample_rate_hertz=PARTNER_GRID_HERTZ,
                    ) for block in self._partner_blocks(record, group_name)
                ) if found is not None), None)
                if centre is None:
                    rejected[self.REJECTED_FOR_NO_PARTNER] += 1
                    break
                reads.append(DetectorRead(ifo, centre, True))
            else:
                planned.append(record)
                plans.append(tuple(reads))
        self._count_rejections(
            self.REJECTED_FOR_NO_PARTNER, rejected[self.REJECTED_FOR_NO_PARTNER], len(segments),
            "a detector they were not seen in has no partner time",
        )
        self._count_rejections(
            self.REJECTED_FOR_OTHER_TRANSIENT, rejected[self.REJECTED_FOR_OTHER_TRANSIENT], len(segments),
            "a detector they were not seen in holds a glitch of its own at their time",
        )
        return planned, plans

    def _events_only(self) -> bool:
        """Whether the obtainer serves real events and nothing else (by its labels' kinds)."""
        return label_kinds(self.data_labels) == {DataLabel.EVENTS}

    def build_feature_index(self, ifos: List[gf.IFO]) -> "TransientIndex":
        """
        Build the TransientIndex of the requested records, each assigned its split
        by ``self.split_spec`` (``TransientIndex.assign_splits``), with every
        catalogued event's footprint kept out of the glitches.
        """
        from gravyflow.src.dataset.features.segment_builders import (
            build_glitch_segments, build_event_segments
        )
        from gravyflow.src.dataset.features.transient_index import TransientIndex
        from gravyflow.src.dataset.features.glitch import GlitchType
        from gravyflow.src.dataset.features.event import SourceType
        
        all_segments = []
        # What is asked for, by each label's kind, never its number (label_kinds).
        asked = label_kinds(self.data_labels)

        if DataLabel.GLITCHES in asked:
            # Specific glitch types, if any are named; otherwise every type.
            glitch_segments = build_glitch_segments(
                ifos=ifos,
                observing_runs=self.observing_runs,
                glitch_types=[label for label in self.data_labels if isinstance(label, GlitchType)] or None,
            )
            all_segments.extend(glitch_segments)

        if DataLabel.EVENTS in asked:
            # Determine observing runs to query: if specific event names are requested, 
            # we should broaden the search (e.g., check all runs) to ensure they are found.
            # Otherwise, respect the strict observing_runs filter.
            effective_runs = self.observing_runs
            
            # Note: Previously we forced effective_runs=None here to search all runs.
            # Now we respect the user's observing_runs filter. If finding an event from a different
            # run is desired, set observing_runs=None or include the target run.
            
            logger.info(f"Building event segments (runs={effective_runs})")
            event_segments = build_event_segments(
                observing_runs=effective_runs,
                confidences=self.event_types if self.event_types else [EventConfidence.CONFIDENT],
                event_names=self.event_names
            )
            all_segments.extend(event_segments)
            logger.info(f"Built {len(event_segments)} event segments")
        
        # Create index
        index = TransientIndex(all_segments)
        index.dedupe()
        index.sort()
        
        # Filter by specific event names if requested
        if self.event_names:
            logger.info(f"Filtering to specific events: {self.event_names}")
            index = index.filter_by_names(self.event_names)
            logger.info(f"After filtering: {len(index)} segments")
        
        from gravyflow.src.dataset.features.event import get_event_catalogue
        from gravyflow.src.dataset.features.transient_index import event_footprints

        catalogue = get_event_catalogue()
        excluded = index.assign_splits(
            self.split_spec,
            event_footprints(catalogue["gps"], catalogue["chirp_mass_source"]),
            train_on_real_events=self.train_on_real_events,
        )
        if excluded:
            logger.info(f"Transient records in no split: {excluded}")

        self._feature_index = index
        return index


    def _prepare_batch(
        self,
        batch_subarrays: list,
        batch_backgrounds: list,
        batch_segments: List[TransientSegment],
        batch_reads: List[Tuple[DetectorRead, ...]],
        window_spec: WindowSpec,
        shifts: Optional[np.ndarray] = None,
    ) -> dict:
        """
        The batch of ``batch_segments``' examples, each one record's reads
        (``_read_example``), each example's windows moved by its ``shifts``
        seconds (B,), none when None.

        Every label is per detector, from the record and that detector's read: a
        partner is noise, with no type.

        Returns:
            Dict with ReturnVariables keys:
            - ONSOURCE: (Batch, IFO, Samples) = BIS
            - OFFSOURCE: (Batch, IFO, Samples) = BIS
            - START_GPS_TIME: (Batch, IFO) = BI, float64: the GPS time of the first
              onsource sample returned after cropping, as for noise
              (``WindowSpec.onsource_start_gps``), exactly: the window centred on the
              sample nearest the detector's read, moved by its shift
              (``WindowSpec.transient_window_start_gps``).
            - TRANSIENT_GPS_TIME: (Batch, IFO) = BI, float64: each detector's read:
              the transient's time, or a partner's; the window's centre unless shifted.
            - DATA_LABEL: (Batch, IFO) = BI: GLITCHES or EVENTS on a detector the
              transient was seen in, NOISE on every other
            - SUB_TYPE: (Batch, IFO) = BI: GlitchType or SourceType value where seen, -1 otherwise
            - GLITCH_TYPE: (Batch, IFO) = BI: GlitchType value where a glitch was
              seen, -1 otherwise
            - SOURCE_TYPE: (Batch, IFO) = BI: SourceType value where an event was
              seen, -1 otherwise
            - COINCIDENT: (Batch,) = B: 1 where two or more detectors are read at the
              transient's time, else 0
        """
        final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
        final_background = ops.cast(ops.stack(batch_backgrounds), "float32")

        transient_gps = as_gps_times([[read.centre for read in reads] for reads in batch_reads])
        moved = np.zeros(len(batch_reads)) if shifts is None else np.asarray(shifts, dtype=np.float64)
        at_time = np.array([[not read.partner for read in reads] for reads in batch_reads])
        seen = at_time & np.array([[read.ifo in segment.seen_in for read in reads]
                                   for segment, reads in zip(batch_segments, batch_reads)])

        def per_detector(value_of) -> np.ndarray:
            values = np.array([value_of(segment) for segment in batch_segments], dtype=np.int32)
            return np.where(seen, values[:, None], -1).astype(np.int32)

        data_label = np.where(
            seen,
            np.array([DataLabel.GLITCHES.value if s.is_glitch else DataLabel.EVENTS.value
                      for s in batch_segments], dtype=np.int32)[:, None],
            DataLabel.NOISE.value,
        ).astype(np.int32)

        return {
            RV.ONSOURCE: final_subarrays,
            RV.OFFSOURCE: final_background,
            RV.START_GPS_TIME: window_spec.onsource_start_gps(
                window_spec.transient_window_start_gps(transient_gps + moved[:, None])
            ),
            RV.TRANSIENT_GPS_TIME: transient_gps,
            RV.DATA_LABEL: data_label,
            RV.SUB_TYPE: per_detector(lambda s: s.kind.value),
            RV.GLITCH_TYPE: per_detector(lambda s: s.kind.value if s.is_glitch else -1),
            RV.SOURCE_TYPE: per_detector(lambda s: s.kind.value if s.is_event else -1),
            RV.COINCIDENT: (np.sum(at_time, axis=1) > 1).astype(np.int32),
        }
    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================


    def get_valid_segments(
        self,
        ifos: List[gf.IFO],
        seed: int,
        group_name: str,
        segment_order: SegmentOrder = None,
        read_span: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Plan split ``group_name`` on ``ifos``: its records, each with its reads.

        The split's members are fixed by the index (`TransientIndex.assign_splits`),
        and their planned order by ``segment_order``: by time, or shortest first
        (SHORTEST_FIRST). ``seed`` changes nothing: each pass's own order (RANDOM) and
        balance are drawn when batches are served (``Passes``). Real events are test
        data: an obtainer of events alone refuses any other split unless it trains on
        real events.

        ``read_span``, (seconds before, seconds after) a read's centre, is the span the
        windows will read, which data quality and partners must hold; it must lie inside
        every record's footprint (``transient_footprint``). None takes each record's
        own span (its start and end).
        """
        if not isinstance(ifos, list):
            ifos = [ifos]
        segment_order = segment_order or self.segment_order

        split = self._split(ifos, group_name)
        if group_name != "test" and not self.train_on_real_events and self._events_only():
            raise ValueError(
                f"Real events are evaluation data, served only in 'test', not {group_name!r}: "
                "to train on them, pass train_on_real_events=True with "
                "coincidence=Coincidence.COINCIDENT."
            )

        # The served state is one split's, on its detectors, for one read span:
        # another rebuilds it, index included, since the index carries the
        # detectors' records.
        self._serve_split(split, order=(segment_order, read_span))
        if self._feature_index is None:
            self.build_feature_index(ifos=ifos)

        # The split's glitches seen in a requested detector, and its events (seen
        # where data quality holds them), each read on every requested detector: its
        # own at its time, the others at partner times.
        candidates = [record for record in self._feature_index.iter(group=group_name)
                      if record.is_event or any(ifo in record.seen_in for ifo in ifos)]
        if read_span is not None:
            self._check_read_span(read_span, candidates)
        records, reads = self._plan_reads(self._passing_data_quality(candidates, ifos, read_span),
                                          ifos, group_name, read_span)
        planned = list(zip(records, reads))
        if segment_order == SegmentOrder.SHORTEST_FIRST:
            planned.sort(key=lambda pair: pair[0].duration)
        self.transient_segments = [record for record, _ in planned]
        self.transient_reads = [plan for _, plan in planned]

        if not planned:
            logger.warning(f"No segments found for group '{group_name}'")
        logger.info(f"TRANSIENT MODE: {len(planned)} segments for '{group_name}'")

        # (records, detectors, 2): the span each detector of each record is read over.
        return np.array([
            [list(_span_about(record, read.centre, read_span)) for read in plan]
            for record, plan in planned
        ], dtype=np.float64).reshape(len(planned), len(ifos), 2)

    def _read_span(self, window_spec: WindowSpec, shift_seconds: float = 0.0) -> Tuple[float, float]:
        """
        The seconds before and after a read's centre that its windows reach: the
        offsource and the padded onsource about the centre sample (within half a sample
        of the centre, ``_read_example``), the resampler's margin and the shared grid's
        lead, and ``shift_seconds`` either way (``RandomShift``).
        """
        rate = float(window_spec.sample_rate_hertz)
        g = store_read.geometry(self.source, rate)
        n_on, n_off = window_spec.num_onsource_samples, window_spec.num_offsource_samples
        before = (n_off + centre_index(n_on) + 0.5 + g.up - 1) / rate + g.margin / g.native
        after = (n_on - centre_index(n_on) + 0.5) / rate + (g.margin + 1) / g.native
        return before + shift_seconds, after + shift_seconds

    @staticmethod
    def _check_read_span(read_span: Tuple[float, float], records: List[TransientSegment]) -> None:
        """
        Refuse, by name and before any fetch, windows whose read could reach past a
        record's footprint (``transient_footprint``): the time its split holds for it.
        """
        if not records:
            return
        before, after = read_span
        footprints = np.array([transient_footprint(record) for record in records])
        times = np.array([float(record.transient_gps_time) for record in records])
        allowed_before = float(np.min(times - footprints[:, 0]))
        allowed_after = float(np.min(footprints[:, 1] - times))
        if before > allowed_before or after > allowed_after:
            raise ValueError(
                f"Transient windows reaching {before:.4f} s before and {after:.4f} s after a "
                f"transient (the offsource, the padded onsource, the resampler's margin and any "
                f"RandomShift) cannot be read: a record's footprint holds only {allowed_before:g} s "
                f"before and {allowed_after:g} s after it. Ask for shorter windows."
            )

    def _shift(self, training: bool) -> Optional["gf.RandomShift"]:
        """The RandomShift that moves this obtainer's training reads, if it has one."""
        from .augmentations import RandomShift
        return next((a for a in self.augmentations if isinstance(a, RandomShift)), None) if training else None

    def _read_example(
        self, record: TransientSegment, reads: Tuple[DetectorRead, ...], window_spec: WindowSpec,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One record's example, (onsource, offsource), each (detectors, samples): on each
        detector, one read through the store of the offsource and then the padded
        onsource, its centre sample the sample of the absolute grid nearest the read's
        centre (``WindowSpec.transient_window_first_sample``). Every sample is exact.

        Raises:
            TransientUnavailable: the source does not deliver strain the timeline holds
                in a read's span; nothing is served, filled or stored in its place.
        """
        n_on, n_off = window_spec.num_onsource_samples, window_spec.num_offsource_samples
        onsource, offsource = [], []
        for read in reads:
            first = int(window_spec.transient_window_first_sample(read.centre)) - n_off
            try:
                strain = store_read.read(self.source, read.ifo, window_spec.sample_rate_hertz, first,
                                         n_off + n_on, stream=self._served_split)
            except TimelineContradiction as contradiction:
                raise TransientUnavailable(record, read, contradiction) from contradiction
            offsource.append(strain.data[:n_off])
            onsource.append(strain.data[n_off:])
        return np.stack(onsource), np.stack(offsource)

    def batcher(self, tree: SeedTree, group: str, **window) -> Callable[[BatchStreams], Dict]:
        """
        A dataset's batcher of split ``group``: ``batch(streams)`` is a function of the
        batch's address alone (``Passes``), read from a fresh copy of this obtainer,
        each example's reads moved by its ``RandomShift`` wherever ``augment`` would flag
        it (``augmented``: training, and no GRID pass). ``window``: the window and batch
        (``sample_rate_hertz``, ``onsource_duration_seconds``, ``crop_duration_seconds``,
        ``offsource_duration_seconds``, ``num_examples_per_batch``, ``ifos``),
        ``sampling_mode``, and ``steps_per_epoch`` (a training epoch's batches; needed
        past the first epoch). ``batch.batches`` is an evaluation split's length,
        floor(N / B), and None for training;
        ``batch.prepare(addresses)`` commits every tile those batches read and returns
        every refusal they would meet (``TransientUnavailable``).
        """
        return deepcopy(self)._batches(tree, group, shift=True, **window)

    def _batches(
            self,
            tree: SeedTree,
            group: str,
            *,
            shift: bool,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            whiten: bool = False,
            crop: bool = False,
            steps_per_epoch: Optional[int] = None,
            sampling_mode: Optional[SamplingMode] = None,
            pool=None,
        ) -> Callable[[BatchStreams], Dict]:
        """
        Split ``group``'s batches, addressed by index (see ``batcher``); ``shift``: move
        training reads by the obtainer's RandomShift. Reads run on ``pool`` (one of the
        store's registered pools when None).
        """
        window_spec = WindowSpec.from_params(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
        )
        rate = window_spec.sample_rate_hertz
        ifos = ensure_list(ifos) or [gf.IFO.L1]
        size = int(num_examples_per_batch or gf.Defaults.num_examples_per_batch)
        training = group == "train"
        # A RandomShift moves a read exactly where ``augment`` flags it: one rule decides both.
        random_shift = self._shift(shift and augmented(group, sampling_mode))
        # The most a read moves, a whole number of samples either way.
        most = 0 if random_shift is None else int(
            random_shift.shift_fraction * window_spec.num_onsource_samples)
        # Before any fetch: a rate or read no store can make, then windows reaching past
        # a record's footprint; then the plan, for the span the windows read.
        n_on, n_off = window_spec.num_onsource_samples, window_spec.num_offsource_samples
        store_read.check(self.source, rate, [(0, n_off + n_on)])
        self.get_valid_segments(ifos, None, group, read_span=self._read_span(window_spec, most / rate))
        passes = Passes.of(self, tree, group, size, training)
        segments, reads = self.transient_segments, self.transient_reads
        pool = pool or thread_pool("transient", min(READ_THREADS, size))

        def planned(streams: BatchStreams):
            """The batch's records, each example's shift in seconds, and its windows' reads (moved)."""
            epoch = streams.epoch or 0
            if training and epoch and steps_per_epoch is None:
                raise ValueError("Batches of epochs after the first need the epoch's steps (steps_per_epoch).")
            g = epoch * (steps_per_epoch or 0) + streams.k if training else streams.k
            indices = passes.batch(g)
            moved = np.zeros(size)
            if random_shift is not None:
                _, fraction = random_shift.draw(streams, size)
                moved = np.round(fraction * most) / rate
            windows = [tuple(read._replace(centre=read.centre + shift_seconds) for read in reads[index])
                       for index, shift_seconds in zip(indices, moved)]
            return indices, moved, windows

        def batch(streams: BatchStreams) -> Dict:
            indices, moved, windows = planned(streams)
            examples = list(pool.map(
                lambda i: self._read_example(segments[indices[i]], windows[i], window_spec), range(size)))
            return self._post_process_batch(
                self._prepare_batch(
                    [onsource for onsource, _ in examples],
                    [offsource for _, offsource in examples],
                    [segments[index] for index in indices],
                    [reads[index] for index in indices],
                    window_spec,
                    moved,
                ),
                rate, window_spec.onsource_duration_seconds, whiten, crop,
            )

        def prepare(addresses: Sequence[BatchStreams]) -> List[TransientUnavailable]:
            """Every refusal the batches at ``addresses`` would meet, each tile committed once."""
            met = {}
            for streams in addresses:
                indices, _, windows = planned(streams)
                for index, example in zip(indices, windows):
                    for read in example:
                        first = int(window_spec.transient_window_first_sample(read.centre)) - n_off
                        try:
                            store_read.ensure(self.source, read.ifo, rate, first, n_off + n_on,
                                              stream=self._served_split)
                        except TimelineContradiction as contradiction:
                            met.setdefault(contradiction.exclusion,
                                           TransientUnavailable(segments[index], read, contradiction))
            return list(met.values())

        batch.batches = passes.batches
        batch.prepare = prepare
        return batch

    def _yield_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            seed: int = None,
            sampling_mode: SamplingMode = SamplingMode.RANDOM,
            whiten: bool = False,
            crop: bool = False,
            group: str = None
        ):
        """
        The obtainer's own batches of split ``group``: batch g of the seed tree rooted
        at ``seed`` (``Passes``), for g = 0, 1, ...; a training split without end, an
        evaluation split once, floor(N / B) batches. Never augmented.
        """
        # With no group, the split served; a first call names one.
        if group is None and self._served_group is None:
            raise ValueError("Name the split to serve: group='train', 'validate' or 'test'.")
        group = group if group is not None else self._served_group
        tree = SeedTree(gf.src.rng.check_root(seed if seed is not None else gf.Defaults.seed))
        size = num_examples_per_batch or gf.Defaults.num_examples_per_batch
        with thread_pool("transient", min(READ_THREADS, size)) as pool:
            batch = self._batches(
                tree, group, shift=False, sample_rate_hertz=sample_rate_hertz,
                onsource_duration_seconds=onsource_duration_seconds, crop_duration_seconds=crop_duration_seconds,
                offsource_duration_seconds=offsource_duration_seconds, num_examples_per_batch=size,
                ifos=ifos, whiten=whiten, crop=crop, pool=pool,
            )
            for k in itertools.count() if batch.batches is None else range(batch.batches):
                yield batch(tree.batch(group, 0, k))


def _span_about(record: TransientSegment, centre: float, read_span: Optional[Tuple[float, float]]):
    """The GPS interval a read of ``record`` centred on ``centre`` covers: ``read_span``,
    or the record's own span (its start and end) when None."""
    if read_span is None:
        return (centre - (record.transient_gps_time - record.start_gps_time),
                centre + (record.end_gps_time - record.transient_gps_time))
    return centre - read_span[0], centre + read_span[1]


@dataclasses.dataclass(frozen=True)
class Passes:
    """
    The order a split's planned records are served in, addressed by index.

    Training concatenates passes without end, and batch g is items gB to (g + 1)B - 1
    of the concatenation, so no batch is partial and a batch may finish one pass and
    begin the next. Pass p is drawn from streams of its own (epoch p, which counts
    passes, not training epochs):

    - balance (``balanced_glitch_types``): every class serves exactly as many items
      as the largest, M: a class of n records serves each one floor(M / n) times, and
      M mod n of them once more, drawn without replacement from ``transient/balance``;
    - order (``SegmentOrder.RANDOM``, and always when balanced): a permutation from
      ``transient/order``.

    An evaluation split is one pass of its records, each once and never balanced,
    in the order of pass 0: floor(N / B) batches (``batches``), the same at every
    epoch, the remainder of N mod B records never served.
    """

    tree: SeedTree
    split: str
    #: Record indices of each class, in planned order: one class unless balanced.
    classes: Tuple[np.ndarray, ...]
    size: int
    training: bool
    randomised: bool

    @classmethod
    def of(cls, obtainer: "TransientDataObtainer", tree: SeedTree, split: str, size: int,
           training: bool) -> "Passes":
        """The passes of ``obtainer``'s planned records for split ``split``, in batches of ``size``."""
        records = obtainer.transient_segments
        if not records:
            raise ValueError(f"No transient records are planned for '{split}': there is nothing to serve.")
        balanced = training and obtainer.balanced_glitch_types
        if balanced:
            # A class is a label's kind: GlitchType and SourceType values overlap as
            # integers (BBH and AIR_COMPRESSOR), so the label keeps them apart.
            keys = [(record.label.name, int(record.kind)) for record in records]
            classes = tuple(np.flatnonzero([k == key for k in keys]) for key in sorted(set(keys)))
        else:
            classes = (np.arange(len(records)),)
        passes = cls(tree, split, classes, int(size), training,
                     balanced or obtainer.segment_order == SegmentOrder.RANDOM)
        if not training:
            if passes.batches == 0:
                raise ValueError(
                    f"The '{split}' split has {len(records)} records, fewer than one batch of {size}: "
                    "an evaluation set serves whole batches only. Ask for smaller batches."
                )
            left = len(records) % size
            if left:
                logger.info(f"The '{split}' split serves {passes.batches} batches of {size}; "
                            f"its last {left} records are never served.")
        return passes

    @property
    def length(self) -> int:
        """The items of one pass."""
        if self.training:
            return len(self.classes) * max(len(members) for members in self.classes)
        return len(self.classes[0])

    @property
    def batches(self) -> Optional[int]:
        """An evaluation split's batches, floor(N / B); None for training, which never ends."""
        return None if self.training else self.length // self.size

    def _stream(self, consumer: str, p: int) -> np.random.Generator:
        return self.tree.stream(self.split, consumer, p)

    def items(self, p: int) -> np.ndarray:
        """The record index of each item of pass ``p``, in order."""
        quota = max(len(members) for members in self.classes)
        if len(self.classes) > 1:
            balance = self._stream("transient/balance", p)
            items = np.concatenate([
                np.concatenate([np.tile(members, quota // len(members)),
                                np.sort(balance.choice(members, quota % len(members), replace=False))])
                for members in self.classes
            ])
        else:
            items = self.classes[0]
        if self.randomised:
            items = items[self._stream("transient/order", p).permutation(len(items))]
        return items

    def batch(self, g: int) -> np.ndarray:
        """The record indices of batch ``g``."""
        if not self.training and g >= self.batches:
            raise IndexError(
                f"Batch {g} is past the {self.batches} batches of the '{self.split}' split: an "
                f"evaluation set is served once, and its {self.length % self.size} record(s) "
                f"short of a batch never are."
            )
        positions = g * self.size + np.arange(self.size)
        passes = positions // self.length
        out = np.empty(self.size, dtype=np.int64)
        for p in np.unique(passes):
            at = passes == p
            out[at] = self.items(int(p))[positions[at] % self.length]
        return out
