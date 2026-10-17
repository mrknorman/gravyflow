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
from typing import Any, List, Dict, Iterator, NamedTuple, Optional, Union, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from numpy.random import default_rng
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
from .partners import PARTNER_GRID_HERTZ, partner_time
from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.dataset.splits import SplitSpec


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
        self, segments: List[TransientSegment], ifos: List[gf.IFO]
    ) -> List[TransientSegment]:
        """
        The records whose span (the most a read may reach about the transient,
        ``_check_request``) lies, on every detector read at their own time
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
            spans = np.array([[s.start_gps_time, s.end_gps_time] for s in records], dtype=np.float64)
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
        self, segments: List[TransientSegment], ifos: List[gf.IFO], group_name: str
    ) -> Tuple[List[TransientSegment], List[Tuple[DetectorRead, ...]]]:
        """
        Each record's reads, one per requested detector in order: its own detectors
        at its own time, every other at a partner time (``partners.partner_time``)
        in that detector's partner segments and the first of the record's partner
        blocks that holds one (``_partner_blocks``), its centre at
        least twice the record's footprint from the transient, keyed by the split's
        fingerprint, the transient's time and the detector. The records with no
        partner time on a detector they need are rejected, and their number is the
        split's ``REJECTED_FOR_NO_PARTNER`` count.

        Under the coincident opt-in every detector is read at the record's time, and
        one the record was not seen in is labelled noise: a record whose span there
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
            reads = []
            for ifo in ifos:
                if ifo in own:
                    if ifo not in record.seen_in:
                        if ifo not in glitches:
                            glitches[ifo] = np.reshape(self.get_glitch_veto_segments(ifo), (-1, 2))
                        if np.any((glitches[ifo][:, 0] < record.end_gps_time)
                                  & (glitches[ifo][:, 1] > record.start_gps_time)):
                            rejected[self.REJECTED_FOR_OTHER_TRANSIENT] += 1
                            break
                    reads.append(DetectorRead(ifo, t, False))
                    continue
                if ifo not in allowed:
                    allowed[ifo] = self._partner_segments(ifo, group_name)
                centre = next((found for found in (
                    partner_time(
                        allowed[ifo], t,
                        before=t - record.start_gps_time, after=record.end_gps_time - t,
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
        
        # Apply balancing
        if self.balanced_glitch_types:
            index.apply_balancing(by="kind")
        
        self._feature_index = index
        return index


    def _prepare_batch(
        self,
        batch_subarrays: list,
        batch_backgrounds: list,
        batch_segments: List[TransientSegment],
        batch_reads: List[Tuple[DetectorRead, ...]],
        window_spec: WindowSpec,
    ) -> dict:
        """
        The batch of ``batch_segments``' examples, each one record's reads
        (``_read_example``).

        Every label is per detector, from the record and that detector's read: a
        partner is noise, with no type.

        Returns:
            Dict with ReturnVariables keys:
            - ONSOURCE: (Batch, IFO, Samples) = BIS
            - OFFSOURCE: (Batch, IFO, Samples) = BIS
            - START_GPS_TIME: (Batch, IFO) = BI, float64: the GPS time of the first
              onsource sample returned after cropping, as for noise
              (``WindowSpec.onsource_start_gps``), exactly: the window centred on the
              sample nearest the detector's read (``WindowSpec.transient_window_start_gps``).
            - TRANSIENT_GPS_TIME: (Batch, IFO) = BI, float64: the centre of each
              detector's window: the transient's time, or a partner's.
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
                window_spec.transient_window_start_gps(transient_gps)
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
    ) -> np.ndarray:
        """
        Get valid segments for TRANSIENT mode.

        Segments come from TransientIndex which contains full TransientSegments.
        The split's members are fixed by the index (`TransientIndex.assign_splits`);
        ``seed`` only orders them (SegmentOrder.RANDOM). Real events are test
        data: an obtainer of events alone refuses any other split unless it
        trains on real events.
        """
        if not isinstance(ifos, list):
            ifos = [ifos]
        
        if not segment_order:
            segment_order = self.segment_order
        
        split = self._split(ifos, group_name)
        if group_name != "test" and not self.train_on_real_events and self._events_only():
            raise ValueError(
                f"Real events are evaluation data, served only in 'test', not {group_name!r}: "
                "to train on them, pass train_on_real_events=True with "
                "coincidence=Coincidence.COINCIDENT."
            )

        # The served state is one split's, on its detectors: another rebuilds
        # it, index included, since the index carries the detectors' records.
        self._serve_split(split)
        if self._feature_index is None:
            self.build_feature_index(ifos=ifos)
        
        # The split's glitches seen in a requested detector, and its events (seen
        # where data quality holds them), each read on every requested detector: its
        # own at its time, the others at partner times.
        records, reads = self._plan_reads(
            self._passing_data_quality(
                [record for record in self._feature_index.iter(group=group_name)
                 if record.is_event or any(ifo in record.seen_in for ifo in ifos)], ifos
            ),
            ifos, group_name,
        )
        planned = list(zip(records, reads))
        if segment_order == SegmentOrder.RANDOM:
            default_rng(seed).shuffle(planned)
        elif segment_order == SegmentOrder.SHORTEST_FIRST:
            planned.sort(key=lambda pair: pair[0].duration)
        self.transient_segments = [record for record, _ in planned]
        self.transient_reads = [plan for _, plan in planned]

        if not planned:
            logger.warning(f"No segments found for group '{group_name}'")
        logger.info(f"TRANSIENT MODE: {len(planned)} segments for '{group_name}'")

        # (records, detectors, 2): the span each detector of each record is read over.
        return np.array([
            [[read.centre - (record.transient_gps_time - record.start_gps_time),
              read.centre + (record.end_gps_time - record.transient_gps_time)] for read in plan]
            for record, plan in planned
        ], dtype=np.float64).reshape(len(planned), len(ifos), 2)


    def _check_request(self, window_spec: WindowSpec) -> None:
        """
        Refuse, by name and before any fetch, a request no read can serve: a rate that
        is not whole hertz, or that the source cannot make, or a read larger than the
        memory cap (``store.read.check``); or windows whose read, the resampler's margin
        and the shared grid's lead counted, could reach past the span a planned record
        may read about its centre (its start and end, which data quality checks). The
        bound holds for every centre: the window's centre sample lies within half a
        sample of it (``_read_example``).
        """
        rate = float(window_spec.sample_rate_hertz)
        g = store_read.geometry(self.source, rate)
        n_on, n_off = window_spec.num_onsource_samples, window_spec.num_offsource_samples
        store_read.check(self.source, rate, [(0, n_off + n_on)])
        if not self.transient_segments:
            return
        before = (n_off + centre_index(n_on) + 0.5 + g.up - 1) / rate + g.margin / g.native
        after = (n_on - centre_index(n_on) + 0.5) / rate + (g.margin + 1) / g.native
        spans = np.array([[r.transient_gps_time - r.start_gps_time, r.end_gps_time - r.transient_gps_time]
                          for r in self.transient_segments])
        allowed_before, allowed_after = spans.min(axis=0)
        if before > allowed_before or after > allowed_after:
            raise ValueError(
                f"Transient windows of {n_off / rate:g} s offsource and {n_on / rate:g} s padded "
                f"onsource cannot be read at {rate:g} Hz: their read, the resampler's margin "
                f"counted, reaches {before:.4f} s before and {after:.4f} s after a transient, past "
                f"the {allowed_before:g} s and {allowed_after:g} s a record may read about it. "
                f"Ask for shorter windows."
            )

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
        Generate onsource/offsource chunks for TRANSIENT mode.
        
        Extracts centered windows around each event/glitch.
        
        Args:
            sample_rate_hertz: Sample rate in Hz
            onsource_duration_seconds: Final onsource window duration
            crop_duration_seconds: Cropping duration on each side (for edge effects)
            offsource_duration_seconds: Background window duration
            num_examples_per_batch: Batch size
            ifos: List of interferometers
            seed: Random seed
            sampling_mode: RANDOM or GRID sampling
        """
        # Create WindowSpec for consistent parameter handling
        window_spec = WindowSpec.from_params(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
        )
        # Before any fetch, and before the plan: a rate no read can make.
        store_read.geometry(self.source, window_spec.sample_rate_hertz)

        # Apply remaining defaults (not in WindowSpec)
        ifos = ensure_list(ifos) or [gf.IFO.L1]
        num_examples_per_batch = num_examples_per_batch or gf.Defaults.num_examples_per_batch
        seed = gf.src.rng.check_root(seed if seed is not None else gf.Defaults.seed)

        # Use shared post-processing from base class
        def _post_process(batch):
            return self._post_process_batch(
                batch, 
                window_spec.sample_rate_hertz, 
                window_spec.onsource_duration_seconds,
                whiten, 
                crop
            )
        
        # With no group, the split served; a first call names one.
        if group is None and self._served_group is None:
            raise ValueError("Name the split to serve: group='train', 'validate' or 'test'.")
        self._serve_group(ifos, seed, group if group is not None else self._served_group)
        # Before any fetch: windows no read of a planned record can serve.
        self._check_request(window_spec)
        if self.rng is None:
            self.rng = default_rng(seed)

        rng = default_rng(seed)
        n_segments = len(self.transient_segments)
        if n_segments == 0:
            raise ValueError("No transient segments available for iteration.")

        # Equal weights: one shuffled order over the records, every pass. Otherwise
        # (apply_balancing), weighted draws with replacement oversample the rare classes
        # in training. An evaluation split is one pass, each record once and unweighted,
        # then the end: its examples are a set, never a sample. Every record drawn is
        # served or refused by name (TransientUnavailable), so no pass is ever empty.
        evaluation = self._served_group != "train"
        weights = np.array([seg.weight for seg in self.transient_segments], dtype=np.float64)
        weighted = not evaluation and not np.allclose(weights, 1.0)
        order = np.arange(n_segments)
        if weighted:
            logger.info(f"Using weighted sampling over {n_segments} transients")
        else:
            rng.shuffle(order)

        def batch(indices, pool):
            """The batch of the records at ``indices``, their examples read together."""
            examples = list(pool.map(
                lambda index: self._read_example(
                    self.transient_segments[index], self.transient_reads[index], window_spec),
                indices,
            ))
            return _post_process(self._prepare_batch(
                [onsource for onsource, _ in examples],
                [offsource for _, offsource in examples],
                [self.transient_segments[index] for index in indices],
                [self.transient_reads[index] for index in indices],
                window_spec,
            ))

        with thread_pool("transient", min(READ_THREADS, num_examples_per_batch)) as pool:
            while True:
                indices = order
                if weighted:
                    indices = rng.choice(n_segments, size=n_segments, replace=True, p=weights / weights.sum())
                # Consecutive batches of the pass; the last holds what is left.
                for start in range(0, len(indices), num_examples_per_batch):
                    yield batch(indices[start:start + num_examples_per_batch], pool)
                if evaluation:
                    return
