"""
Transient mode data acquisition.

This module provides TransientDataObtainer for acquiring data around
specific transient events (GW events, glitches) in TRANSIENT mode.
"""

import dataclasses
import itertools
import logging
from collections import Counter
from typing import Any, List, Dict, Iterator, NamedTuple, Optional, Union, Tuple
from concurrent.futures import as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

import numpy as np
from numpy.random import default_rng
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features.event import EventConfidence
from gravyflow.src.dataset.features.glitch_cache import TransientCache, generate_transient_cache_path
from .cache import DiskCache
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, Coincidence,
    AcquisitionMode, SamplingMode, ObservingRun, label_kinds
)
from gravyflow.src.utils.numerics import ensure_list
from gravyflow.src.utils.resample import resample, whole_hertz
from gravyflow.src.utils.gps import as_gps_times
from gravyflow.src.dataset.features.transient_index import TransientIndex, transient_footprint
from .transient_segment import TransientSegment
from .gwosc_data import within_segments
from .sources import DataSource, fetch_rate
from gravyflow.src.dataset.store.pools import thread_pool
from .partners import partner_time
from gravyflow.src.dataset.config import TransientDefaults, WindowSpec
from gravyflow.src.dataset.splits import SplitSpec


# The transient caches an obtainer may open, one per prefix (`_initialize_transient_cache`):
# records' own detectors in "glitch" and "event", partners in "partner".
TRANSIENT_CACHE_PREFIXES = ("transient", "glitch", "event", "partner")


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
            force_acquisition: bool = False,
            cache_segments: bool = True,
            overrides: dict = None,
            event_types: List[EventConfidence] = None,
            logging_level: int = logging.WARNING,
            augmentations: List = None,  # Training only; None is none
            prefetch_segments: int = 64,  # Higher for smaller TRANSIENT segments
            # Class balancing
            balanced_glitch_types: bool = False,
            # Specific Names
            event_names: List[str] = None,
            split_spec: Optional[SplitSpec] = None,
            train_on_real_events: bool = False,
            source: Optional[DataSource] = None,
            coincidence: Coincidence = Coincidence.SEPARATED,
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
            force_acquisition=force_acquisition,
            cache_segments=cache_segments,
            overrides=overrides,
            logging_level=logging_level,
            augmentations=augmentations,
            prefetch_segments=prefetch_segments,
            split_spec=split_spec,
            source=source,
            coincidence=coincidence,
        )

        self.acquisition_mode = AcquisitionMode.TRANSIENT
        self.train_on_real_events = bool(train_on_real_events)
        
        # Event type filtering
        if event_types is None:
            event_types = [EventConfidence.CONFIDENT]
        self.event_types = event_types
        
        # Class balancing
        self.balanced_glitch_types = balanced_glitch_types

    _segment_cache_maxsize = 128  # TRANSIENT uses larger cache for many small segments

    def _served_state(self) -> Dict[str, Any]:
        caches = {}
        for prefix in TRANSIENT_CACHE_PREFIXES:
            caches[f"_{prefix}_cache"] = None
            caches[f"_{prefix}_cache_logged"] = False
        return {
            **super()._served_state(),
            # TransientIndex - canonical source of truth for transient examples
            "_feature_index": None,
            # The split's records, in serving order, and each record's reads, one
            # per requested detector in order (_plan_reads)
            "transient_segments": [],
            "transient_reads": [],
            # Records not served, by reason: a census of the split (_passing_data_quality,
            # _plan_reads, _reject_for_no_data)
            "_rejections": Counter(),
            # The exact GPS times of records found without a requested detector's
            # strain: not served, nor fetched again, while this state stands
            "_unavailable": set(),
            **caches,
        }

    #: Why a record is not served: its window leaves the usable segments of a
    #: detector it is read on at its own time (an event: of every requested
    #: detector); a requested detector has no finite strain for it; a detector it
    #: was not seen in has no partner time; or, under the coincident opt-in, its
    #: window on a detector it was not seen in meets a catalogued glitch there.
    REJECTED_BY_DATA_QUALITY = "data quality"
    REJECTED_FOR_NO_DATA = "no data"
    REJECTED_FOR_NO_PARTNER = "no partner"
    REJECTED_FOR_OTHER_TRANSIENT = "transient in another detector"

    def _reject_for_no_data(self, segment: TransientSegment) -> None:
        """
        A requested detector has no strain for ``segment``: it is never zero-filled.
        It is counted once, as data quality is, and neither served nor fetched again
        while this split's served state stands (``_unavailable``); a later run fetches
        it again, so a failed fetch never persists as absence.
        """
        if segment.transient_gps_time in self._unavailable:
            return
        self._unavailable.add(segment.transient_gps_time)
        self._rejections[self.REJECTED_FOR_NO_DATA] = len(self._unavailable)
        logger.info(
            f"Transient at GPS {segment.transient_gps_time:.3f} not served (no data); "
            f"rejected: {dict(self._rejections)}"
        )

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """
        As for noise, with ``rejections``: the records of the split not served, by
        reason (``REJECTED_BY_DATA_QUALITY``, ``REJECTED_FOR_NO_DATA``,
        ``REJECTED_FOR_NO_PARTNER``), each record counted once however many passes
        meet it; ``dropped_segments`` is their total.
        """
        return {
            **super().get_acquisition_stats(),
            "dropped_segments": sum(self._rejections.values()),
            "rejections": dict(self._rejections),
        }

    def reset_acquisition_stats(self) -> None:
        super().reset_acquisition_stats()
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
        The records whose acquired span lies, on every detector read at their own
        time (``_own_detectors``), inside one usable segment
        (``get_all_segment_times``: the obtainer's GWOSC flags, so by default no
        hardware injection either). The others are rejected, and their number is the
        split's ``REJECTED_BY_DATA_QUALITY`` count. A partner's data quality is its
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
        from gravyflow.src.dataset.features.glitch_cache import CACHE_SAMPLE_RATE_HERTZ

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
                        sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
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


    def _get_sample_from_cache(
        self,
        cache,
        gps_time: float,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        target_ifos: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Attempt to retrieve a sample from cache (chunked -> disk), as windows of the
        request (``TransientCache.read_windows``).

        Args:
            gps_time: The transient's exact GPS time, its row's identity
            target_ifos: List of IFO names to retrieve (must be present in cache)

        Returns:
            (onsource, offsource, source) where source is 'chunk', 'disk', or 'miss'
        """
        idx = cache.index_of(gps_time)
        if idx is None:
            return None, None, 'miss'

        # Try chunked memory cache first (new fast path)
        # Chunks hold the request's windows already (_load_chunk_around_index).
        if cache._chunk_in_memory:
            result = cache.get_from_chunk(idx, target_ifos=target_ifos)
            if result is not None:
                onsource, offsource, _, _ = result
                return onsource, offsource, 'chunk'

        try:
            ons, off, _, _ = cache.get_batch(
                np.array([idx]),
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration=onsource_duration,
                offsource_duration=offsource_duration,
                target_ifos=target_ifos
            )
            return ons[0], off[0], 'disk'
        except (ValueError, KeyError):
            # Request parameters incompatible or IFO missing
            return None, None, 'miss'

    def _prepare_batch(
        self,
        batch_subarrays: list,
        batch_backgrounds: list,
        batch_segments: List[TransientSegment],
        batch_reads: List[Tuple[DetectorRead, ...]],
        window_spec: WindowSpec,
    ) -> dict:
        """
        The batch of the examples accumulated, each one record's reads.

        Every label is per detector, from the record and that detector's read: a
        partner is noise, with no type.

        Returns:
            Dict with ReturnVariables keys:
            - ONSOURCE: (Batch, IFO, Samples) = BIS
            - OFFSOURCE: (Batch, IFO, Samples) = BIS
            - START_GPS_TIME: (Batch, IFO) = BI, float64: the GPS time of the first
              onsource sample returned after cropping, as for noise
              (``WindowSpec.onsource_start_gps``), the window centred on the
              detector's read (``WindowSpec.transient_window_start_gps``).
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
    @staticmethod
    def _extract_feature_event(
        channels: List[Optional[Tuple[np.ndarray, float]]],
        sample_rate_hertz: float,
        num_onsource_samples: int,
        num_offsource_samples: int,
        transient_gps_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The stored windows of one transient, one row per channel: the onsource
        centred on the transient's sample (``centre_index``), the sample nearest
        ``transient_gps_time`` counted from the channel's first returned sample, and
        the offsource of the samples just before it (the layout
        ``TransientCache.read_windows`` reads).

        ``channels`` holds, per channel, its samples and the GPS time of the first, or
        None. A channel that is None, does not hold both windows, or has any
        non-finite sample in them is a row of NaN: missing, never zero-filled.
        """
        from gravyflow.src.utils.tensor import centre_index

        onsource = np.full((len(channels), num_onsource_samples), np.nan, dtype=np.float32)
        offsource = np.full((len(channels), num_offsource_samples), np.nan, dtype=np.float32)
        for row, channel in enumerate(channels):
            if channel is None:
                continue
            samples, first_gps = channel
            # The transient's sample, from the true first-sample time, in float64.
            transient = int(np.round((float(transient_gps_time) - float(first_gps)) * sample_rate_hertz))
            start = transient - centre_index(num_onsource_samples)
            off_start = start - num_offsource_samples
            if off_start < 0 or start + num_onsource_samples > len(samples):
                continue
            on, off = samples[start:start + num_onsource_samples], samples[off_start:start]
            if np.all(np.isfinite(on)) and np.all(np.isfinite(off)):
                onsource[row], offsource[row] = on, off
        return onsource, offsource



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


    @staticmethod
    def _cache_prefix(record: TransientSegment, read: DetectorRead) -> str:
        """The cache a read is stored in: partners', or its record's kind's."""
        if read.partner:
            return "partner"
        return "event" if record.is_event else "glitch"

    def _cached_windows(
        self, caches: Dict[str, Any], record: TransientSegment, reads: Tuple[DetectorRead, ...],
        window_spec: WindowSpec,
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Each read's (onsource, offsource) row from its cache, or None where the cache
        does not hold its detector's strain (a row without it, or no row): a miss,
        fetched by ``_fetch_windows``.
        """
        windows = []
        for read in reads:
            onsource, offsource, source = None, None, "miss"
            if not self.force_acquisition:
                onsource, offsource, source = self._get_sample_from_cache(
                    caches[self._cache_prefix(record, read)],
                    read.centre,
                    window_spec.sample_rate_hertz,
                    window_spec.total_onsource_duration_seconds,
                    window_spec.offsource_duration_seconds,
                    target_ifos=[read.ifo.name],
                )
            held = (source != "miss" and onsource is not None
                    and np.all(np.isfinite(onsource)) and np.all(np.isfinite(offsource)))
            windows.append((np.asarray(onsource)[0], np.asarray(offsource)[0]) if held else None)
        return windows

    def _download(self, spans: Dict[Tuple, Tuple[gf.IFO, float, float]], sample_rate_hertz: float) -> Dict:
        """
        Each span's strain, {key: (samples, GPS time of the first)} at
        ``sample_rate_hertz``, fetched from the obtainer's source in parallel at the
        native rate it needs (``fetch_rate``); a span whose strain could not be had is
        left out (never zeros). ``spans`` is {key: (ifo, start, end)}.
        """
        native_rate = fetch_rate(self.source, sample_rate_hertz)

        def download(key):
            ifo, start, end = spans[key]
            try:
                strain = self.source.fetch(ifo, native_rate, (start, end))
                # The stored windows end 0.2 s inside the span, past the resampler's
                # margin (under 0.02 s from any public rate), so each stored sample is exact.
                return key, resample(strain.data, strain.t0, strain.sample_rate_hertz, sample_rate_hertz)
            except Exception as error:
                logger.debug(f"No {ifo.name} strain over GPS {start:.3f}-{end:.3f}: {error}")
                return key, None

        downloaded = {}
        if not spans:
            return downloaded
        workers = min(32, len(spans))
        with thread_pool("download", workers) as executor:
            futures = [executor.submit(download, key) for key in spans]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading ({workers}w)",
                               unit="read", leave=False, dynamic_ncols=True):
                key, channel = future.result()
                if channel is not None:
                    downloaded[key] = channel
        logger.info(f"Downloaded {len(downloaded)}/{len(spans)} detector windows ({workers} workers)")
        return downloaded

    def _store_read(
        self, cache, record: TransientSegment, read: DetectorRead, channel, download_rate: float,
        window_spec: WindowSpec,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Store one read's downloaded strain in its cache, as a row of the universal
        detectors holding only the read's (a detector not held is NaN, ``_fill_row``),
        and return its (onsource, offsource) windows of the request; None when the
        strain does not hold them, finite.
        """
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION,
        )
        from .transient_segment import ifos_to_bitmask

        universal = list(TransientDefaults.UNIVERSAL_IFOS)
        row = universal.index(read.ifo.name)
        channels = [None] * len(universal)
        channels[row] = channel
        onsource_full, offsource_full = self._extract_feature_event(
            channels, download_rate,
            int(CACHE_ONSOURCE_DURATION * download_rate), int(CACHE_OFFSOURCE_DURATION * download_rate),
            read.centre,
        )
        if not (np.all(np.isfinite(onsource_full[row])) and np.all(np.isfinite(offsource_full[row]))):
            return None
        if read.partner:
            label, metadata = -1, {"data_label": DataLabel.NOISE.value, "seen_in": 0}
        else:
            label = record.kind.value if record.is_glitch else -1
            metadata = {
                "data_label": DataLabel.GLITCHES.value if record.is_glitch else DataLabel.EVENTS.value,
                "seen_in": ifos_to_bitmask(record.seen_in) if record.seen_in else 0,
                "snr": record.snr,
                "peak_freq": record.peak_freq,
                "duration": record.duration,
                "ml_confidence": record.ml_confidence,
                "mass1": record.mass1,
                "mass2": record.mass2,
                "distance": record.distance,
                "p_astro": record.p_astro,
            }
        try:
            cache.append_single(onsource_full, offsource_full, read.centre, label, metadata=metadata)
        except Exception as error:
            logger.warning(f"Failed to store {read.ifo.name} at GPS {read.centre:.3f} in the cache: {error}")
        onsource, offsource = TransientCache.read_windows(
            onsource_full[row:row + 1], offsource_full[row:row + 1], download_rate,
            window_spec.sample_rate_hertz, window_spec.total_onsource_duration_seconds,
            window_spec.offsource_duration_seconds,
        )
        return onsource[0], offsource[0]

    def _download_rate(self, window_spec: WindowSpec) -> float:
        """
        The rate a request's strain is downloaded and stored at: the higher of its own
        and the cache's (a cache can downsample, never upsample).

        Refuses, by name, a request no read could serve: a rate that is not whole hertz,
        that the obtainer's source does not hold (``fetch_rate``), or whose windows reach
        past the stored strain, the resampler's margin counted
        (``TransientCache.check_request``). Called once, as a request is configured,
        before any cache is opened or any strain fetched.
        """
        from gravyflow.src.dataset.features.glitch_cache import CACHE_SAMPLE_RATE_HERTZ

        download_rate = max(whole_hertz(window_spec.sample_rate_hertz), CACHE_SAMPLE_RATE_HERTZ)
        fetch_rate(self.source, download_rate)
        TransientCache.check_request(
            download_rate, window_spec.sample_rate_hertz,
            window_spec.total_onsource_duration_seconds, window_spec.offsource_duration_seconds,
        )
        return download_rate

    def _fetch_windows(
        self, caches: Dict[str, Any], pending: List[Tuple[int, List]], window_spec: WindowSpec,
        download_rate: float,
    ):
        """
        Complete the windows of the ``pending`` records ([(index, windows)], each
        read's cached window or None): download every missing read's span in
        parallel at ``download_rate`` (``_download_rate``), store it in its cache, and
        yield (index, onsource, offsource) for each record whose every read now holds
        strain. A record with a read that does not is rejected for no data
        (``_reject_for_no_data``).
        """
        def key(record, read):
            return self._cache_prefix(record, read), read.ifo.name, read.centre

        spans = {}
        for index, windows in pending:
            record = self.transient_segments[index]
            before = record.transient_gps_time - record.start_gps_time
            after = record.end_gps_time - record.transient_gps_time
            for read, window in zip(self.transient_reads[index], windows):
                if window is None:
                    spans[key(record, read)] = (read.ifo, read.centre - before, read.centre + after)
        downloaded = self._download(spans, download_rate)

        for index, windows in pending:
            record = self.transient_segments[index]
            for d, read in enumerate(self.transient_reads[index]):
                if windows[d] is None and key(record, read) in downloaded:
                    windows[d] = self._store_read(
                        caches[self._cache_prefix(record, read)], record, read,
                        downloaded[key(record, read)], download_rate, window_spec,
                    )
            if any(window is None for window in windows):
                self._reject_for_no_data(record)
                continue
            yield index, np.stack([on for on, _ in windows]), np.stack([off for _, off in windows])
        for cache in caches.values():
            cache.flush_write_buffer()
    def _initialize_transient_cache(
        self,
        ifos: List,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        offsource_duration_seconds: float,
        enable_chunked_mode: bool = True,  # NEW: Enable chunked cache by default
        chunk_size: int = 5000,  # Number of samples per memory chunk
        prefix: str = "transient"
    ):
        """
        Initialize or retrieve the transient cache of one prefix: events, glitches or partners.
        
        Handles cache path generation, file creation, validation, and
        optional chunked memory caching for improved performance.
        
        Args:
            ifos: List of IFOs to cache
            sample_rate_hertz: Requested sample rate
            onsource_duration_seconds: Requested onsource duration
            offsource_duration_seconds: Requested offsource duration
            enable_chunked_mode: If True, load cache chunks into memory for faster access
            chunk_size: Number of samples per memory chunk (default 5000, ~1-2GB)
            prefix: Cache type prefix (``TRANSIENT_CACHE_PREFIXES``: "transient", "glitch", "event", "partner")
            
        Returns:
            Tuple of (cache, cache_sample_rate) - TransientCache instance and its sample rate
        """
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        # Determine data directory (None = the user cache, gwosc_data.user_cache_dir)
        data_dir = getattr(self, 'data_directory', None)
        
        # A cache file per prefix (glitches, events and partners) and source: one format
        cache_path = generate_transient_cache_path(data_directory=data_dir, prefix=prefix, source=self.source)
        
        # Manage multiple caches (glitch, event)
        cache_attr = f"_{prefix}_cache"
        
        # Reuse existing cache if path matches (preserves GPS index)
        if getattr(self, cache_attr) is None or getattr(self, cache_attr).path != cache_path:
            setattr(self, cache_attr, DiskCache(cache_path, mode='a'))
        
        cache = getattr(self, cache_attr)
        
        # Validate cache compatibility (sample rate and duration only)
        # IFO validation is not needed - All caches use H1+L1+V1 format
        if cache.exists:
            try:
                cache.validate_request(sample_rate_hertz, onsource_duration_seconds, offsource_duration_seconds)
                
                # Use a unique logging attribute per cache
                logged_attr = f"_{prefix}_cache_logged"
                if not getattr(self, logged_attr):
                    meta = cache.get_metadata()
                    logger.info(f"{prefix.capitalize()} Cache: {meta['num_glitches']} items at {CACHE_SAMPLE_RATE_HERTZ}Hz")
                    setattr(self, logged_attr, True)
                    
                # Enable chunked mode for faster cache hits (if not already enabled)
                if enable_chunked_mode and not cache._chunk_in_memory:
                    meta = cache.get_metadata()
                    if meta['num_glitches'] > 0:
                        cache.enable_chunked_mode(
                            chunk_size=chunk_size,
                            sample_rate_hertz=sample_rate_hertz,
                            onsource_duration=onsource_duration_seconds,
                            offsource_duration=offsource_duration_seconds
                        )
            except ValueError as e:
                logger.warning(f"{prefix.capitalize()} cache incompatible: {e}. Resetting.")
                cache.reset()
        
        # Initialize cache file if needed (or if reset)
        # All caches use H1+L1+V1 format regardless of requested IFOs
        if not cache.exists:
            num_cache_onsource = int(CACHE_ONSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
            num_cache_offsource = int(CACHE_OFFSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
            universal_ifos = list(TransientDefaults.UNIVERSAL_IFOS)
            
            cache.initialize_file(
                sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
                onsource_duration=CACHE_ONSOURCE_DURATION,
                offsource_duration=CACHE_OFFSOURCE_DURATION,
                ifo_names=universal_ifos,
                num_ifos=len(universal_ifos),
                onsource_samples=num_cache_onsource,
                offsource_samples=num_cache_offsource
            )
            logger.info(f"Created {prefix.capitalize()} Cache: {cache_path}")
        
        return cache, CACHE_SAMPLE_RATE_HERTZ

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
        # Before any cache is opened or any strain fetched: a request refused as a
        # cache is read would meet the handlers that reset the cache or fetch a miss.
        download_rate = self._download_rate(window_spec)

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
        if self.rng is None:
            self.rng = default_rng(seed)
        
        # The caches the split's reads are stored in. A sparse request (few records)
        # reads rows from disk rather than loading chunks into memory.
        is_sparse_request = (len(self.transient_segments) < 100) or (self.event_names is not None)
        caches = {
            prefix: self._initialize_transient_cache(
                ifos,
                window_spec.sample_rate_hertz,
                window_spec.total_onsource_duration_seconds,
                window_spec.offsource_duration_seconds,
                enable_chunked_mode=not is_sparse_request,
                prefix=prefix,
            )[0]
            for prefix in sorted({
                self._cache_prefix(record, read)
                for record, reads in zip(self.transient_segments, self.transient_reads)
                for read in reads
            })
        }

        rng = default_rng(seed)
        n_segments = len(self.transient_segments)
        if n_segments == 0:
            raise ValueError("No transient segments available for iteration.")

        # Each pass draws among the records still servable, so a pass that serves
        # nothing has found every record it drew unavailable: the servable records
        # shrink with each such pass, and the split is served or refused, never spun.
        # Equal weights: one shuffled order over the records, kept for cache hits.
        # Otherwise (apply_balancing), weighted draws with replacement oversample the
        # rare classes in training. An evaluation split is one pass, each record once
        # and unweighted, then the end: its examples are a set, never a sample.
        evaluation = self._served_group != "train"
        weights = np.array([seg.weight for seg in self.transient_segments], dtype=np.float32)
        weighted = not evaluation and not np.allclose(weights, 1.0)
        order = np.arange(n_segments)
        if weighted:
            logger.info(f"Using weighted sampling over {n_segments} transients")
        else:
            rng.shuffle(order)

        batch_subarrays, batch_backgrounds, batch_indices = [], [], []
        served = [0]  # Examples served this pass

        def add(index, onsource, offsource):
            """Add one example; the batch, once full."""
            served[0] += 1
            batch_subarrays.append(onsource)
            batch_backgrounds.append(offsource)
            batch_indices.append(index)
            if len(batch_subarrays) < num_examples_per_batch:
                return None
            batch = self._prepare_batch(
                batch_subarrays, batch_backgrounds,
                [self.transient_segments[i] for i in batch_indices],
                [self.transient_reads[i] for i in batch_indices],
                window_spec,
            )
            batch_subarrays.clear()
            batch_backgrounds.clear()
            batch_indices.clear()
            return _post_process(batch)

        def fetch(pending):
            """The batches the pending records complete, each cached before it is served."""
            for index, onsource, offsource in self._fetch_windows(caches, pending, window_spec, download_rate):
                batch = add(index, onsource, offsource)
                if batch is not None:
                    # A batch served is cached first: the caller may never resume.
                    for cache in caches.values():
                        cache.flush_write_buffer()
                    yield batch

        while True:
            served[0] = 0
            servable = np.array([
                index for index in order
                if self.transient_segments[index].transient_gps_time not in self._unavailable
            ], dtype=int)
            if len(servable) == 0:
                raise ValueError(
                    f"None of the {n_segments} transients of split '{self._served_group}' "
                    f"can be served on {[ifo.name for ifo in ifos]}: each was rejected for "
                    f"no data (rejections: {dict(self._rejections)})."
                )
            segment_indices = servable
            if weighted:
                probs = weights[servable] / weights[servable].sum()
                segment_indices = rng.choice(servable, size=len(servable), replace=True, p=probs)

            # Cache hits are served at once, in order; records with a read the
            # caches do not hold are fetched a batch at a time.
            pending = []
            for index in tqdm(segment_indices, desc="Assembling batches", unit="seg",
                              leave=False, dynamic_ncols=True):
                record = self.transient_segments[index]
                if record.transient_gps_time in self._unavailable:  # found unavailable this pass
                    continue
                windows = self._cached_windows(caches, record, self.transient_reads[index], window_spec)
                if all(window is not None for window in windows):
                    batch = add(index, np.stack([on for on, _ in windows]),
                                np.stack([off for _, off in windows]))
                    if batch is not None:
                        yield batch
                    continue
                pending.append((index, windows))
                if len(pending) >= num_examples_per_batch:
                    yield from fetch(pending)
                    pending = []
            yield from fetch(pending)

            # The partial batch left at the end of a pass
            if batch_subarrays:
                batch = self._prepare_batch(
                    batch_subarrays, batch_backgrounds,
                    [self.transient_segments[i] for i in batch_indices],
                    [self.transient_reads[i] for i in batch_indices],
                    window_spec,
                )
                batch_subarrays.clear()
                batch_backgrounds.clear()
                batch_indices.clear()
                yield _post_process(batch)
            # An evaluation pass ends once it has served. One that served nothing
            # found every record it drew unavailable, so the next check refuses it.
            if evaluation and served[0]:
                return
