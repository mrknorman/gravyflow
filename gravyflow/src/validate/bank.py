from typing import Any, Dict, List, Tuple, Optional, Union
import heapq
import logging
import math
import random
import sys
from copy import deepcopy

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

import gravyflow as gf
from .config import ValidationConfig
from .utils import calculate_far_score_thresholds, roc_curve_and_auc, _extract_sample_data

logger = logging.getLogger(__name__)


class ValidationBank:
    """Generate/store validation data: noise (FAR), injections (efficiency), and real events."""

    _NOISE_OUTPUTS = (gf.ReturnVariables.WHITENED_ONSOURCE, gf.ReturnVariables.GPS_TIME)
    _INJ_OUTPUTS = (
        gf.ReturnVariables.WHITENED_ONSOURCE,
        gf.ReturnVariables.WHITENED_INJECTIONS,
        gf.ReturnVariables.INJECTION_MASKS,
        gf.ReturnVariables.GPS_TIME,
        gf.ReturnVariables.CENTRAL_TIME,
        gf.ScalingTypes.SNR,
        gf.ScalingTypes.HPEAK,
        gf.ScalingTypes.HRSS,
        gf.WaveformParameters.MASS_1_MSUN,
        gf.WaveformParameters.MASS_2_MSUN,
    )

    def __init__(
        self,
        model,
        dataset_args: dict,
        config: Optional[ValidationConfig] = None,
        heart: Optional[gf.Heart] = None,
    ):
        self.model, self.dataset_args = model, dataset_args
        self.config, self.heart = config or ValidationConfig(), heart

        self.model, self.dataset_args = model, dataset_args
        self.config, self.heart = config or ValidationConfig(), heart

        self.__dict__.update(
            {
                k: None
                for k in (
                    "far_scores",
                    "noise_gps_times",
                    "_worst_false_positives",
                    "snrs",
                    "scores",
                    "injection_masks",
                    "mass1",
                    "mass2",
                    "gps_times",
                    "central_times",
                    "hpeak",
                    "hrss",
                    "_worst_false_negatives",
                    "real_events",
                )
            }
        )

    @staticmethod
    def _extract_scores(predictions) -> np.ndarray:
        """Extract scores from model predictions, handling common output shapes."""
        p = np.asarray(predictions)
        if p.ndim == 2:
            return p[:, 1] if p.shape[1] == 2 else p[:, 0]
        return p.reshape(-1)

    @staticmethod
    def _as_1d(arr, n: int) -> np.ndarray:
        """Flatten to 1D and crop to length n."""
        a = np.asarray(arr)
        return (a.reshape(-1) if a.ndim > 1 else a)[:n]

    def _beat(self) -> None:
        if self.heart:
            self.heart.beat()

    def _prefetch_valid_segments(self) -> None:
        """Pre-fetch valid segments before deepcopy so any internal cache persists."""
        noise_obt = self.dataset_args.get("noise_obtainer")
        obt = getattr(noise_obt, "ifo_data_obtainer", None) if noise_obt else None
        if not obt:
            return
        try:
            if getattr(obt, "valid_segments", None) is not None:
                return
            logger.info("Pre-fetching valid segments for cache consistency...")
            ifos = getattr(noise_obt, "ifos", None)
            if ifos:
                obt.get_valid_segments(
                    ifos=ifos,
                    seed=self.dataset_args.get("seed", 42),
                    group_name=self.dataset_args.get("group", "test"),
                )
        except Exception as e:
            logger.warning(f"Failed to pre-fetch segments: {e}")

    def _build_dataset(self, dataset_args: dict, num_batches: int, outputs: tuple) -> gf.Dataset:
        dataset_args.update(
            num_examples_per_batch=self.config.batch_size,
            steps_per_epoch=num_batches,
            output_variables=list(outputs),
            group="test",
        )
        return gf.Dataset(**dataset_args)

    def generate_noise(self) -> None:
        """Generate noise scores for FAR calculation, plus worst false positives."""
        self._prefetch_valid_segments()
        cfg = self.config
        num_batches = math.ceil(cfg.num_examples / cfg.batch_size)

        dataset_args = deepcopy(self.dataset_args)
        dataset_args["waveform_generators"] = []  # noise-only
        dataset = self._build_dataset(dataset_args, num_batches, self._NOISE_OUTPUTS)

        score_chunks: List[np.ndarray] = []
        gps_list: List[float] = []
        fp_heap: List[Tuple[float, float, Any]] = []
        keep = cfg.num_worst_per_bin * 5
        gps_key = gf.ReturnVariables.GPS_TIME.name

        for b in tqdm(range(num_batches), desc="Scoring noise", unit="batch"):
            self._beat()
            x, y = dataset[b]
            y = y or {}

            scores = self._extract_scores(self.model.predict_on_batch(x))
            score_chunks.append(scores)

            if gps_key in y:
                gps_list.extend(np.asarray(y[gps_key]).reshape(-1).tolist())

            for i, s in enumerate(scores):
                sv, tie = float(s), random.random()
                if len(fp_heap) < keep:
                    heapq.heappush(fp_heap, (sv, tie, _extract_sample_data(x, y, i, sv)))
                else:
                    lo_s, lo_t = fp_heap[0][0], fp_heap[0][1]
                    if sv > lo_s or (sv == lo_s and tie > lo_t):
                        heapq.heapreplace(fp_heap, (sv, tie, _extract_sample_data(x, y, i, sv)))

        self.far_scores = np.concatenate(score_chunks, axis=0)
        self.noise_gps_times = np.asarray(gps_list) if gps_list else None
        self._worst_false_positives = [t[2] for t in sorted(fp_heap, key=lambda x: -x[0])]

    def generate(self) -> None:
        """Generate injection scores for efficiency calculation (injection_chance=1, Uniform SNR)."""
        cfg = self.config
        min_snr, max_snr = cfg.snr_range

        dataset_args = deepcopy(self.dataset_args)
        if "waveform_generators" not in dataset_args:
            raise ValueError("dataset_args must contain waveform_generators")

        wfs = dataset_args["waveform_generators"]
        dataset_args["waveform_generators"] = wfs if isinstance(wfs, list) else [wfs]
        for gen in dataset_args["waveform_generators"]:
            gen.scaling_method.value = gf.Distribution(
                min_=min_snr, max_=max_snr, type_=gf.DistributionType.UNIFORM
            )
            gen.injection_chance = 1.0

        num_batches = math.ceil(cfg.num_examples / cfg.batch_size)
        dataset = self._build_dataset(dataset_args, num_batches, self._INJ_OUTPUTS)

        acc = {k: [] for k in ("snrs", "scores", "mass1", "mass2", "gps", "ct", "hpeak", "hrss")}
        n_bins = int((max_snr - min_snr) / cfg.snr_bin_width)
        worst: List[List[Tuple[float, int, Any]]] = [[] for _ in range(n_bins)]

        keys = {
            "snrs": "SNR",
            "mass1": gf.WaveformParameters.MASS_1_MSUN.name,
            "mass2": gf.WaveformParameters.MASS_2_MSUN.name,
            "gps": gf.ReturnVariables.GPS_TIME.name,
            "ct": gf.ReturnVariables.CENTRAL_TIME.name,
            "hpeak": gf.ScalingTypes.HPEAK.name,
            "hrss": gf.ScalingTypes.HRSS.name,
        }

        for b in tqdm(range(num_batches), desc="Scoring injections", unit="batch"):
            self._beat()
            x, y = dataset[b]
            y = y or {}

            batch_scores = self._extract_scores(self.model.predict_on_batch(x))
            n = len(batch_scores)

            batch = {
                "snrs": self._as_1d(y.get(keys["snrs"], np.zeros(n)), n),
                "mass1": self._as_1d(y.get(keys["mass1"], np.zeros(n)), n),
                "mass2": self._as_1d(y.get(keys["mass2"], np.zeros(n)), n),
                "gps": self._as_1d(y.get(keys["gps"], np.zeros(n)), n),
                "hpeak": self._as_1d(y.get(keys["hpeak"], np.zeros(n)), n),
                "hrss": self._as_1d(y.get(keys["hrss"], np.zeros(n)), n),
            }

            ct = np.asarray(y.get(keys["ct"], np.zeros(n)))
            if ct.ndim > 1 and ct.shape[-1] > 1:
                ct = ct.mean(axis=-1)
            batch["ct"] = self._as_1d(ct, n)
            batch["scores"] = np.asarray(batch_scores).reshape(-1)

            for k, v in batch.items():
                acc[k].extend(np.asarray(v).tolist())

            for i, (sv, sc) in enumerate(zip(batch["snrs"], batch_scores)):
                score_val = float(sc)
                bin_idx = min(int((float(sv) - min_snr) / cfg.snr_bin_width), n_bins - 1)
                bin_idx = max(0, bin_idx)

                heap = worst[bin_idx]
                if len(heap) < cfg.num_worst_per_bin:
                    heapq.heappush(
                        heap,
                        (-score_val, b * cfg.batch_size + i, _extract_sample_data(x, y, i, score_val)),
                    )
                elif score_val < -heap[0][0]:
                    heapq.heapreplace(
                        heap,
                        (-score_val, b * cfg.batch_size + i, _extract_sample_data(x, y, i, score_val)),
                    )

        self.snrs = np.asarray(acc["snrs"])
        self.scores = np.asarray(acc["scores"])
        self.injection_masks = np.ones_like(self.scores)
        self.mass1, self.mass2 = np.asarray(acc["mass1"]), np.asarray(acc["mass2"])
        self.gps_times, self.central_times = np.asarray(acc["gps"]), np.asarray(acc["ct"])
        self.hpeak, self.hrss = np.asarray(acc["hpeak"]), np.asarray(acc["hrss"])

        self._worst_false_negatives = {}
        for i, heap in enumerate(worst):
            start = min_snr + i * cfg.snr_bin_width
            key = f"{start:.0f}-{(start + cfg.snr_bin_width):.0f}"
            self._worst_false_negatives[key] = [t[2] for t in sorted(heap, key=lambda x: -x[0])]

    def generate_real_events(self, observing_runs: List = None) -> None:
        """Score real GW events from GWTC catalogs via TransientObtainer."""
        from gravyflow.src.dataset.features.event import EventType, get_events_with_params
        from gravyflow.src.dataset.conditioning.whiten import whiten

        if observing_runs is None:
            observing_runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]

        logger.info("Fetching real GW events from GWTC catalogs...")
        events = get_events_with_params(
            observing_runs, event_types=[EventType.CONFIDENT, EventType.MARGINAL]
        )
        for e in events:
            e["event_type"] = (
                "MARGINAL" if "marginal" in str(e.get("catalog", "")).lower() else "CONFIDENT"
            )

        seen, unique = set(), []
        for e in events:
            k = round(e["gps"], 1)
            if k not in seen:
                seen.add(k)
                unique.append(e)
        unique.sort(key=lambda x: x["gps"])

        conf = sum(1 for e in unique if e["event_type"] == "CONFIDENT")
        logger.info(
            f"Found {len(unique)} unique events ({conf} confident, {len(unique) - conf} marginal)"
        )

        noise_obt = self.dataset_args.get("noise_obtainer")
        if not noise_obt:
            logger.warning("No noise_obtainer in dataset_args - cannot score events")
            self.real_events = unique
            return

        ifos = getattr(noise_obt, "ifos", [gf.IFO.H1, gf.IFO.L1])
        
        # Filter events by detector availability - only keep events where requested IFOs were active
        # Network format: "HL", "HLV", "LV", etc. - single letter per detector
        ifo_letters = set(ifo.name[0] for ifo in ifos)  # e.g., {"H", "L"}
        logger.info(f"Required detectors: {ifo_letters}")
        
        filtered = []
        no_network_count = 0
        for e in unique:
            network = e.get("network", "")
            if not network:
                # No network info - log warning and include anyway
                no_network_count += 1
                logger.debug(f"No network info for {e['name']} (GPS={e['gps']:.1f}, run={e.get('observing_run', '?')})")
                filtered.append(e)
                continue
            # Check if all requested IFOs were active for this event
            if all(letter in network for letter in ifo_letters):
                filtered.append(e)
            else:
                logger.info(f"Skipping {e['name']}: network={network}, need={ifo_letters}")
        
        if no_network_count > 0:
            logger.warning(f"{no_network_count} events have no network info in catalog (will attempt fetch)")
        
        skipped = len(unique) - len(filtered)
        if skipped > 0:
            logger.info(f"Filtered out {skipped} events (detectors not in network)")
        unique = filtered
        
        names = [e["name"] for e in unique]
        logger.info(f"Creating TransientObtainer for {len(names)} events...")

        sample_rate = self.dataset_args.get("sample_rate_hertz", 2048.0)
        on_dur = self.dataset_args.get("onsource_duration_seconds", 1.0)
        off_dur = self.dataset_args.get("offsource_duration_seconds", 16.0)

        try:
            obt = gf.TransientObtainer(
                ifo_data_obtainer=gf.IFODataObtainer(
                    observing_runs=observing_runs,
                    data_quality=gf.DataQuality.BEST,
                    data_labels=[gf.DataLabel.EVENTS],
                ),
                ifos=ifos,
                event_names=names,
            )

            gen = obt(
                sample_rate_hertz=sample_rate,
                onsource_duration_seconds=on_dur,
                offsource_duration_seconds=off_dur,
                num_examples_per_batch=min(len(names), 256),
                scale_factor=gf.Defaults.scale_factor,
            )

            logger.info("Phase 1: Collecting event data...")
            all_ons, all_offs, all_gps = [], [], []
            collected = 0
            missing_segment_count = 0

            # Suppress gwdatafind "Missing segments" warnings - these are expected
            # for events with data gaps. We'll count and log them separately.
            import warnings
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.filterwarnings("always", message="Missing segments")
                
                for onsource, offsource, gps_times, _labels in gen:
                    if collected >= len(names):
                        break
                    self._beat()

                    onsource, offsource, gps_times = map(np.asarray, (onsource, offsource, gps_times))
                    batch_len = onsource.shape[0] if onsource.ndim > 2 else 1

                    for j in range(batch_len):
                        if collected >= len(names):
                            break
                        all_ons.append(np.array(onsource[j : j + 1] if batch_len > 1 else onsource))
                        all_offs.append(np.array(offsource[j : j + 1] if batch_len > 1 else offsource))
                        all_gps.append(float(gps_times[j, 0] if gps_times.ndim > 1 else gps_times[j]))
                        collected += 1

                    if collected and collected % 20 == 0:
                        logger.info(f"Data collection: {collected}/{len(names)} events")
                
                # Count missing segment warnings
                missing_segment_count = sum(1 for w in caught_warnings if "Missing segments" in str(w.message))
            
            if missing_segment_count > 0:
                logger.info(f"Note: {missing_segment_count} events had data gaps (expected for some O1/O2 events)")

            logger.info(f"Collected {len(all_ons)} events for batch scoring")

            if all_ons:
                logger.info("Phase 2: Running batched model inference...")
                x = {
                    "ONSOURCE": np.concatenate(all_ons, axis=0),
                    "OFFSOURCE": np.concatenate(all_offs, axis=0),
                }
                scores = self._extract_scores(self.model.predict(x, verbose=0))
                logger.info(f"Batch inference complete: {len(scores)} scores")

                logger.info("Phase 3: Matching scores to events...")
                for i, (gps, score) in enumerate(zip(all_gps, scores)):
                    ev = next((e for e in unique if abs(e["gps"] - gps) < 20.0), None)
                    if not ev:
                        continue
                    ev["score"], ev["status"] = float(score), "scored"
                    try:
                        whitened = whiten(all_ons[i], all_offs[i], sample_rate_hertz=sample_rate)
                        ev["whitened_strain"] = np.array(
                            gf.crop_samples(
                                whitened,
                                sample_rate_hertz=sample_rate,
                                onsource_duration_seconds=on_dur,
                            )
                        )[0]
                    except Exception as e:
                        logger.warning(f"Whitening failed for event at {gps}: {e}")

        except StopIteration:
            pass
        except Exception as e:
            logger.warning(f"TransientObtainer failed: {e}")
            logger.info("Events stored without scores")

        if self.far_scores is not None and len(self.far_scores) > 0:
            thresholds = calculate_far_score_thresholds(
                self.far_scores,
                self.dataset_args.get("onsource_duration_seconds", 1.0),
                np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
            )
            for ev in unique:
                if ev.get("score") is None:
                    continue
                for far in sorted(thresholds):
                    if ev["score"] >= thresholds[far][1]:
                        ev["min_far"] = far
                        break

        self.real_events = unique
        logger.info(
            f"Real events scoring complete: "
            f"{sum(1 for e in unique if e.get('score') is not None)}/{len(unique)} scored"
        )

    def get_efficiency_data(self) -> Dict:
        """Return injection scatter + binned means/stds and a fitted sigmoid curve."""
        if self.snrs is None:
            raise ValueError("Must call generate() first")

        cfg = self.config
        snrs, scores = self.snrs, self.scores
        min_snr, max_snr = cfg.snr_range
        num_bins = max(1, int((max_snr - min_snr) / cfg.snr_bin_width))

        centers, means, stds = [], [], []
        for i in range(num_bins):
            lo = min_snr + i * cfg.snr_bin_width
            hi = lo + cfg.snr_bin_width
            m = (snrs >= lo) & (snrs < hi)
            if np.sum(m) > 10:
                centers.append((lo + hi) / 2)
                means.append(float(np.mean(scores[m])))
                stds.append(float(np.std(scores[m])))

        bin_centers, bin_means, bin_stds = map(np.asarray, (centers, means, stds))

        def sigmoid(x, x0, k, L, b):
            return L / (1 + np.exp(-k * (x - x0))) + b

        try:
            popt, _ = curve_fit(
                sigmoid,
                bin_centers,
                bin_means,
                p0=[10.0, 0.5, 0.8, 0.1],
                bounds=([0, 0.01, 0, 0], [30, 5, 1, 0.5]),
                maxfev=5000,
            )
            fit_snrs = np.linspace(min_snr, max_snr, 100)
            fit_efficiency = sigmoid(fit_snrs, *popt)
        except Exception:
            fit_snrs, fit_efficiency = bin_centers, bin_means

        def arr(x):
            return x if x is not None else np.array([])

        return {
            "snrs": snrs,
            "scores": scores,
            "fit_snrs": fit_snrs,
            "fit_efficiency": fit_efficiency,
            "bin_centers": bin_centers,
            "bin_means": bin_means,
            "bin_stds": bin_stds,
            "mass1": arr(self.mass1),
            "mass2": arr(self.mass2),
            "gps_times": arr(self.gps_times),
            "central_times": arr(self.central_times),
            "hpeak": arr(self.hpeak),
            "hrss": arr(self.hrss),
        }

    def get_worst_performers(self) -> Dict[str, Any]:
        """Worst samples: false negatives (per SNR bin) and false positives (noise)."""
        return {
            "false_negatives": self._worst_false_negatives or {},
            "false_positives": self._worst_false_positives or [],
        }

    def get_real_events(self) -> List[Dict]:
        """Return real events data (CONFIDENT first, then MARGINAL)."""
        if self.real_events is None:
            return []
        confident = [e for e in self.real_events if e.get("event_type") == "CONFIDENT"]
        marginal = [e for e in self.real_events if e.get("event_type") == "MARGINAL"]
        return confident + marginal

    def get_roc_curves(self, scaling_ranges: List[Union[Tuple[float, float], float]] = None) -> Dict:
        """Calculate ROC curves using generated noise + injection data."""
        if self.far_scores is None or self.scores is None:
            raise ValueError("Must call generate_noise() and generate() first")

        noise = self.far_scores
        if len(noise) == 0:
            logger.warning("No noise scores found. ROC calculation requires noise data.")
            return {}

        def roc_for(signal_scores: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
            y_scores = np.concatenate([noise, signal_scores])
            y_true = np.concatenate([np.zeros(len(noise)), np.ones(len(signal_scores))])
            fpr, tpr, auc = roc_curve_and_auc(y_true, y_scores)
            return {"fpr": np.asarray(fpr), "tpr": np.asarray(tpr), "roc_auc": float(auc)}

        results: Dict[str, Any] = {}

        # Default balanced ROC pool
        min_snr = self.config.default_roc_min_snr
        sig = self.scores[(self.snrs >= min_snr) & (self.injection_masks > 0.5)]
        if len(sig):
            n = min(len(noise), len(sig))
            balanced_noise, balanced_sig = noise, sig
            if len(noise) > n:
                rng = np.random.default_rng(42)
                balanced_noise = noise[rng.choice(len(noise), n, replace=False)]
            if len(sig) > n:
                rng = np.random.default_rng(42)
                balanced_sig = sig[rng.choice(len(sig), n, replace=False)]

            y_scores = np.concatenate([balanced_noise, balanced_sig])
            y_true = np.concatenate([np.zeros(len(balanced_noise)), np.ones(len(balanced_sig))])
            fpr, tpr, auc = roc_curve_and_auc(y_true, y_scores)

            results[f"SNR≥{min_snr} (balanced)"] = {
                "fpr": np.asarray(fpr),
                "tpr": np.asarray(tpr),
                "roc_auc": float(auc),
            }
            logger.info(f"Default ROC pool: {n} balanced samples (SNR≥{min_snr})")

        # Extra ROC pools
        pools = scaling_ranges if scaling_ranges is not None else self.config.extra_roc_pools
        for scaling in pools:
            if isinstance(scaling, (tuple, list)):
                lo, hi = scaling
                mask = (self.snrs >= lo) & (self.snrs < hi) & (self.injection_masks > 0.5)
                key = f"SNR {lo}-{hi}"
            else:
                center, width = float(scaling), 0.5
                mask = (
                    (self.snrs >= center - width)
                    & (self.snrs < center + width)
                    & (self.injection_masks > 0.5)
                )
                key = f"SNR={center}"

            pool_sig = self.scores[mask]
            if len(pool_sig) == 0:
                logger.warning(f"No signal samples found for ROC pool {key}")
                continue
            results[key] = roc_for(pool_sig)

        return results