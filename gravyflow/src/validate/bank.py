from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import logging
import math
import heapq
import sys
from copy import deepcopy
from scipy.optimize import curve_fit

import gravyflow as gf
from .config import ValidationConfig
from .utils import calculate_far_score_thresholds, roc_curve_and_auc, _extract_sample_data

logger = logging.getLogger(__name__)

class ValidationBank:
    """
    Manages generation and storage of validation data (noise and injections).
    Separates data generation logic from plotting and validation mechanics.
    """
    def __init__(
        self,
        model,
        dataset_args: dict,
        config: ValidationConfig = None,
        heart: gf.Heart = None
    ):
        self.model = model
        self.dataset_args = dataset_args
        self.config = config or ValidationConfig()
        self.heart = heart
        
        self.logger = logging.getLogger("validation_bank")
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)
        
        # Data storage
        self.far_scores = None
        self.noise_gps_times = None
        self._worst_false_positives = None
        
        self.snrs = None
        self.scores = None
        self.injection_masks = None
        self.mass1 = None
        self.mass2 = None
        self.gps_times = None
        self.central_times = None
        self.hpeak = None
        self.hrss = None
        self._worst_false_negatives = None
        
        # Real events storage (confident + marginal)
        self.real_events = None  # List of dicts with PE + score
    
    def _extract_scores(self, predictions) -> np.ndarray:
        """Extract scores from model predictions, handling various output shapes."""
        if len(predictions.shape) == 2:
            if predictions.shape[1] == 2:
                return predictions[:, 1]  # Binary classification
            return predictions[:, 0]
        return predictions.flatten()
        
    def generate_noise(self) -> None:
        """Generate noise scores for False Alarm Rate (FAR) calculation."""
        # Pre-fetch segments before deepcopy to ensure cache persists
        if "noise_obtainer" in self.dataset_args:
            noise_obt = self.dataset_args["noise_obtainer"]
            if hasattr(noise_obt, "ifo_data_obtainer") and noise_obt.ifo_data_obtainer:
                try:
                    if not hasattr(noise_obt.ifo_data_obtainer, "valid_segments") or noise_obt.ifo_data_obtainer.valid_segments is None:
                        self.logger.info("Pre-fetching valid segments for cache consistency...")
                        ifos = getattr(noise_obt, "ifos", None)
                        seed = self.dataset_args.get("seed", 42)
                        if ifos:
                            noise_obt.ifo_data_obtainer.get_valid_segments(
                                ifos=ifos, seed=seed,
                                group_name=self.dataset_args.get("group", "test")
                            )
                except Exception as e:
                    self.logger.warning(f"Failed to pre-fetch segments: {e}")

        config = self.config
        dataset_args = deepcopy(self.dataset_args)
        
        # Configure for noise-only (no injections)
        dataset_args["waveform_generators"] = []
        # Request data needed for plotting false positive samples
        dataset_args["output_variables"] = [
            gf.ReturnVariables.WHITENED_ONSOURCE,
            gf.ReturnVariables.GPS_TIME
        ]
        
        # Calculate number of batches for desired duration
        num_batches = math.ceil(config.num_examples / config.batch_size)
        dataset_args["num_examples_per_batch"] = config.batch_size
        dataset_args["steps_per_epoch"] = num_batches
        dataset_args["group"] = "test"
        
        dataset = gf.Dataset(**dataset_args)
        
        self.logger.info(f"Generating noise data: {num_batches} batches ({config.num_examples} examples)")
        
        all_scores = []
        all_noise_gps = []  # Collect GPS times for distribution visualization
        
        # Collect worst false positives (noise with highest scores)
        import random
        num_fp_to_keep = config.num_worst_per_bin * 5  # Keep more since no SNR bins
        fp_heap = []  # Min-heap: stores (score, random_tiebreak, sample_data)
        
        for batch_idx in range(num_batches):
            if self.heart:
                self.heart.beat()
            
            x_batch, y_batch = dataset[batch_idx]
            
            predictions = self.model.predict_on_batch(x_batch)
            batch_scores = self._extract_scores(predictions)
            
            all_scores.append(batch_scores)
            
            # Collect GPS times for distribution plot
            if y_batch and gf.ReturnVariables.GPS_TIME.name in y_batch:
                batch_gps = np.array(y_batch[gf.ReturnVariables.GPS_TIME.name])
                if batch_gps.ndim == 2:  # Shape (batch, 1) for single detector
                    batch_gps = batch_gps.flatten()
                all_noise_gps.extend(batch_gps)
            
            # Collect worst false positives (highest scores on noise)
            for i in range(len(batch_scores)):
                score_val = float(batch_scores[i])
                # Random tie-breaker to ensure diverse sampling when many scores are 1.0
                random_tie = random.random()
                
                if len(fp_heap) < num_fp_to_keep:
                    sample_data = _extract_sample_data(x_batch, y_batch if y_batch else {}, i, score_val)
                    heapq.heappush(fp_heap, (score_val, random_tie, sample_data))
                elif score_val > fp_heap[0][0] or (score_val == fp_heap[0][0] and random_tie > fp_heap[0][1]):
                    sample_data = _extract_sample_data(x_batch, y_batch if y_batch else {}, i, score_val)
                    heapq.heapreplace(fp_heap, (score_val, random_tie, sample_data))
            
            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"Noise Progress: {batch_idx + 1}/{num_batches} batches")
        
        self.far_scores = np.concatenate([np.array(s) for s in all_scores])
        self.noise_gps_times = np.array(all_noise_gps) if all_noise_gps else None
        self.logger.info(f"Generated {len(self.far_scores)} noise scores")
        
        # Store worst false positives (sorted by score descending)
        sorted_fp = sorted(fp_heap, key=lambda x: -x[0])
        self._worst_false_positives = [item[2] for item in sorted_fp]

    def generate(self) -> None:
        """
        Generate injection scores for Efficiency calculation.
        Sets injection_chance = 1.0 and samples Uniform SNR.
        """
        config = self.config
        dataset_args = deepcopy(self.dataset_args)
        
        min_snr, max_snr = config.snr_range
        
        if "waveform_generators" not in dataset_args:
            raise ValueError("dataset_args must contain waveform_generators")
        if not isinstance(dataset_args["waveform_generators"], list):
            dataset_args["waveform_generators"] = [dataset_args["waveform_generators"]]
            
        for gen in dataset_args["waveform_generators"]:
            gen.scaling_method.value = gf.Distribution(
                min_=min_snr,
                max_=max_snr,
                type_=gf.DistributionType.UNIFORM
            )
            gen.injection_chance = 1.0
            
        dataset_args["num_examples_per_batch"] = config.batch_size
        dataset_args["output_variables"] = [
            gf.ReturnVariables.WHITENED_ONSOURCE,
            gf.ReturnVariables.WHITENED_INJECTIONS,
            gf.ReturnVariables.INJECTION_MASKS,
            gf.ReturnVariables.GPS_TIME,
            gf.ReturnVariables.CENTRAL_TIME,
            gf.ScalingTypes.SNR,
            gf.ScalingTypes.HPEAK,
            gf.ScalingTypes.HRSS,
            gf.WaveformParameters.MASS_1_MSUN,
            gf.WaveformParameters.MASS_2_MSUN
        ]
        
        num_batches = math.ceil(config.num_examples / config.batch_size)
        dataset_args["steps_per_epoch"] = num_batches
        dataset_args["group"] = "test"
        
        dataset = gf.Dataset(**dataset_args)
        
        all_snrs = []
        all_scores = []
        all_masks = []
        all_mass1 = []
        all_mass2 = []
        all_mass2 = []
        all_mass2 = []
        all_gps = []
        all_central_times = []
        all_hpeak = []
        all_hrss = []
        
        # Worst performer tracking
        num_bins = int((max_snr - min_snr) / config.snr_bin_width)
        worst_heaps = [[] for _ in range(num_bins)]
        
        self.logger.info(f"Generating injection data: {num_batches} batches")
        
        for batch_idx in range(num_batches):
            if self.heart:
                self.heart.beat()
                
            x_batch, y_batch = dataset[batch_idx]
            
            predictions = self.model.predict_on_batch(x_batch)
            batch_scores = self._extract_scores(predictions)
                
            batch_snrs = np.array(y_batch.get("SNR", np.zeros(len(batch_scores))))
            batch_mass1 = np.array(y_batch.get(gf.WaveformParameters.MASS_1_MSUN.name, np.zeros(len(batch_scores))))
            batch_mass2 = np.array(y_batch.get(gf.WaveformParameters.MASS_2_MSUN.name, np.zeros(len(batch_scores))))
            batch_gps = np.array(y_batch.get(gf.ReturnVariables.GPS_TIME.name, np.zeros(len(batch_scores))))
            
            # Flatten any multi-dimensional arrays (e.g., shape (generators, batch) -> (batch,))
            if batch_snrs.ndim > 1:
                batch_snrs = batch_snrs.flatten()[:len(batch_scores)]
            if batch_mass1.ndim > 1:
                batch_mass1 = batch_mass1.flatten()[:len(batch_scores)]
            if batch_mass2.ndim > 1:
                batch_mass2 = batch_mass2.flatten()[:len(batch_scores)]
            if batch_gps.ndim > 1:
                batch_gps = batch_gps.flatten()[:len(batch_scores)]
            
            batch_central_times = np.array(y_batch.get(gf.ReturnVariables.CENTRAL_TIME.name, np.zeros(len(batch_scores))))
            
            # Handle multi-detector time (Batch, Detectors) -> Average to get central time
            if batch_central_times.ndim > 1 and batch_central_times.shape[-1] > 1:
                 batch_central_times = np.mean(batch_central_times, axis=-1)
            
            # Flatten to 1D array to match batch_scores
            if batch_central_times.size >= len(batch_scores):
                 batch_central_times = batch_central_times.flatten()
            
            batch_central_times = batch_central_times[:len(batch_scores)]
            
            batch_hpeak = np.array(y_batch.get(gf.ScalingTypes.HPEAK.name, np.zeros(len(batch_scores))))
            batch_hrss = np.array(y_batch.get(gf.ScalingTypes.HRSS.name, np.zeros(len(batch_scores))))
            
            batch_hpeak = batch_hpeak.flatten()[:len(batch_scores)]
            batch_hrss = batch_hrss.flatten()[:len(batch_scores)]
            
            # Injections are 100%, so marks are arguably all 1, but let's read if available
            batch_masks = np.ones(len(batch_scores)) 
            
            all_snrs.extend(batch_snrs)
            all_scores.extend(batch_scores)
            all_masks.extend(batch_masks)
            all_mass1.extend(batch_mass1)
            all_mass2.extend(batch_mass2)
            all_gps.extend(batch_gps)
            all_central_times.extend(batch_central_times)
            all_hpeak.extend(batch_hpeak)
            all_hrss.extend(batch_hrss)
            
            # Worst performers
            for i in range(len(batch_scores)):
                snr_val = float(batch_snrs[i])
                score_val = float(batch_scores[i])
                
                bin_idx = min(int((snr_val - min_snr) / config.snr_bin_width), num_bins - 1)
                bin_idx = max(0, bin_idx)
                
                if len(worst_heaps[bin_idx]) < config.num_worst_per_bin:
                    sample_data = _extract_sample_data(x_batch, y_batch, i, score_val)
                    heapq.heappush(worst_heaps[bin_idx], (-score_val, batch_idx * config.batch_size + i, sample_data))
                elif score_val < -worst_heaps[bin_idx][0][0]:
                    sample_data = _extract_sample_data(x_batch, y_batch, i, score_val)
                    heapq.heapreplace(worst_heaps[bin_idx], (-score_val, batch_idx * config.batch_size + i, sample_data))
            
            if (batch_idx + 1) % 50 == 0:
                self.logger.info(f"Injection Progress: {batch_idx + 1}/{num_batches} batches")
                
        self.snrs = np.array(all_snrs)
        self.scores = np.array(all_scores)
        self.injection_masks = np.array(all_masks)
        self.mass1 = np.array(all_mass1)
        self.mass2 = np.array(all_mass2)
        self.gps_times = np.array(all_gps)
        self.central_times = np.array(all_central_times)
        self.hpeak = np.array(all_hpeak)
        self.hrss = np.array(all_hrss)
        
        self._worst_false_negatives = {}
        for bin_idx, heap in enumerate(worst_heaps):
            bin_start = min_snr + bin_idx * config.snr_bin_width
            bin_end = bin_start + config.snr_bin_width
            bin_key = f"{bin_start:.0f}-{bin_end:.0f}"
            sorted_worst = sorted(heap, key=lambda x: -x[0])
            self._worst_false_negatives[bin_key] = [item[2] for item in sorted_worst]
    
    def generate_real_events(self, observing_runs: List = None) -> None:
        """
        Generate scores for real GW events from GWTC catalogs.
        
        Uses TransientObtainer with event_names to directly fetch and score
        specific events, simplifying the previous manual catalog/GPS matching approach.
        
        Args:
            observing_runs: List of ObservingRun enums to include.
                           Defaults to [O1, O2, O3].
        """
        from gravyflow.src.dataset.features.event import (
            get_events_with_params, EventType
        )
        from gravyflow.src.dataset.conditioning.whiten import whiten
        
        if observing_runs is None:
            observing_runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]
        
        self.logger.info("Fetching real GW events from GWTC catalogs...")
        
        # Fetch all events (confident + marginal) with PE parameters
        all_events = get_events_with_params(
            observing_runs,
            event_types=[EventType.CONFIDENT, EventType.MARGINAL]
        )
        
        # Add event_type field based on catalog name
        for event in all_events:
            catalog = event.get("catalog", "")
            event["event_type"] = "MARGINAL" if "marginal" in catalog.lower() else "CONFIDENT"
        
        # Deduplicate by GPS (some events appear in multiple catalogs)
        seen_gps = set()
        unique_events = []
        for event in all_events:
            gps_key = round(event["gps"], 1)
            if gps_key not in seen_gps:
                seen_gps.add(gps_key)
                unique_events.append(event)
        
        unique_events.sort(key=lambda x: x["gps"])
        
        confident_count = sum(1 for e in unique_events if e["event_type"] == "CONFIDENT")
        marginal_count = len(unique_events) - confident_count
        self.logger.info(f"Found {len(unique_events)} unique events "
                        f"({confident_count} confident, {marginal_count} marginal)")
        
        # Get IFO configuration from existing noise_obtainer
        if "noise_obtainer" not in self.dataset_args:
            self.logger.warning("No noise_obtainer in dataset_args - cannot score events")
            self.real_events = unique_events
            return
        
        orig_noise_obt = self.dataset_args["noise_obtainer"]
        ifos = getattr(orig_noise_obt, "ifos", [gf.IFO.H1, gf.IFO.L1])
        
        # Get event names for TransientObtainer
        event_names = [e["name"] for e in unique_events]
        
        self.logger.info(f"Creating TransientObtainer for {len(event_names)} events...")
        
        try:
            # Use TransientObtainer with event_names to target specific events
            transient_obt = gf.TransientObtainer(
                ifo_data_obtainer=gf.IFODataObtainer(
                    observing_runs=observing_runs,
                    data_quality=gf.DataQuality.BEST,
                    data_labels=[gf.DataLabel.EVENTS]
                ),
                ifos=ifos,
                event_names=event_names
            )
            
            # Get sample/onsource durations from dataset_args
            sample_rate = self.dataset_args.get("sample_rate_hertz", 2048.0)
            onsource_duration = self.dataset_args.get("onsource_duration_seconds", 1.0)
            offsource_duration = self.dataset_args.get("offsource_duration_seconds", 16.0)
            
            # Create generator - use larger batch size for efficiency
            # The generator already supports batching internally
            batch_size = min(len(event_names), 256)  # Reasonable batch for GPU memory
            generator = transient_obt(
                sample_rate_hertz=sample_rate,
                onsource_duration_seconds=onsource_duration,
                offsource_duration_seconds=offsource_duration,
                num_examples_per_batch=batch_size,
                scale_factor=gf.Defaults.scale_factor  # Apply 1e21 scaling
            )
            
            # =================================================================
            # PHASE 1: Collect all event data (I/O bound - can't parallelize)
            # =================================================================
            self.logger.info("Phase 1: Collecting event data...")
            all_onsource = []
            all_offsource = []
            all_gps = []
            event_count = 0
            
            for onsource, offsource, gps_times, labels in generator:
                if event_count >= len(event_names):
                    break
                
                if self.heart:
                    self.heart.beat()
                
                # Handle both single and batched returns
                batch_len = onsource.shape[0] if len(onsource.shape) > 2 else 1
                
                for idx in range(batch_len):
                    if event_count >= len(event_names):
                        break
                    
                    # Extract single event from batch
                    if batch_len > 1:
                        ons = onsource[idx:idx+1]
                        offs = offsource[idx:idx+1]
                        gps = float(gps_times[idx, 0]) if len(gps_times.shape) > 1 else float(gps_times[idx])
                    else:
                        ons = onsource
                        offs = offsource
                        gps = float(gps_times[0, 0]) if len(gps_times.shape) > 1 else float(gps_times[0])
                    
                    all_onsource.append(np.array(ons))
                    all_offsource.append(np.array(offs))
                    all_gps.append(gps)
                    event_count += 1
                
                if event_count % 20 == 0:
                    self.logger.info(f"Data collection: {event_count}/{len(event_names)} events")
            
            self.logger.info(f"Collected {len(all_onsource)} events for batch scoring")
            
            # =================================================================
            # PHASE 2: Batch model inference (FAST - single forward pass)
            # =================================================================
            if len(all_onsource) > 0:
                self.logger.info("Phase 2: Running batched model inference...")
                
                # Stack all events into single batch
                stacked_ons = np.concatenate(all_onsource, axis=0)  # (N, IFOs, samples)
                stacked_offs = np.concatenate(all_offsource, axis=0)
                
                # Single forward pass through model
                x_batch = {"ONSOURCE": stacked_ons, "OFFSOURCE": stacked_offs}
                predictions = self.model.predict(x_batch, verbose=0)
                all_scores = self._extract_scores(predictions)
                
                self.logger.info(f"Batch inference complete: {len(all_scores)} scores")
                
                # =================================================================
                # PHASE 3: Match scores to events and whiten for plotting
                # =================================================================
                self.logger.info("Phase 3: Matching scores to events...")
                scored_count = 0
                
                for i, (gps, score) in enumerate(zip(all_gps, all_scores)):
                    # Match to event by GPS time
                    for event in unique_events:
                        if abs(event["gps"] - gps) < 20.0:
                            event["score"] = float(score)
                            event["status"] = "scored"
                            
                            # Whiten for plotting (store as numpy)
                            try:
                                ons = all_onsource[i]
                                offs = all_offsource[i]
                                whitened = whiten(
                                    ons,
                                    offs,
                                    sample_rate_hertz=sample_rate
                                )
                                cropped = gf.crop_samples(
                                    whitened,
                                    sample_rate_hertz=sample_rate,
                                    onsource_duration_seconds=onsource_duration
                                )
                                event["whitened_strain"] = np.array(cropped)[0]  # (IFOs, T)
                            except Exception as e:
                                self.logger.warning(f"Whitening failed for event at {gps}: {e}")
                            
                            scored_count += 1
                            break
                    
        except StopIteration:
            pass  # Normal generator exhaustion
        except Exception as e:
            self.logger.warning(f"TransientObtainer failed: {e}")
            self.logger.info("Events stored without scores")
        
        # Calculate min FAR for scored events
        if self.far_scores is not None and len(self.far_scores) > 0:
            from .utils import calculate_far_score_thresholds
            thresholds = calculate_far_score_thresholds(
                self.far_scores,
                self.dataset_args.get("onsource_duration_seconds", 1.0),
                np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
            )
            
            for event in unique_events:
                if event.get("score") is not None:
                    score = event["score"]
                    for far in sorted(thresholds.keys()):
                        if score >= thresholds[far][1]:
                            event["min_far"] = far
                            break
        
        self.real_events = unique_events
        scored = sum(1 for e in unique_events if e.get("score") is not None)
        self.logger.info(f"Real events scoring complete: {scored}/{len(unique_events)} scored")
            
    def get_efficiency_data(self) -> Dict:
        """
        Get efficiency scatter data and fitted curve.
        """
        if self.snrs is None:
            raise ValueError("Must call generate() first")
        
        snrs = self.snrs
        scores = self.scores
        
        config = self.config
        min_snr, max_snr = config.snr_range
        num_bins = max(1, int((max_snr - min_snr) / config.snr_bin_width))
        
        bin_centers = []
        bin_means = []
        bin_stds = []
        
        for i in range(num_bins):
            bin_start = min_snr + i * config.snr_bin_width
            bin_end = bin_start + config.snr_bin_width
            bin_mask = (snrs >= bin_start) & (snrs < bin_end)
            
            if np.sum(bin_mask) > 10:
                bin_centers.append((bin_start + bin_end) / 2)
                bin_means.append(np.mean(scores[bin_mask]))
                bin_stds.append(np.std(scores[bin_mask]))
        
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        # Fit sigmoid curve
        def sigmoid(x, x0, k, L, b):
            return L / (1 + np.exp(-k * (x - x0))) + b
        
        try:
            popt, _ = curve_fit(
                sigmoid, bin_centers, bin_means,
                p0=[10.0, 0.5, 0.8, 0.1],
                bounds=([0, 0.01, 0, 0], [30, 5, 1, 0.5]),
                maxfev=5000
            )
            
            fit_snrs = np.linspace(min_snr, max_snr, 100)
            fit_efficiency = sigmoid(fit_snrs, *popt)
        except:
            fit_snrs = bin_centers
            fit_efficiency = bin_means
        
        return {
            "snrs": snrs,
            "scores": scores,
            "fit_snrs": fit_snrs,
            "fit_efficiency": fit_efficiency,
            "bin_centers": bin_centers,
            "bin_means": bin_means,
            "bin_stds": bin_stds,
            # Parameter space data
            "mass1": self.mass1 if self.mass1 is not None else np.array([]),
            "mass2": self.mass2 if self.mass2 is not None else np.array([]),
            "mass2": self.mass2 if self.mass2 is not None else np.array([]),
            "mass2": self.mass2 if self.mass2 is not None else np.array([]),
            "gps_times": self.gps_times if self.gps_times is not None else np.array([]),
            "central_times": self.central_times if self.central_times is not None else np.array([]),
            "hpeak": self.hpeak if self.hpeak is not None else np.array([]),
            "hrss": self.hrss if self.hrss is not None else np.array([])
        }
    
    def get_worst_performers(self) -> Dict[str, any]:
        """Get worst performing samples: false negatives (by SNR bin) and false positives."""
        return {
            "false_negatives": self._worst_false_negatives or {},
            "false_positives": self._worst_false_positives or []
        }
    
    def get_real_events(self) -> List[Dict]:
        """
        Get real events data with scores and detection status.
        
        Returns:
            List of event dicts, separated by type (CONFIDENT first, then MARGINAL).
        """
        if self.real_events is None:
            return []
        
        # Separate by event type, confident first
        confident = [e for e in self.real_events if e.get("event_type") == "CONFIDENT"]
        marginal = [e for e in self.real_events if e.get("event_type") == "MARGINAL"]
        
        return confident + marginal

    def get_roc_curves(self, scaling_ranges: List[Union[Tuple[float, float], float]] = None) -> Dict:
        """
        Calculate ROC curves using generated data.
        """
        if self.far_scores is None or self.scores is None:
            raise ValueError("Must call generate_noise() and generate() first")
            
        results = {}
        noise_scores = self.far_scores
        
        if len(noise_scores) == 0:
            self.logger.warning("No noise scores found. ROC calculation requires noise data.")
            return {}
        
        # === Default balanced ROC pool ===
        min_snr = self.config.default_roc_min_snr
        signal_mask = (self.snrs >= min_snr) & (self.injection_masks > 0.5)
        signal_scores = self.scores[signal_mask]
        
        if len(signal_scores) > 0:
            # Balance sample counts
            n_noise = len(noise_scores)
            n_signal = len(signal_scores)
            n_balanced = min(n_noise, n_signal)
            
            # Random subsample if needed
            if n_noise > n_balanced:
                rng = np.random.default_rng(42)
                noise_idx = rng.choice(n_noise, n_balanced, replace=False)
                balanced_noise = noise_scores[noise_idx]
            else:
                balanced_noise = noise_scores
                
            if n_signal > n_balanced:
                rng = np.random.default_rng(42)
                signal_idx = rng.choice(n_signal, n_balanced, replace=False)
                balanced_signal = signal_scores[signal_idx]
            else:
                balanced_signal = signal_scores
            
            # Compute ROC for balanced pool
            y_scores = np.concatenate([balanced_noise, balanced_signal])
            y_true = np.concatenate([np.zeros(len(balanced_noise)), np.ones(len(balanced_signal))])
            
            fpr, tpr, auc = roc_curve_and_auc(y_true, y_scores)
            
            results[f"SNR≥{min_snr} (balanced)"] = {
                "fpr": np.array(fpr),
                "tpr": np.array(tpr),
                "roc_auc": float(auc)
            }
            self.logger.info(f"Default ROC pool: {n_balanced} balanced samples (SNR≥{min_snr})")
        
        # === Extra ROC pools (if any) ===
        extra_pools = scaling_ranges if scaling_ranges is not None else self.config.extra_roc_pools
        
        for scaling in extra_pools:
            if isinstance(scaling, (tuple, list)):
                min_s, max_s = scaling
                mask = (self.snrs >= min_s) & (self.snrs < max_s)
                mask &= (self.injection_masks > 0.5)
                key = f"SNR {min_s}-{max_s}"
            else:
                center = float(scaling)
                width = 0.5
                mask = (self.snrs >= center - width) & (self.snrs < center + width)
                mask &= (self.injection_masks > 0.5)
                key = f"SNR={center}"
            
            pool_signal_scores = self.scores[mask]
            
            if len(pool_signal_scores) == 0:
                self.logger.warning(f"No signal samples found for ROC pool {key}")
                continue
            
            y_scores = np.concatenate([noise_scores, pool_signal_scores])
            y_true = np.concatenate([np.zeros(len(noise_scores)), np.ones(len(pool_signal_scores))])
            
            fpr, tpr, auc = roc_curve_and_auc(y_true, y_scores)
            
            results[key] = {
                "fpr": np.array(fpr),
                "tpr": np.array(tpr),
                "roc_auc": float(auc)
            }
            
        return results
