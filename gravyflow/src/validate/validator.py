from pathlib import Path
import sys
import logging
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
import h5py
import panel as pn
import keras

import gravyflow as gf
from .config import ValidationConfig
from .bank import ValidationBank
from .plotting import (
    generate_efficiency_plot, 
    generate_far_curves, 
    generate_gps_distribution_plot, 
    generate_roc_curves, 
    generate_parameter_space_plot,
    generate_waveform_plot,
    generate_real_events_table
)

logger = logging.getLogger(__name__)

class Validator:
    
    @staticmethod
    def _setup_logger(logging_level: int = logging.INFO):
        """Configure and return validator logger."""
        log = logging.getLogger("validator")
        if not log.handlers:
            log.addHandler(logging.StreamHandler(sys.stdout))
        log.setLevel(logging_level)
        return log
    
    @classmethod
    def validate(
        cls, 
        model : keras.Model, 
        name : str,
        dataset_args : dict,
        config: ValidationConfig = None,
        checkpoint_file_path : Path = None,
        logging_level : int = logging.INFO,
        heart : gf.Heart = None,
        **kwargs # Absorb legacy kwargs
    ):
        validator = cls()
        validator.logger = cls._setup_logger(logging_level)
        
        validator.name = str(name) if name else "model"
        validator.config = config or ValidationConfig()
        
        validator.input_duration_seconds = dataset_args.get(
            "onsource_duration_seconds", 
            gf.Defaults.onsource_duration_seconds
        )
        validator.offsource_duration_seconds = dataset_args.get(
            "offsource_duration_seconds",
            gf.Defaults.offsource_duration_seconds
        )
        
        # Store dataset_args for extracting valid_segments in plot()
        validator.dataset_args = dataset_args
        
        validator.efficiency_data = None
        validator.far_scores = None
        validator.roc_data = None
        validator.worst_performers = None
        validator.real_events = None
        
        if checkpoint_file_path and checkpoint_file_path.exists():
            try:
                validator = validator.load(checkpoint_file_path, logging_level)
                
                # Always restore runtime attributes after load
                validator.config = config or ValidationConfig()
                validator.dataset_args = dataset_args
                validator.logger = cls._setup_logger(logging_level)
                
                if validator.efficiency_data is not None and validator.far_scores is not None:
                    # Check if real events are scored (if expected)
                    real_events_scored = False
                    if validator.real_events:
                        real_events_scored = any(e.get("score") is not None for e in validator.real_events)
                    
                    if real_events_scored:
                        validator.logger.info("Loaded complete validation data (including scored real events) from checkpoint.")
                        return validator
                    else:
                        validator.logger.info("Loaded partial validation data. Real events missing scores - regenerating...")
            except Exception as e:
                validator.logger.warning(f"Failed to load checkpoint: {e}. Regenerating.")
        
        bank = ValidationBank(
            model=model,
            dataset_args=dataset_args,
            config=validator.config,
            heart=heart
        )
        
        if validator.far_scores is None:
            bank.generate_noise()
            validator.far_scores = bank.far_scores
            validator.noise_gps_times = bank.noise_gps_times
            
        if validator.efficiency_data is None:
            bank.generate()
            validator.efficiency_data = bank.get_efficiency_data()
            validator.worst_performers = bank.get_worst_performers()
            
        if validator.roc_data is None:
            validator.roc_data = bank.get_roc_curves()
        
        # Generate real events evaluation (optional - calls GWTC catalogs)
        # Check if we need to generate: None, empty, or unscored
        need_real_events = True
        if validator.real_events:
             if any(e.get("score") is not None for e in validator.real_events):
                 need_real_events = False
        
        if need_real_events:
            try:
                bank.generate_real_events()
                validator.real_events = bank.get_real_events()
            except Exception as e:
                validator.logger.warning(f"Real events generation failed: {e}")
                validator.real_events = []
        
        if checkpoint_file_path:
            validator.save(checkpoint_file_path)
            
        validator.bank = bank
        return validator

    def save(self, file_path: Path):
        self.logger.info(f"Saving validation data to {file_path}")
        gf.ensure_directory_exists(file_path.parent)
        
        with gf.open_hdf5_file(file_path, self.logger, mode="w") as f:
            f.create_dataset('name', data=self.name.encode())
            f.create_dataset('input_duration_seconds', data=self.input_duration_seconds)
            
            if self.efficiency_data:
                g = f.create_group('efficiency_data')
                g.create_dataset('snrs', data=self.efficiency_data['snrs'])
                g.create_dataset('scores', data=self.efficiency_data['scores'])
                # Parameter space data
                if 'mass1' in self.efficiency_data and len(self.efficiency_data['mass1']) > 0:
                    g.create_dataset('mass1', data=self.efficiency_data['mass1'])
                if 'mass2' in self.efficiency_data and len(self.efficiency_data['mass2']) > 0:
                    g.create_dataset('mass2', data=self.efficiency_data['mass2'])
                if 'gps_times' in self.efficiency_data and len(self.efficiency_data['gps_times']) > 0:
                    g.create_dataset('gps_times', data=self.efficiency_data['gps_times'])
                
            if self.far_scores is not None:
                g = f.create_group('far_scores')
                g.create_dataset('scores', data=self.far_scores)
                if hasattr(self, 'noise_gps_times') and self.noise_gps_times is not None and len(self.noise_gps_times) > 0:
                    g.create_dataset('gps_times', data=self.noise_gps_times)
                
            if self.roc_data:
                g = f.create_group('roc_data')
                keys = list(self.roc_data.keys())
                g.create_dataset('keys', data=np.array([k.encode() for k in keys], dtype=h5py.string_dtype()))
                for k, v in self.roc_data.items():
                    sub = g.create_group(k)
                    sub.create_dataset('fpr', data=v['fpr'])
                    sub.create_dataset('tpr', data=v['tpr'])
                    sub.create_dataset('roc_auc', data=v['roc_auc'])
            
            if self.worst_performers:
                g = f.create_group('worst_performers')
                
                # Save false negatives (by SNR bin)
                false_negatives = self.worst_performers.get("false_negatives", {})
                if false_negatives:
                    fn_g = g.create_group('false_negatives')
                    for bin_key, samples in false_negatives.items():
                        bg = fn_g.create_group(bin_key)
                        for i, sample in enumerate(samples):
                            sg = bg.create_group(f"sample_{i}")
                            for k, v in sample.items():
                                if isinstance(v, np.ndarray):
                                    sg.create_dataset(k, data=v)
                                else:
                                    sg.create_dataset(k, data=v)
                
                # Save false positives (flat list)
                false_positives = self.worst_performers.get("false_positives", [])
                if false_positives:
                    fp_g = g.create_group('false_positives')
                    for i, sample in enumerate(false_positives):
                        sg = fp_g.create_group(f"sample_{i}")
                        for k, v in sample.items():
                            if isinstance(v, np.ndarray):
                                sg.create_dataset(k, data=v)
                            else:
                                sg.create_dataset(k, data=v)
            
            # Save real events data
            if self.real_events:
                re_g = f.create_group('real_events')
                for i, event in enumerate(self.real_events):
                    eg = re_g.create_group(f"event_{i}")
                    for k, v in event.items():
                        if v is not None:
                            if isinstance(v, str):
                                eg.create_dataset(k, data=v.encode())
                            elif isinstance(v, (int, float)):
                                eg.create_dataset(k, data=v)
                            elif isinstance(v, np.ndarray):
                                eg.create_dataset(k, data=v)

    @classmethod
    def load(cls, file_path: Path, logging_level=logging.INFO):
        validator = cls()
        validator.logger = logging.getLogger("validator")
        validator.logger.setLevel(logging_level)
        
        with h5py.File(file_path, 'r') as f:
            validator.name = f['name'][()].decode() if 'name' in f else "Unknown"
            validator.input_duration_seconds = float(f['input_duration_seconds'][()]) if 'input_duration_seconds' in f else 1.0
            
            if 'efficiency_data' in f:
                g = f['efficiency_data']
                validator.efficiency_data = {
                    "snrs": g['snrs'][:],
                    "scores": g['scores'][:],
                    "mass1": g['mass1'][:] if 'mass1' in g else np.array([]),
                    "mass2": g['mass2'][:] if 'mass2' in g else np.array([]),
                    "gps_times": g['gps_times'][:] if 'gps_times' in g else np.array([])
                }
            
            if 'far_scores' in f:
                g = f['far_scores']
                validator.far_scores = g['scores'][:]
                validator.noise_gps_times = g['gps_times'][:] if 'gps_times' in g else None
                
            if 'roc_data' in f:
                g = f['roc_data']
                validator.roc_data = {}
                keys = [k.decode() for k in g['keys'][:]]
                for k in keys:
                    if k in g:
                        sub = g[k]
                        validator.roc_data[k] = {
                            "fpr": sub['fpr'][:],
                            "tpr": sub['tpr'][:],
                            "roc_auc": float(sub['roc_auc'][()])
                        }
            
            if 'worst_performers' in f:
                g = f['worst_performers']
                validator.worst_performers = {"false_negatives": {}, "false_positives": []}
                
                # Load false negatives (by SNR bin)
                if 'false_negatives' in g:
                    fn_g = g['false_negatives']
                    for bin_key in fn_g:
                        validator.worst_performers["false_negatives"][bin_key] = []
                        bg = fn_g[bin_key]
                        for sample_key in sorted(bg.keys()):
                            sg = bg[sample_key]
                            sample = {}
                            for k in sg:
                                sample[k] = sg[k][()]
                            validator.worst_performers["false_negatives"][bin_key].append(sample)
                
                # Load false positives (flat list)
                if 'false_positives' in g:
                    fp_g = g['false_positives']
                    for sample_key in sorted(fp_g.keys()):
                        sg = fp_g[sample_key]
                        sample = {}
                        for k in sg:
                            sample[k] = sg[k][()]
                        validator.worst_performers["false_positives"].append(sample)
            
            # Load real events if present (otherwise will regenerate)
            if 'real_events' in f:
                re_g = f['real_events']
                validator.real_events = []
                for event_key in sorted(re_g.keys()):
                    eg = re_g[event_key]
                    event = {}
                    for k in eg:
                        val = eg[k][()]
                        if isinstance(val, bytes):
                            event[k] = val.decode()
                        else:
                            event[k] = val if not isinstance(val, np.ndarray) or val.shape else float(val)
                    validator.real_events.append(event)
            else:
                validator.real_events = None  # Will trigger regeneration

        return validator

    def plot(self, output_path: Path = None, comparison_validators: list = []):
        """Generate validation dashboard."""
        
        all_validators = comparison_validators + [self]
        
        eff_plot, eff_slider = generate_efficiency_plot(
            all_validators, 
            fars=self.config.far_thresholds
        )
        far_plot = generate_far_curves(all_validators)
        
        # Try to extract valid_segments from dataset_args for GPS plot
        valid_segments = None
        onsource_dur = getattr(self, 'input_duration_seconds', 1.0)
        offsource_dur = getattr(self, 'offsource_duration_seconds', 16.0)
        
        if hasattr(self, 'dataset_args') and self.dataset_args:
            noise_obtainer = self.dataset_args.get('noise_obtainer')
            if noise_obtainer and hasattr(noise_obtainer, 'ifo_data_obtainer'):
                ifo_obtainer = noise_obtainer.ifo_data_obtainer
                if ifo_obtainer:
                    # Check for adjusted segments first, then raw segments
                    if hasattr(ifo_obtainer, 'valid_segments_adjusted') and ifo_obtainer.valid_segments_adjusted is not None:
                        valid_segments = ifo_obtainer.valid_segments_adjusted
                        self.logger.info(f"GPS plot: Using valid_segments_adjusted, shape={valid_segments.shape}")
                    elif hasattr(ifo_obtainer, 'valid_segments') and ifo_obtainer.valid_segments is not None:
                        valid_segments = ifo_obtainer.valid_segments
                        self.logger.info(f"GPS plot: Using valid_segments, shape={valid_segments.shape}")
                    else:
                        self.logger.info("GPS plot: Reloading valid segments for distribution plot...")
                        try:
                            # Try to fetch segments if they are missing (e.g. loaded from checkpoint)
                            # Extract ifos from noise_obtainer or config
                            ifos = getattr(noise_obtainer, 'ifos', None)
                            if not ifos and hasattr(self, 'config'):
                                ifos = getattr(self.config, 'ifos', None)
                                
                            if ifos:
                                self.logger.info(f"GPS plot: Fetching valid segments for {len(ifos)} IFOs...")
                                # Use get_valid_segments which handles caching
                                segments = ifo_obtainer.get_valid_segments(
                                    ifos=ifos,
                                    seed=42, # Default seed
                                    group_name=self.dataset_args.get("group", "test")
                                )
                                if segments is not None:
                                    valid_segments = np.array(segments)
                                    # Cache it back to object
                                    ifo_obtainer.valid_segments = valid_segments
                                    self.logger.info(f"GPS plot: Successfully fetched segments, shape={valid_segments.shape}")
                                else:
                                    self.logger.warning("GPS plot: get_valid_segments returned None")
                            else:
                                self.logger.warning("GPS plot: ifo_obtainer/noise_obtainer has no IFOs configured")
                        except Exception as e:
                            self.logger.warning(f"GPS plot: Failed to fetch valid segments: {e}")
                else:
                    self.logger.info("GPS plot: ifo_data_obtainer is None")
            else:
                self.logger.info("GPS plot: noise_obtainer has no ifo_data_obtainer")
        else:
            self.logger.info("GPS plot: No dataset_args available")
        
        gps_dist_plot = generate_gps_distribution_plot(
            all_validators,
            valid_segments=valid_segments,
            onsource_duration_seconds=onsource_dur,
            offsource_duration_seconds=offsource_dur
        )
        roc_plot, roc_select = generate_roc_curves(all_validators)
        param_plot, param_controls = generate_parameter_space_plot(all_validators)
        
        tabs = pn.Tabs(
            ("Efficiency", pn.Column(eff_slider, eff_plot)),
            ("ROC Curves", pn.Column(roc_select, roc_plot)),
            ("FAR Curves", pn.Column(far_plot, gps_dist_plot)),
            ("Parameter Space", pn.Column(param_controls, param_plot) if param_controls else param_plot)
        )
        
        # Add Real Events tab if data available
        if hasattr(self, 'real_events') and self.real_events:
            events_table, _ = generate_real_events_table(
                self.real_events,
                far_scores=self.far_scores,
                input_duration_seconds=self.input_duration_seconds,
                far_thresholds=self.config.far_thresholds
            )
            tabs.append(("Real Events", events_table))
        
        if self.worst_performers:
            # False Negatives tab (injections with low scores, by SNR bin)
            false_negatives = self.worst_performers.get("false_negatives", {})
            if false_negatives:
                fn_tabs = []
                sorted_bins = sorted(false_negatives.keys(), key=lambda x: float(str(x).split('-')[0]) if '-' in str(x) else 0)
                
                for bin_key in sorted_bins:
                    samples = false_negatives[bin_key]
                    if not samples: continue
                    
                    plots = []
                    for sample in samples[:10]:
                         try:
                             plots.append(generate_waveform_plot(sample, self.input_duration_seconds))
                         except Exception as e:
                             self.logger.warning(f"Error plotting FN sample: {e}")
                    
                    if plots:
                        fn_tabs.append((f"SNR {bin_key}", pn.Column(*plots)))
                
                if fn_tabs:
                    tabs.append(("Worst False Negatives", pn.Tabs(*fn_tabs)))
            
            # False Positives tab (noise with high scores)
            false_positives = self.worst_performers.get("false_positives", [])
            if false_positives:
                fp_plots = []
                for sample in false_positives[:25]:  # Show up to 25 false positives
                    try:
                        fp_plots.append(generate_waveform_plot(sample, self.input_duration_seconds))
                    except Exception as e:
                        self.logger.warning(f"Error plotting FP sample: {e}")
                
                if fp_plots:
                    tabs.append(("Worst False Positives", pn.Column(*fp_plots)))
                
        # Restore FastListTemplate as requested (Dark Mode forced)
        template = pn.template.FastListTemplate(
            title=f"Validation Dashboard: {self.name}",
            theme="dark",
            theme_toggle=False,
            main=[tabs]
        )
        
        if output_path:
            gf.ensure_directory_exists(Path(output_path).parent)
            # Templates do not support embedding, so no embed=True
            template.save(str(output_path), resources='inline')
            self.logger.info(f"Saved report to {output_path}")
            
        return template
