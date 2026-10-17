# The public API

`import gravyflow as gf` gives the names below, and only these (`gf.__all__`; a test holds the two lists equal).
Each imports its module on first use, so `import gravyflow` itself is instant and changes nothing in the process.
Everything else under `gravyflow.src` is internal: importable, but free to change without notice.

gravyflow runs on Keras's JAX backend. Set `KERAS_BACKEND=jax` before Keras is imported (for example,
`export KERAS_BACKEND=jax` in the shell); the first use of a name refuses with that instruction otherwise.

`gf.__version__` is the package's version.

## Configuration and randomness

- `Defaults`: the defaults every dataset takes (sample rate, durations, seed, batch size).
- `WindowSpec`: the onsource, offsource and crop durations of one window.
- `SplitSpec`: how GPS time is split into train, validate and test.
- `DEFAULT_SPLIT`: the default `SplitSpec`.
- `EXCLUDED`: the label of time that belongs to no split.
- `SeedTree`: the one seed tree every random draw comes from.
- `BatchStreams`: the random streams of one batch, from a `SeedTree`.
- `set_random_seeds`: seed Python's, NumPy's and Keras's global generators (for your own code; gravyflow never draws from them).
- `Distribution`: a distribution of one parameter.
- `DistributionType`: the kinds of `Distribution`.

## Datasets

- `Dataset`: build a dataset: noise, injections, conditioning, inputs and outputs.
- `GravyflowDataset`: the class `Dataset` returns, a Keras `PyDataset` of batches.
- `ComposedDataset`: examples drawn from feature pools by a composition.
- `EpochMode`: how a dataset's batches are addressed across epochs.
- `Unprepared`: raised by `Dataset.prepare`, naming every contradiction between the timeline and the source, and the `exclusions` that plan around them.
- `FeaturePool`: a pool of one kind of example for a `ComposedDataset`.
- `PoolSampler`: how a `ComposedDataset` draws from its pools.
- `Curriculum`: parameters that change over training.
- `CurriculumSchedule`: how a `Curriculum` moves between its stages.
- `CurriculumProgressCallback`: the Keras callback that advances a `Curriculum`.
- `DiversityCallback`: the Keras callback that tracks how diverse training batches are.
- `LabelTrackingDataset`: a dataset wrapper that records the labels it serves.
- `compute_diversity_score`: the diversity of a set of labels.
- `BatchConfig`: the settings of batch generation.
- `BatchContract`: the shape and dtype of every output of a dataset's batches.
- `BatchContractError`: raised when a batch breaks its contract.
- `ReturnVariables`: the arrays a dataset can return.

## Noise and detectors

- `NoiseType`: white, coloured, pseudo-real or real noise.
- `NoiseObtainer`: where a dataset's noise comes from.
- `Obtainer`: the base of the data obtainers (noise and transients).
- `IFO`: the detectors gravyflow knows.
- `Network`: a network of detectors, and the projection of waves onto it.
- `NoiseProfile`: a named set of per-detector noise curves for coloured noise (bundled with the package).
- `project_wave`: project polarisations onto detectors.
- `greenwich_mean_sidereal_time`: GMST at a GPS time.
- `REFERENCE_GPS_TIME_SECONDS`: the GPS time sky positions are referred to.

## Real data (the `realdata` extra)

- `IFODataObtainer`: real detector data from GWOSC or GWDataFind.
- `NoiseDataObtainer`: real noise segments, with events and glitches vetoed.
- `TransientDataObtainer`: windows around real glitches and events.
- `IFOData`: strain from one or more detectors, with its GPS times.
- `ObservingRun`: the observing runs and their bounds.
- `DataQuality`: the data quality to request.
- `DataLabel`: what the data holds: noise, glitches or events.
- `SegmentOrder`: the order segments are served in.
- `AcquisitionMode`: noise windows or transients.
- `SamplingMode`: how windows are drawn from segments.
- `Coincidence`: whether an example's detectors are read at one time: never by default (`SEPARATED`), or under the opt-in `COINCIDENT`; every real-noise batch says which examples are (`ReturnVariables.COINCIDENT`), and a dataset returns it when asked (0 for simulated noise).
- `SignReversal`: an augmentation that flips the sign of the strain.
- `TimeReversal`: an augmentation that reverses the strain in time.
- `RandomShift`: an augmentation that shifts the strain in time.
- `AddNoise`: an augmentation that adds Gaussian noise.
- `GWOSC_DATA_FLAGS`: GWOSC's data-quality flags for usable data.
- `GWOSC_BURST_FLAGS`: GWOSC's burst vetoes.
- `GWOSC_CBC_FLAGS`: GWOSC's CBC vetoes.
- `GWOSC_NO_HW_INJECTION_FLAGS`: GWOSC's flags for time without hardware injections.
- `DEFAULT_GWOSC_FLAGS`: the default selection: data, burst vetoes, no hardware injections.
- `DataSource`: where real strain and its segments come from; every obtainer takes one (`source=`).
- `GWOSC`: the default source: GWOSC's public bulk strain, downloaded and cached.
- `LocalFrames`: frames on this host, found by GWDataFind, with their frame types and channels named.
- `StrainUnavailable`: a source could not serve the strain asked for; it never falls back to another.
- `UpsamplingRefused`: a sample rate above what the source publishes.
- `GlitchType`: the Gravity Spy glitch classes.
- `get_glitch_times`: Gravity Spy glitch times.
- `get_glitch_times_with_labels`: Gravity Spy glitch times with their classes.
- `get_glitch_segments`: windows around Gravity Spy glitches.
- `get_glitch_type_from_index`: the glitch class of an index.
- `get_glitch_catalogue`: the Gravity Spy catalogue.
- `TransientIndex`: an index of transients and their windows.
- `EventConfidence`: confident or marginal GWOSC events.
- `SourceType`: the source class of an event.
- `search_events`: GWOSC events by name, time or parameters.
- `get_event_catalogue`: the GWOSC event catalogue.
- `get_all_event_times`: the GPS times of every GWOSC event.
- `get_confident_event_times`: the GPS times of confident events.
- `get_marginal_event_times`: the GPS times of marginal events.
- `get_event_times_by_type`: event times by source class.
- `get_confident_events_with_params`: confident events with their parameters.
- `get_event_veto_segments`: the segments real noise excludes around events.

## Injections

- `InjectionGenerator`: injects waveforms into a dataset's noise.
- `WaveformGenerator`: the base of every waveform generator.
- `CBCGenerator`: compact binary coalescences (ripplegw), the one public CBC path.
- `Approximant`: the CBC waveform models.
- `WNBGenerator`: white-noise bursts.
- `IncoherentGenerator`: a different waveform in each detector.
- `WaveformParameter`: one waveform parameter and its label.
- `WaveformParameters`: the waveform parameters a dataset can return.
- `ScalingMethod`: how injections are scaled (SNR, HRSS or HPEAK).
- `ScalingType`: one scaling quantity.
- `ScalingTypes`: the scaling quantities.
- `ScalingOrdinality`: whether a scaling is measured before or after projection onto the detectors.
- `calculate_hrss`: the root-sum-square strain of waveforms.
- `calculate_hpeak`: the peak strain of waveforms.
- `WaveShape`: the simple waveform shapes.
- `PeriodicWaveGenerator`: sines, squares, sawtooths and triangles.
- `SineGaussianGenerator`: sine-Gaussians.
- `ChirpletGenerator`: chirplets.
- `RingdownGenerator`: ringdowns.
- `wnb`: white-noise bursts as arrays.

## Signal processing

- `psd`: the Welch power spectral density of strain, as a `PowerSpectrum` (the one public PSD).
- `PowerSpectrum`: a PSD kept scale-free: the normalised spectrum and its amplitude scale.
- `SpectralConfig`: the settings of a PSD estimate and of whitening.
- `snr`: the optimal SNR of injections in noise.
- `scale_to_snr`: scale injections to a target SNR.
- `whiten`: whiten strain by its offsource PSD.
- `Whiten`: the whitening Keras layer.
- `WhitenPass`: the layer that crops the onsource and scales each example to unit RMS, without whitening.
- `rolling_pearson`: the rolling Pearson correlation between detectors.
- `spectrogram`: the spectrogram of strain.
- `spectrogram_shape`: the shape `spectrogram` returns.
- `resample`: resample strain.
- `NonFiniteWarning`: warned when a NaN or infinity is replaced.

## Matched filtering

- `MatchedFilter`: a matched-filter search over a template bank.
- `MatchedFilterLayer`: the matched filter as a Keras layer.
- `TemplateBank`: a CBC template bank, placed or bundled.
- `TemplateGrid`: a grid of template parameters.
- `matched_filter_fft`: the matched-filter SNR time series.
- `optimal_snr`: the optimal SNR of a template.

## Models

The example models (`Gabbard2017`, `MatchedFilterBaseline`) come with their training recipe, which
validates as it trains: they need the `viz` extra.

- `Gabbard2017`: the CNN of Gabbard et al. (2017).
- `MatchedFilterBaseline`: matched filtering as a baseline model.
- `MatchedFilterBaselineConfig`: its configuration.
- `Model`: a gravyflow model: its network, training run and run record.
- `BaseLayer`: the base of the genome's layers.
- `Reshape`: the reshape layer.
- `DenseLayer`: a dense layer gene.
- `FlattenLayer`: a flatten layer gene.
- `ConvLayer`: a convolutional layer gene.
- `PoolLayer`: a pooling layer gene.
- `DropLayer`: a dropout layer gene.
- `BatchNormLayer`: a batch-normalisation layer gene.
- `WhitenLayer`: a whitening layer gene.
- `WhitenPassLayer`: a whitened-input layer gene.
- `Population`: a population of models for the genetic search.
- `PopulationSector`: one sector of a `Population`.
- `HyperParameter`: a hyperparameter and its distribution.
- `HyperInjectionGenerator`: injection settings as hyperparameters.
- `ModelGenome`: a model's hyperparameters as a genome.

## Validation (the `viz` extra)

- `Validator`: validate a model: efficiency, false-alarm rate, ROC and the dashboard.
- `ValidationConfig`: the settings of a validation.

## Plotting (the `viz` extra)

- `generate_strain_plot`: a Bokeh plot of strain.
- `generate_psd_plot`: a Bokeh plot of PSDs.
- `generate_spectrogram`: a Bokeh plot of a spectrogram.
- `generate_correlation_plot`: a Bokeh plot of a Pearson correlation.
- `generate_segment_timeline_plot`: a Bokeh timeline of segments.
- `generate_example_extraction_plot`: a Bokeh plot of extracted examples.

## Training, I/O and process utilities

- `crop_samples`: crop the padding from both ends of strain.
- `replace_nan_and_inf_with_zero`: replace non-finite values with zero.
- `ensure_directory_exists`: create a directory if it is missing.
- `replace_placeholders`: fill a configuration's placeholders.
- `snake_to_capitalized_spaces`: `snake_case` to `Capitalised Words`.
- `is_redirected`: whether the process's output is redirected.
- `load_history`: a model's saved training history.
- `save_dict_to_hdf5`: save a dictionary of arrays to HDF5.
- `model_file`: the file a model directory keeps its model in.
- `CustomHistorySaver`: the Keras callback that saves the training history.
- `EarlyStoppingWithLoad`: early stopping that restores the best weights.
- `PrintWaitCallback`: the Keras callback that reports early stopping's patience.
- `Heart`: a heartbeat a supervised process beats.
- `HeartbeatCallback`: the Keras callback that beats a `Heart`.
- `Process`: a supervised training process.
- `Manager`: runs `Process`es across GPUs.
- `send_email`: send a notification email.

## GPUs (the `cuda` extra)

- `env`: a context that selects free GPUs.
- `setup_cuda`: select a GPU and limit its memory.
- `find_available_GPUs`: GPUs with enough free memory.
- `get_gpu_memory_info`: each GPU's memory, from `nvidia-smi`.
- `get_tf_memory_usage`: TensorFlow's GPU memory use.
