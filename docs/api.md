# The public API

`import gravyflow as gf` gives the names below, and only these (`gf.__all__`; a test holds the two lists equal).
Each imports its module on first use, so `import gravyflow` itself is instant and changes nothing in the process.
Everything else under `gravyflow.src` is internal: importable, but free to change without notice.

gravyflow runs on Keras's JAX backend. Set `KERAS_BACKEND=jax` before Keras is imported (for example,
`export KERAS_BACKEND=jax` in the shell); the first use of a name refuses with that instruction otherwise.

`gf.__version__` is the package's version.

## Configuration and randomness

- `Defaults`: the defaults every dataset takes (sample rate, durations, seed, batch size).
- `WindowSpec`: the sample rate and the onsource, offsource and crop durations of one window. A rate or onsource that is not above zero, or an offsource or crop below zero, is refused by name as it is built.
- `SplitSpec`: how GPS time is split into train, validate and test.
- `DEFAULT_SPLIT`: the default `SplitSpec`.
- `EXCLUDED`: the label of time that belongs to no split.
- `RemoteFile`: a file published elsewhere, pinned by size and sha256, with its licence and whether gravyflow may redistribute it.
- `fetch`: download a file of gravyflow's manifest (or a `RemoteFile`) into the store, explicitly, hashing a copy already there again (a file the package ships is returned as shipped); nothing else downloads one.
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
- `BatchConfig`: the settings of batch generation. A batch size that is not a whole number of one or more is refused by name, as it is everywhere a batch size is taken.
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
- `NoGlitchVetoWarning`: warned when noise is served with no glitch veto (O4, Virgo: no Gravity Spy labels).
- `PartialGlitchCoverageWarning`: warned when glitches are requested over runs only some of which Gravity Spy covers (the covered runs' glitches are served).
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

## Standard datasets

Fixed, fingerprinted test data every model is measured on: results on one are comparable, results on two
are never mixed. Each preset's files ship with the package, pinned by sha256; the strain is not downloaded.

- `standard_dataset`: a preset by name (`"O4 standard"`, `"O3 open-weights"`), or its Gaussian twin
  (`gaussian_twin=True`).
- `STANDARD_DATASETS`: every preset, by name.
- `StandardDataset`: one preset: `segments` ((S, 2) GPS seconds, H1 and L1 both holding each whole: the
  test split of gravyflow's partition, gravyflow's default flags, every published event vetoed, no glitch
  vetoed), `events` (the published events in its test time, scored apart), `population`, `volume_set` and
  `snr_set`, `volume_injections` and `snr_injections` (the tables, the same for the twin),
  `sample_rate_hertz` (2048), `livetime_seconds`, `gaussian_twin()`, `fingerprint()` and `metadata()`.
- `GaussianTwin`: how a twin is made: coloured Gaussian noise from each segment's measured spectrum.
- `BBHPopulation`: a broad binary-black-hole population with precession and higher modes (IMRPhenomXPHM).
- `VolumeSet`: sources uniform in distance out to their masses' `Reach`, each with the weight `4 pi d^2 R` that
  makes the sensitive volume exact (the found weights' sum over the number injected).
- `Reach`: R(m1, m2), where the best-case source of those masses has network SNR 2: a pinned table of the
  best-case SNR at 1 Gpc and a safety factor for precession and higher modes.
- `SNRSet`: sources scaled at run time to a network optimal SNR uniform over a range (0 to 20).
- `InjectionSchedule`: when injections happen: one per slot per pass, mergers at least half a slot apart.
- `PinnedFileChanged`: refused: a preset's pinned file, or an injection table it makes, is not the one pinned.

## Injections

- `InjectionGenerator`: injects waveforms into a dataset's noise.
- `WaveformGenerator`: the base of every waveform generator.
- `CBCGenerator`: compact binary coalescences (ripplegw), the one public CBC path.
- `Approximant`: the CBC waveform models.
- `WNBGenerator`: white-noise bursts.
- `IncoherentGenerator`: a different waveform in each detector.
- `WaveformParameter`: one waveform parameter and its label.
- `WaveformParameters`: the waveform parameters a dataset can return.
- `ScalingMethod`: how injections are scaled (SNR, HRSS or HPEAK), or placed at a physical distance
  (`WaveformParameters.DISTANCE_MPC`).
- `ScalingType`: one scaling quantity.
- `ScalingTypes`: the scaling quantities.
- `ScalingOrdinality`: whether a scaling is measured before or after projection onto the detectors.
- `calculate_hrss`: the root-sum-square strain of waveforms.
- `calculate_hpeak`: the peak strain of waveforms.
- `INJECTION_RECORDS`: the records of each injection a dataset returns when asked, for later analysis: `IN_BAND_DURATION_SECONDS` (a CBC's time from the band's lower frequency to its merger, by its waveform model, whatever the window; NaN for other sources) and `WHITENED_PEAK` / `NETWORK_WHITENED_PEAK` (the peak of the injection's whitened strain in each detector over the whitened noise's deviation, `gf.whitened_noise_std`, and their root-sum-square). With the existing loudness labels (`ScalingTypes.SNR`, the optimal network SNR; `ScalingTypes.HRSS`, the source's root-sum-square strain) and `ReturnVariables.BNS_RANGE_MPC` (each example's BNS range per detector), they are the records a validation run keeps for each injection. The merger's GPS time at each detector is `START_GPS_TIME + CENTRAL_TIME x onsource_duration_seconds` for real noise; simulated noise has no GPS time (NaN), and its projection uses the generator's `reference_gps_time_seconds`.
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
- `whitened_noise_std`: the standard deviation of stationary Gaussian noise whitened over a band.
- `bns_range_mpc`: the BNS range (LIGO's SenseMonitor range, as LALSimulation's) of each row of a `PowerSpectrum`, in Mpc.
- `Whiten`: the whitening Keras layer.
- `WhitenPass`: the layer that crops the onsource and scales each example to unit RMS, without whitening.
- `rolling_pearson`: the rolling Pearson correlation between detectors.
- `spectrogram`: the spectrogram of strain.
- `spectrogram_shape`: the shape `spectrogram` returns.
- `resample`: resample strain.
- `NonFiniteWarning`: warned when a NaN or infinity is replaced.

## Matched filtering

- `StreamingMatchedFilter`: the matched filter over continuous strain, each detector filtered once into triggers with Allen's chi-squared, the sine-Gaussian veto, autogating and the PSD-variation statistic, ranked by PyCBC's `newsnr_sgveto_psdvar`; its `search` gives each detector's triggers and the network's, `score_injections` scores injections locally, and `device=` runs its float32 kernels on another device (an accelerator). Its triggers (`gravyflow.src.detection.DetectorTriggers`) keep every ranking input and derive the ranking from them exactly (never storing it; `ranked` changes its corrections; the inputs and the ranking's definition are immutable, so the cached ranking cannot go stale), carry their bank's identity and their ranking's definition, and save whole in a strict, versioned format that `load(path, bank)` checks; PSD variation requires strain with noise (`psd_variation=None` for noise-free strain); the same module has `autogate` (gating for any search's input) and `fit_templates` (per-template exponential background fits).
- `coincide`: the network triggers of per-detector triggers (the same template within the light travel time); a time slide is a call with one detector's triggers shifted.
- `MatchedFilter`: a matched-filter search over a template bank, on whitened windows.
- `MatchedFilterLayer`: the window matched filter as a Keras layer.
- `TemplateBank`: a CBC template bank, non-spinning or aligned-spin, placed or bundled.
- `TemplateGrid`: a grid of template parameters.
- `matched_filter_fft`: the matched-filter SNR time series. A PSD array is normalised on the host, so it cannot be a `jax.jit` argument (close over it, or pass a `PowerSpectrum`).
- `optimal_snr`: the optimal SNR of a template. A PSD array is normalised on the host, so it cannot be a `jax.jit` argument (close over it, or pass a `PowerSpectrum`).

## Triggers and external models

A search's triggers are written as the MLGWSC-1 challenge's trigger file, byte for byte what its examples write and
its `evaluate.py` reads. An external model (a published search with released weights, in another framework) runs
through an adapter, a package of its own that gravyflow never imports, in its own environment: a subprocess under a
deadline. Its results are in one mode, `AS_PUBLISHED`: the published statistic, never rescored as gravyflow's own
models are, and never through the `Validator`.

- `Triggers`: a search's triggers (GPS `time`, ranking `stat`, timing tolerance `var`); `Triggers.read` and `write` read and write the MLGWSC-1 trigger file.
- `SearchSegment`: the strain an external search runs on: every detector over the same samples, searched whole; `SearchSegment.from_strains` builds one from a data source's strain.
- `ExternalSearch`: what an adapter implements: `prepare(fetch)`, `preprocess(segment)` and `search(segment)`, published under the entry-point group `gravyflow.external`.
- `ExternalCard`: an external model: its adapter, the adapter's version and environment lock, its contracts, weights, published rows and declared training data; `card.fingerprint`, its identity, is the sha256 of exactly what decides a run's outputs (adapter name and version, lock, contracts, each weights file's name, sha256, format and framework, mode), never of its provenance (published rows, training data).
- `InputContract`: the input a model takes: rate, detectors in order, window, stride, context, whitening, band-pass and normalisation.
- `OutputContract`: the statistic it returns: what it is, its signal column, its direction, threshold, clustering and time offset.
- `Weights`: one released weights file: its `RemoteFile` (pinned by sha256, with its licence), format and framework.
- `PublishedRow`: a number a model's authors published, which a reproduction must match.
- `AS_PUBLISHED`: the mode of a model run as its authors released it.
- `AdapterEnvironment`: where an adapter runs: the environment's interpreter, its lock file, and extra directories on its path.
- `ExternalRunner`: run a card's adapter on a segment (`search`, `preprocess`) as a `python -I` subprocess under a deadline, in a temporary directory it removes; weights come only through `fetch`.
- `ExternalResult`: an external search's triggers with the card's name, mode, fingerprint and output contract, and its published rows and training data as provenance.
- `ExternalRunFailed`: an external run did not return its result; its log's end is in the message.
- `ExternalDeadlineExpired`: an external run outlived its deadline; every process in its session was stopped.
- `AdapterMissing`: the environment has no adapter by the card's name.
- `AdapterMismatch`: the environment is not the card's: another lock, another adapter release, or not an `ExternalSearch`.
## Models

The example models (`Gabbard2017`, `MatchedFilterBaseline`) come with their training recipe, which
validates as it trains: they need the `viz` extra.

- `Gabbard2017`: the CNN of Gabbard et al. (2017).
- `MatchedFilterBaseline`: matched filtering as a baseline model.
- `MatchedFilterBaselineConfig`: its configuration.
- `Model`: a gravyflow model: its network, training run and run record.
- `BaseLayer`: the base of a `Model`'s layer specifications.
- `Reshape`: the reshape layer.
- `DenseLayer`: a dense layer specification.
- `FlattenLayer`: a flatten layer specification.
- `ConvLayer`: a convolutional layer specification.
- `PoolLayer`: a pooling layer specification.
- `DropLayer`: a dropout layer specification.
- `BatchNormLayer`: a batch-normalisation layer specification.
- `WhitenLayer`: a whitening layer specification.
- `WhitenPassLayer`: a whitened-input layer specification.
- `HyperParameter`: a model or layer setting: a value drawn from its distribution by its seed.

## Validation (the `viz` extra)

- `Validator`: validate a model: efficiency against SNR (the rolling curve with pointwise bands, its sigmoid fit and
  50% and 90% SNR points, the binned record), false-alarm rate, sensitive volume and distance within the per-mass
  reach against false-alarm rate (Euclidean, detector-frame masses; on injections uniform in chirp distance out to
  the chirp reach, weighted by chirp mass, with injection, background and joint intervals), every interval at 1σ
  and 2σ and the chance of a noise-only pass subtracted, each quoted only with at least ten independent background
  events at or above its threshold (a volume where noise alone finds more than a tenth of what is found is quoted
  with a warning), ROC and the dashboard. By default in stream mode: the model slid along continuous raw strain at
  its stride, its scores clustered into triggers, its background the triggers of time slides of the data, each
  injection scored locally and found by the loudest trigger within 0.1 s of its merger; window mode
  (`mode="window"`) is kept for backward compatibility and reproduction.
- `generate_volume_3d_plot` (`gravyflow.src.validate.volume_3d`): the dashboard's 3D volume tab: the volume set's
  injections at sky direction times distance, found or missed at a false-alarm rate chosen with a slider, a sphere at
  the sensitive distance and the reach's outline, coloured by found state or chirp mass, at physical or scaled
  (chirp) distance; None for a record without sky positions. `volume_3d_levels` gives what it reads at each level.
  The metric family, from the same injections and background, each with its 1σ and 2σ intervals:
  `detection_roc` (detection probability against false-alarm rate per SNR band and for the population),
  `sensitive_vt` (VT), `binned_volumes` (by chirp mass, mass ratio and spin), `recovered_counts`,
  `matched_filter_ratio` (on the same injections), `timing_accuracy` (median and 90th percentile),
  `far_calibration` (on held-out background) and `real_event_summary` (statistics, FARs, counts per level and a
  training-period flag), at the levels `FAR_LEVELS_HZ` (one per month, year, ten and hundred years). The dashboard's
  tabs: Summary, Efficiency, Volume, ROC and false alarms, Parameter Space, Real events and the worst cases.
- `ValidationConfig`: the settings of a validation (`mode`, `stride_seconds`, `cluster_seconds`,
  `found_window_seconds`, `stream_segment_seconds` for stream mode).
- Wrapping time slides and the per-detector path (`gravyflow.src.validate`): `WrappedSlides` lays the network's
  segments end to end as one ring and reads detector j `j lag` further round it in each slide, wrapping round each
  segment and across segments, so every slide adds the whole ring of background (`background_seconds`) and a 30-day
  test set yields thousands of years; every lag between two detectors lies further than `minimum_shift_seconds`
  (the light-travel time plus the model's input length) round the ring from zero lag and from every other slide's,
  or is refused by slide. A model that scores each detector separately is scored once per detector
  (`score_detectors`), and each slide costs only the coincidence step (`slide_streams`, streams for
  `StreamBackground`); `joint_slide_streams` scores the same pairings with a model that sees every detector at once (a
  full run per slide). Stored matched-filter triggers (`DetectorTriggers`) take the same slides
  (`slide_coincidences`).
- `Glitches`: labelled glitches (detector, peak GPS time, class), from Gravity Spy's public release
  (`Glitches.from_gravity_spy`). With them the validator's glitch robustness (`gravyflow.src.validate.glitches`): the
  false-positive rate per glitch class at each false-alarm-rate level, with 1σ and 2σ intervals, from the model run
  over every labelled glitch; each background trigger's nearest glitch and its class; the volume on real data against
  its Gaussian twin. The dashboard's Glitches tab shows them.
- `TrainingCost`: what a model cost to train, measured where we train or quoted with its source for released
  weights; a validation record's `training_cost`. The validator's timing (`gravyflow.src.validate.timing`):
  throughput per second of strain and data-to-trigger latency of a model slid along a stream, median and tail, on
  the machine and at the thread count recorded, with the host's load; rounds of A-B-B-A; no ratio across hardware.
  The dashboard's Timing tab shows them.

## Plotting (the `viz` extra)

- `generate_strain_plot`: a Bokeh plot of strain.
- `generate_psd_plot`: a Bokeh plot of PSDs.
- `generate_spectrogram`: a Bokeh plot of a spectrogram.
- `generate_correlation_plot`: a Bokeh plot of a Pearson correlation.
- `generate_segment_timeline_plot`: a Bokeh timeline of segments.
- `generate_example_extraction_plot`: a Bokeh plot of extracted examples.

## Timing

Throughput, latency and A-B-B-A comparisons of any model or search callable, by the wall time of the whole call, on
the machine and at the thread count the caller declares. See README section 11.

- `time_batch`: throughput, the compute-seconds a call takes per second of strain it processes, over a fixed segment.
- `time_streaming`: latency, the per-step compute time of a streaming replay (median, p90, p99, max over at least
  1000 steps) plus the algorithmic wait the caller declares.
- `compare_abba`: two callables run A, B, B, A, reporting the four walls, both pair means and the spread of the two A
  runs.
- `Timing`: one timing's record: its walls, cold-start time, strain seconds, thread count, machine, host load and
  runtime records.
- `AbbaComparison`: the result of `compare_abba`.

## Training and I/O utilities

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
