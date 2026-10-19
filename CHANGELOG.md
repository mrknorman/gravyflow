# Changelog

## 1.0.0 (unreleased)

The first release built as a wheel. It changes how GravyFlow is installed and imported: code written against the
repository before it may need the changes below.

### Breaking changes

- **`gf.LocalFrames()` names no O4 frames by default.** Its table named LVK's O3 frame type and channel
  (`HOFT_C01`, `DCS-CALIB_STRAIN_CLEAN_C01`) for O4a and O4b too, which GWOSC's O4 releases do not come from
  (O4a `GDS-CALIB_STRAIN_CLEAN_AR`, O4b `DCS-CALIB_STRAIN_CLEAN_AR01`, Virgo `Hrec_hoftRepro1AR_16384Hz`). Reading
  O4 from local frames now names them (`frametypes=`, `channels=`); an unnamed run is refused by name.
- **O4 is a default observing run.** `DEFAULT_OBSERVING_RUNS` is O1, O2, O3 and O4, and every default reads it: an
  obtainer given no runs (so `gf.NoiseObtainer(noise_type=REAL)` and a transient obtainer), the validation bank's real
  events and the segment timeline plot. Every default real-data dataset changes, with its fingerprint; split blocks are
  absolute GPS, so O1 to O3's time keeps its splits. O4's noise has no glitch veto (`gf.NoGlitchVetoWarning`), and
  glitches are read only in the runs Gravity Spy's release covers (O1 to O3b, H1 and L1): over the default runs, O1
  to O3's, under a `gf.PartialGlitchCoverageWarning` naming the runs and detectors left without glitches. Name the runs to keep the old default: `observing_runs=[gf.ObservingRun.O1, gf.ObservingRun.O2,
  gf.ObservingRun.O3]`.
- **The aLIGO `O3_LOW` and `DESIGN` noise curves stop where their tables do** (4995.38 Hz and 5000 Hz). Their rows
  above that were lalsimulation's extrapolation of its table (1.9 times the last tabulated ASD by 8191 Hz), not
  tabulated values; no curve is extrapolated now, so coloured noise from them at 16384 Hz (any sample rate whose
  Nyquist frequency is past the table) is refused by name. Every row below the cut is unchanged.
- **One manifest of pinned remote files, one explicit fetch.** `gravyflow.src.resources.download_weights` and
  `RELEASE_WEIGHTS` are gone: `resources.MANIFEST` holds the pinned remote files (trained weights) as `gf.RemoteFile`s (URL, size, SHA-256, licence, redistributable), and `gf.fetch(name)`
  downloads one into the strain store's `files/`, checked, and is the only download of a pinned file; an explicit
  fetch hashes a file already in the store again. GWOSC strain and Gravity Spy's catalogues keep their own fetchers.
  `resources.weights(name)` refuses a release download not yet fetched, naming the fetch.
- **`import gravyflow` sets nothing.** It used to set `JAX_PLATFORMS=cuda,cpu`, `KERAS_BACKEND=jax`,
  `TF_CPP_MIN_LOG_LEVEL`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `LD_LIBRARY_PATH`, add a warnings filter and
  change the log levels of `jax`, `gwpy` and `gwosc`. Now set `KERAS_BACKEND=jax` yourself, before Keras is imported:
  the first use of a GravyFlow name without it raises an `ImportError` that says so. Set `JAX_PLATFORMS` and
  `XLA_PYTHON_CLIENT_PREALLOCATE` yourself if you relied on them.
- **The top level is a curated API**, listed in `docs/api.md`, and each name imports its module on first use. These
  left the top level; they are internal, importable from their modules under `gravyflow.src`:
  `estimate_psd` (`gf.psd` is the one public PSD), `PATH` (package paths are in `gravyflow.src.resources`),
  `data` (use `gf.Dataset`), `ensure_even`, `ensure_list`, `ShapeContract`, `ShapeEnforcer`, `generate_mask`,
  `batch_injection_parameters`, `RENAMED_SKY_KEYS` and `generate_transient_cache_path`. These are deleted: `get_current_repo`, `get_file_parent_path`, `transform_string`,
  `randomise_arguments` (it called a function that did not exist), `get_element_shape`, `calculate_sample_counts`,
  `Axis_BIS`, `Axis_BI`, and, since nothing in GravyFlow used them, `expand_tensor`, `batch_tensor`,
  `roll_vector_zero_padding` and `is_not_inherited`.
- **The CUDA-cluster tooling was removed from the library.** `gf.env`, `gf.setup_cuda`,
  `gf.find_available_GPUs` and `gf.get_tf_memory_usage` left the public API, with the GPU monitor script
  (`gravyflow/src/tools/gpu_checker.py`, which called two functions that did not exist). Selecting free GPUs and
  running on CUDA clusters is no longer part of GravyFlow. A `with gf.env():` block needs nothing in its place:
  GravyFlow runs on whatever devices JAX sees.
- **The cluster tools are removed from the library**: the genetic model search and the multi-process training
  supervisor it ran on. These left the public API: `gf.Population`, `gf.PopulationSector`, `gf.ModelGenome`,
  `gf.HyperInjectionGenerator`, `gf.Process`, `gf.Manager`, `gf.Heart`, `gf.HeartbeatCallback`, `gf.send_email` and
  `gf.get_gpu_memory_info`, with the internal `calculate_fitness` and `explain_exit_code`, `Model.from_genome`,
  `Model`'s `genome` argument, `Model.load`'s `genome`, `load_genome` and `dataset_args`, `Model.train`'s `heart`
  (pass a callback in `callbacks` instead), the hyperparameters' and layers' `mutate` and `crossover`, and
  `Model.crossover`, `Model.mutate` and `Model.test` (the first two could not run: they called `Model` with too few
  arguments). What a single host needs stays: `gf.Model` builds, trains, validates and loads as before, from the same
  seeds; `gf.HyperParameter` and the layer specifications keep their seeded draws (`reseed`, `randomize`);
  `Model.validate`'s `heart` takes any object with a `beat()` method. `Model.restore` is new: it loads the checkpoint
  in a model's directory into a model built another way, as `Model.load` does for its blueprints.
- **Unused code is deleted** (none of it had a caller in GravyFlow): from `gravyflow.src.dataset.acquisition`,
  `concatenate_batches`, `ifo_canonical_key`, `NoiseSegment`, `SegmentCollection` and the obtainers'
  `compress_segments` (use `coalesce_segments` in `gravyflow.src.dataset.acquisition.gwosc_data`),
  `calculate_bin_indices`, `get_segments_as_collection` and `reset_acquisition_stats`; `DataLabelSpec`; the
  compressed-segment helpers; `TransientIndex.to_segments`, `to_numpy_segments`, `invalidate_cache`, `to_gps_array`
  and `get_kind_counts`; `MatchedFilter.template_masses`; `Defaults.quantize_batch_size`, `quantize_sample_rate` and
  their tiers; the rotation matrices; the conditioning classes (`DataConditioner` and its subclasses); the Gamma,
  truncated-normal and beta-prime output layers and losses; and the tensor helpers `pad_if_odd`, `round_to_even`,
  `pad_to_power_of_two`, `check_tensor_integrity` and `print_active_gpu_memory_info`.
- **`SegmentOrder.CHRONOLOGICAL` sorts.** An obtainer's segments in that order are sorted by time, each detector's
  piece in turn; they used to keep the order they were found in.
- **An obtainer's `overrides` set observing-run attributes only** (`start_gps_times`, `end_gps_times`,
  `state_flags`), each to one span. Any other key is refused by name: a scalar setting such as `saturation` used to be
  wrapped in a list, which broke its first use.
- **One CBC path.** `generate_cbc_waveform` (in `gravyflow.src.dataset.features.waveforms.cbc`) is deleted, with
  `Defaults.quantize_duration` and `Defaults.STANDARD_DURATIONS`: it rounded the duration up to a tier and kept the
  last samples, so between tiers its merger moved or was cut off. `gf.CBCGenerator` is the one CBC path; its merger is
  at the window's centre.
- **A CBC's merger is the peak of its envelope**, |analytic h+|² + |analytic h×|², for every approximant. It used to
  be each model's own coalescence time, which sits up to tens of milliseconds from the peak, by a convention that
  differs between models (18 ms for IMRPhenomD at 95 + 95 solar masses). Every label that names the merger moves with
  it: the central time, the GPS times and the onsource placement. For IMRPhenomD the merger is LALSimulation's
  `SimInspiralChooseTDWaveform` t = 0, to the sample. For IMRPhenomXAS it is within the jitter of LAL's t = 0, the
  peak of h+² + h×². For IMRPhenomPv2 and the tidal model LAL keeps the model's own time, so the two differ.
- **Dependencies.** The core install no longer brings gwpy, gwdatafind, pycbc, bokeh, panel, holoviews, GitPython,
  SQLAlchemy, psycopg2-binary or ipykernel. Real data needs `gravyflow[realdata]`; plots, the validator and the example
  models need `gravyflow[viz]`; the notebooks need `gravyflow[notebooks]` and the tests `gravyflow[test]`. The `metal`
  extra is gone (jax-metal has no float64 and breaks `import jax`), and `dev` is now `test`.
- **JAX 0.12.** GravyFlow needs JAX and jaxlib 0.12 (`>=0.12.0,<0.13`; it took 0.8.1 or later). A seed draws the
  same random streams as under JAX 0.9: the same random bits, uniforms and parameters. Batches differ only by float
  rounding (JAX 0.12's FFTs and float64 `erfinv` round differently). The whitened strain moves by up to 4e-5 of its
  peak, less than JAX 0.9's multi-threaded CPU kernels already move it. A dataset's fingerprint names the
  jax and jaxlib versions, so it changes, and a materialised set written under an earlier JAX is refused by name:
  write it again. The `cuda` extra is now `jax[cuda12]` at the same release, which installs NVIDIA's libraries at
  the versions JAX's CUDA plugin declares (cuDNN 9.10.2 or later) in place of the CUDA 12.5 and cuDNN 9.8 pins.
  The `tpu` extra installs JAX 0.12's libtpu.
- **Model directories from before this release.** A `gf.Model` directory, or a recipe's run directory, saved without
  `run.json` (the record of the data split its weights were trained and selected on) can no longer be validated or
  trained: retrain it.
- **The deep George-Huerta (2017) weights** (66 MB each) are no longer in the package but attached to each release:
  fetch one once with `gf.fetch("model/<name>")`.
- `NonFiniteWarning` is defined in `gravyflow.src.utils.numerics` (still `gf.NonFiniteWarning`): a warnings filter
  can name it there without importing Keras.
- **The recipe's `materialise(batches, num_batches, root)`** writes a gravyflow dataset's batches as a materialised
  set under `root` and serves them from it; it refuses batches that are not a gravyflow dataset's (`TypeError`).
  `FixedBatches(batches, num_batches)` is the first batches of a dataset that serves them from its set; it no longer
  holds a list. It serves the set's own epoch every epoch: its `on_epoch_end` no longer moves its dataset on.
- `config_fingerprint` (and so a checkpoint's, a validation's and a dataset's fingerprint) refuses a NaN or infinite
  setting by name (`UnfingerprintableError`).
- **`waveform_generators` takes a list (or a single generator), no longer a dict.** The dict form crashed in
  `gf.InjectionGenerator`, and its `excluded` and `exclusive` keys were read nowhere. A dict is refused by name
  (`TypeError`) in `gf.Dataset` and `gf.InjectionGenerator`. `Prefetcher.close` is gone: nothing called it.
- **MLGWSC-1 support was removed before this release.** The default split's reservation of the challenge's O3a test
  segments is gone, with `gf.Reservation`, `gf.MLGWSC1_O3A`, `gf.CHALLENGE_FAITHFUL`, `SplitSpec.reserved`,
  `SplitSpec.mode` (and a dataset's `seed_metadata["split_mode"]`), the challenge-faithful mode and the shipped
  segment list; so is `gf.MLGWSC1Stream`, which generated the challenge's first dataset. A `gf.SplitSpec` holds out
  only its guards again, and its fingerprint names no reservation: the default split's is `ad5a8b11f77a360c`, as
  before the reservation. `gf.Triggers` and its file format, the external-model runner, `gf.fetch` and
  `gf.NoiseProfile.ZERO_DET_HIGH_POWER` stay.

### Added

- **Injection records for analysis.** A dataset returns, when asked: `ReturnVariables.IN_BAND_DURATION_SECONDS`
  (B, G), a CBC's time from the band's lower frequency to its merger (the envelope's peak) by its waveform model,
  not cut by the window (`gf.CBCGenerator.in_band_duration_seconds`; NaN for other sources);
  `ReturnVariables.WHITENED_PEAK` (B, G, I), the peak of the injection's whitened strain over the whitened noise's
  deviation (`gf.whitened_noise_std`), and `NETWORK_WHITENED_PEAK` (B, G), their root-sum-square; and
  `ReturnVariables.BNS_RANGE_MPC` (B, I), each example's BNS range from the batch's PSD (`gf.bns_range_mpc`,
  LIGO's SenseMonitor range; refused for WHITE noise). `gf.INJECTION_RECORDS` lists the per-injection records.
- **A timing harness.** `gf.time_batch` (throughput: compute-seconds per second of strain, over a fixed segment),
  `gf.time_streaming` (latency: the per-step compute time of a streaming replay, its median, p90, p99 and max over at
  least 1000 steps, plus the algorithmic wait the caller declares) and `gf.compare_abba` (two callables run A, B, B, A,
  reporting the four walls, both pair means and the spread of the two A runs) time any model or search callable by the
  wall time of the whole call. Each `gf.Timing` keeps the walls, the cold-start time apart, the strain seconds, the
  thread count, the machine's description and the host's load at the start and end (a timing taken under load is
  flagged), and gives its numbers as runtime records (quantity, statistic, value, units, normalised, hardware, scope,
  mode, grade `measured`). See README section 11.
- **Standard datasets**: `gf.standard_dataset("O4 standard")` and `gf.standard_dataset("O3 open-weights")`, each
  with a Gaussian twin (`gaussian_twin=True`): fixed, fingerprinted test data (the test split of gravyflow's
  partition on H1 and L1 at 2048 Hz, default flags, events vetoed and listed apart, no glitch veto), with a pinned
  segment list, event list and binary-black-hole injection population shipped in `res/standard/`. Two injection
  sets are drawn from a fixed seed on the same segments: the volume set (100,000, uniform in distance out to each
  mass's reach, where the best-case source has network SNR 2, weighted so the sensitive volume is exact) and the SNR
  set (50,000, network SNR uniform over 0 to 20); the population is IMRPhenomXPHM, with precessing spins. A changed
  pinned file or table is refused (`gf.PinnedFileChanged`). See `docs/api.md`. `gf.StandardDataset`,
  `gf.STANDARD_DATASETS`, `gf.GaussianTwin`, `gf.BBHPopulation`, `gf.VolumeSet`, `gf.SNRSet`, `gf.Reach` and
  `gf.InjectionSchedule` are exported.
- **O4 noise profiles**: `gf.NoiseProfile.O4_HIGH` (aLIGO's high projection for H1 and L1, Advanced Virgo's for
  V1) and `gf.NoiseProfile.O4_LOW` (aLIGO's low projection, H1 and L1 only), LIGO-T2000012-v2's O4 tables as
  published: the observing scenarios' projections (Living Rev. Relativ. 23, 3 (2020)), not measurements.
- **Materialised sets.** `ds.materialise(root, batches=N)` (or `k_range=`) writes a dataset's batches once, as
  content-named files under a manifest that records its fingerprint (`ds.fingerprint`), platform, contract, split,
  epoch and range; `ds.load_materialised(set)` serves `ds[k]` from it, verified on every read (README §7). The
  fingerprint records the versions of Python and of every library that makes batch bytes
  (`gravyflow.src.utils.fingerprint.library_versions`). A dataset read in order is refused (`NotIndexAddressed`).
- **The platform record and a `tpu` extra.** `ds.seed_metadata["platform"]` and every materialised manifest record
  the backend and device kind, `XLA_FLAGS`, JAX's matmul precision, the CBC stage's device and flag, and the host's
  CPU architecture, so a set resumed on another architecture is a set of its own. Training and `ds.materialise`
  warn on any backend whose determinism flag is missing (CPU, GPU) or unknown (TPU). A dataset's CBC waveforms are
  generated on the host's CPU; a process without a CPU backend refuses the first CBC batch (`NoHostBackend`)
  (README, "Platforms").
- **Triggers in the MLGWSC-1 format.** `gf.Triggers` (GPS `time`, ranking `stat`, timing tolerance `var`) reads and
  writes the challenge's trigger file, byte for byte the file its examples write and its `evaluate.py` reads.
- **External models, run as published.** An `gf.ExternalCard` describes a published search with released weights:
  its input and output contracts (`gf.InputContract`, `gf.OutputContract`: the signal column and direction of its
  published statistic), its weights (`gf.Weights`, each a `gf.RemoteFile` pinned by sha256 with its licence), the
  rows its authors published (`gf.PublishedRow`), its declared training data, and the version and environment lock
  of its adapter; `card.fingerprint` hashes exactly what decides a run's outputs, and the published rows and
  training data are provenance outside it, carried with each result. An adapter implements `gf.ExternalSearch` in a package of its
  own, published under the entry-point group `gravyflow.external`; gravyflow imports none, nor their frameworks.
  `gf.ExternalRunner` runs one in its own environment (`gf.AdapterEnvironment`) as an isolated `python -I`
  subprocess under a deadline, on a whole `gf.SearchSegment`, with its weights fetched only by `gf.fetch`; its
  `gf.ExternalResult` names the card's mode (`gf.AS_PUBLISHED`) and fingerprint. A missing adapter is refused by name
  (`gf.AdapterMissing`), as is another release of it or another environment (`gf.AdapterMismatch`) (README §10).
- **`gf.NoiseProfile.ZERO_DET_HIGH_POWER`**: Advanced LIGO's zero-detuning, high-power design curve
  (LIGO-T0900288-v3, aLIGOZeroDetHighPower), for H1 and L1 only; a profile refuses a detector it does not cover, by
  name.
- **Sensitive volume and distance.** `Validator.sensitive_volume_curve()` gives the sensitive volume (Gpc³) and
  distance (Mpc) at each target FAR, with three intervals: the injections' (exact binomial), the background's
  livetime, and a joint bootstrap over both whose ends are at least as far out as the background's (so it covers
  where the background allows only a trigger or two, or has only a few blocks). The volume is Euclidean (no
  cosmology) and the population's masses are in the detector frame, as in MLGWSC-1's `evaluate.py`. It is measured,
  as an injection campaign measures it, on a set of its own (`volume_num_injections`, by default `num_examples`): CBC
  sources at physical distances drawn uniformly in volume out to the population's reach, or
  `volume_max_distance_mpc`; the reach is recorded. By default the reach is the distance beyond which no source of
  the population can have network SNR `volume_horizon_snr` (3), whatever its sky, orientation, masses and aligned
  spins, in the noise's PSD: it does not depend on the batch size, and a population whose masses are unbounded, or
  whose spins precess, refuses it by name. The volume noise alone finds (the chance a noise window passes, times the
  reference volume) is recorded at each target, and a point where it exceeds `volume_noise_floor_fraction` (0.1) of
  V is marked noise-dominated. A point where the reach's outermost shell is found more often than noise is (an exact
  one-sided test) is marked reach-limited: the detector finds sources beyond the reach, which V cannot count.
  `SensitiveVolumeCurve.quote` refuses a noise-dominated or reach-limited point by name. The dashboard plots them
  beside efficiency, marking both. A burst population, or WHITE noise, has none.
- **Injections at a physical distance.** `ScalingMethod(value=distances, type_=WaveformParameters.DISTANCE_MPC)`
  places each generated CBC source at a target distance (Mpc), scaling it by the ratio of its drawn distance to the
  target, and no other measure; WHITE noise refuses it, as it refuses HRSS and HPEAK.
- **Validation records are format 5.** A record holds the sensitive-volume set (`volume_data`). `Validator.load`
  refuses a record of any other format by name (`StaleCheckpointError`), and converts none: validate the model again.
  The new settings change the validation's fingerprint, so older checkpoints are not resumed.
- **`Background.bootstrap`** draws a bootstrap replicate of a background (its blocks, or its independent windows),
  `Background.windows_at_or_above` the number of its windows that pass a threshold, and
  `Background.window_exceedance` the chance that a noise window passes it.

- `gravyflow.__version__`.
- `docs/api.md`, the public API, which a test holds equal to `gravyflow.__all__`.
- `gravyflow.src.resources`: the package's data files, the manifest of pinned remote files (`MANIFEST`), `path(name)`
  and `weights(name)` to find a file bundled or fetched, and `fetch(name)` (`gf.fetch`), the one explicit download of a pinned file.
- `gf.RemoteFile` (see Breaking changes).
- `gf.NoGlitchVetoWarning`: noise with no public Gravity Spy labels (O4, and Virgo in any run) is served with no
  glitch veto under this warning, shown by default and naming the detector and time; it was a log message, hidden by
  default.

### Fixed

- O4's GWTC releases (GWTC-4.0 and GWTC-4.1 for O4a, GWTC-5.0 for O4b) were read as neither confident nor
  marginal: the event-time queries by confidence left them out (the noise vetoes always held them). They are
  confident catalogues now.
- A detector was asked for in every configured run, so Virgo over `ObservingRun.O4` asked GWOSC for an O4a
  timeline it does not publish (404, then the slow paginated fallback), and Virgo in O1 alone read as no data.
  `GWOSC_RUNS` now names each run's detectors, from GWOSC's run API: a detector is read only in the runs that
  publish it, and one no configured run publishes is refused by name.
- No GWOSC strain could be fetched: `gf.GWOSC()` asked gwosc for the release by name (`O3a_4KHZ_R1`), which
  gwosc 0.8 does not list (it names a run's bulk data by the run, `O3a`, and picks the release by the rate), so
  every real-noise read, in every run, was refused. It now names the run, and refuses any file GWOSC lists outside
  the release it names for that run and rate.
- Building or loading a `gf.Model` drew its initialisers and dropout seeds from Python's global random: it changed
  that state, and two models of one seed differed. They now come from a stream of the model's `seed`.
- The chirp-time helpers took the solar mass in seconds as G·1.98855e30/c³, 7.05e-5 above LALSimulation's
  `MTSUN_SI`, so their times to merger were 1.2e-4 short. They now use ripple's value, which is LAL's.
- `import gravyflow` failed after a clean install: it imported datashader, which no requirement named.
- A wheel or non-editable install lacked the package's data files, so coloured noise (its noise profiles), the bundled
  template bank and the shipped models were missing. The wheel no longer ships the tests or their data.
- `import gravyflow` took 3.6 s and imported 119 third-party packages; it now imports none.
- The SNR from an estimated PSD read low whenever the onsource was longer than a Welch segment: the inverse-bias
  correction, exact at the PSD's own bins, over-corrected the linearly interpolated PSD between them (SNR^2 8% low
  with 7 segments, 0.4% with 127). The SNR and the matched filter now interpolate the inverse PSD, which one factor
  corrects exactly at every bin (`PowerSpectrum.inverse`). SNR labels and SNR-scaled injections at onsources longer
  than `fft_duration_seconds` move by that much; at the default 1 s onsource and 1 s segments they do not move.
- `pearson` added 1e-5 to its denominator, so its coefficients depended on the data's scale (proportional rows
  correlated at 0.29 at a thousandth of the scale). It now divides exactly, and gives 0 for a row with no variance.
