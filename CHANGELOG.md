# Changelog

## 1.0.0 (unreleased)

The first release built as a wheel. It changes how GravyFlow is installed and imported: code written against the
repository before it may need the changes below.

### Breaking changes

- **`import gravyflow` sets nothing.** It used to set `JAX_PLATFORMS=cuda,cpu`, `KERAS_BACKEND=jax`,
  `TF_CPP_MIN_LOG_LEVEL`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `LD_LIBRARY_PATH`, add a warnings filter and
  change the log levels of `jax`, `gwpy` and `gwosc`. Now set `KERAS_BACKEND=jax` yourself, before Keras is imported:
  the first use of a GravyFlow name without it raises an `ImportError` that says so. Set `JAX_PLATFORMS` and
  `XLA_PYTHON_CLIENT_PREALLOCATE` yourself if you relied on them.
- **The top level is a curated API**, listed in `docs/api.md`, and each name imports its module on first use. These
  left the top level; they are internal, importable from their modules under `gravyflow.src`:
  `estimate_psd` (`gf.psd` is the one public PSD), `PATH` (package paths are in `gravyflow.src.resources`),
  `data` (use `gf.Dataset`), `expand_tensor`, `batch_tensor`, `ensure_even`, `ensure_list`, `ShapeContract`,
  `ShapeEnforcer`, `explain_exit_code`, `calculate_fitness`, `roll_vector_zero_padding`, `generate_mask`,
  `is_not_inherited`, `batch_injection_parameters`, `RENAMED_SKY_KEYS` and `generate_transient_cache_path`. These are
  deleted: `get_current_repo`, `get_file_parent_path`, `transform_string`, `randomise_arguments` (it called a function
  that did not exist), `get_element_shape`, `calculate_sample_counts`, `Axis_BIS` and `Axis_BI`.
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
- **Model directories from before this release.** A `gf.Model` directory, or a recipe's run directory, saved without
  `run.json` (the record of the data split its weights were trained and selected on) can no longer be validated or
  trained: retrain it.
- **The deep George-Huerta (2017) weights** (66 MB each) are no longer in the package but attached to each release:
  fetch one once with `gravyflow.src.resources.download_weights(name)`.
- `NonFiniteWarning` is defined in `gravyflow.src.utils.numerics` (still `gf.NonFiniteWarning`): a warnings filter
  can name it there without importing Keras.

### Added

- `gravyflow.__version__`.
- `docs/api.md`, the public API, which a test holds equal to `gravyflow.__all__`.
- `gravyflow.src.resources`: the package's data files, and `weights(name)` and `download_weights(name)` for trained
  weights, bundled or released.

### Fixed

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
