# GravyFlow

TensorFlow tools to facilitate machine learning for gravitational-wave data analysis. 

# Installation Guide

## 1. Clone the Repository
GravyFlow can be installed by cloning the Git repository:

```bash
git clone https://github.com/mrknorman/gravyflow
```

## 2. Install Gravyflow
It is recommended to install GravyFlow within a new conda environment. GravyFlow requires Python 3.11:

```bash
conda create --name gravyflow python=3.13
conda activate gravyflow
```

Next, ensure pip is installed by running:

```bash
conda install pip 
```

Then, install GravyFlow and its requirements into your conda environment:

```bash
pip install -e .[cuda]
```

Note that GravyFlow is under active development, and you may encounter issues during installation. Ensure TensorFlow can recognize GPUs in your environment, as GravyFlow is optimized for GPU use and relies on vectorized GPU functions.

## 3. Quick start: a CNN on a laptop CPU

One command trains the CNN of Gabbard *et al.* (2017) on simulated binary black hole signals (IMRPhenomD, masses
5–95 M☉, optimal SNR 8–15) in whitened coloured Gaussian noise, then validates it. It runs offline, on a CPU:

```bash
python -m gravyflow.src.model.examples.train_gabbard_2017 --output-dir ~/gravyflow_runs/gabbard_2017_cpu --preset cpu \
  && python -m gravyflow.src.model.examples.validate_gabbard_2017 --run-dir ~/gravyflow_runs/gabbard_2017_cpu
```

- **Training** (about 3 minutes on an Apple M-series CPU): the paper's network at 2048 Hz (41,842 weights), 10 epochs of
  300 batches of 32 one-second windows, Nadam at 0.002; the best epoch by validation loss is saved as `model.keras`.
  The run directory keeps a checkpoint per epoch: the same command resumes an interrupted run, and refuses a
  directory holding a run of another configuration.
- **Validation** (about 3 minutes): 100,000 noise windows (100,000 s of background) and 100,000 injections at SNR
  0–15, from the validator's own seeds. It prints the score threshold for each false-alarm rate (in Hz, with its
  achieved rate), the efficiency by SNR bin at that threshold, and the ROC AUC, and writes the dashboard
  `validation/validation.html` in the run directory. Expect an AUC near 0.976 and, at a false-alarm rate of 1e-3 Hz,
  an efficiency near 0.69 at SNR 7–8 and 0.96 at SNR 9–10.
- **Reproducible from the seed** (`--seed`): the same seed gives the same batches, in the same order, from the same
  initial weights. For bit-identical weights from run to run, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`:
  with XLA's multi-threaded CPU kernels the whitened strain differs in its last bits between calls (see section 6),
  and training amplifies the difference.

Notebooks 07 and 08 in `example_notebooks/` walk through the same run.

## 4. Setup permissions:

Follow these guides for setting up permissions to access real data:

https://computing.docs.ligo.org/guide/auth/scitokens/
https://computing.docs.ligo.org/guide/auth/kerberos/

## 5. Public data sources

Real noise and glitches use public data only; no credentials are needed:

- Segments and observing-run bounds come from the GWOSC timeline and run API (https://gwosc.org). The default selection is `DATA` with the `BURST_CAT1` and `BURST_CAT2` vetoes applied, without CBC, burst, stochastic or detchar hardware injections (continuous-wave injections, which run throughout, are kept). Real noise also excludes every GW event GWOSC publishes (its whole signal in band; see `docs/acquisition.md`) and every Gravity Spy glitch (its whole extent).
- Glitches come from the Gravity Spy machine-learning classifications: Glanzer et al. 2023, *Class. Quantum Grav.* 40 065004, released on Zenodo (doi:10.5281/zenodo.5649212, v1.0.0, CC-BY-4.0). Please cite both if you use them. Each file is downloaded on first use and cached in `~/.gravyflow/cache` (override with `GRAVYFLOW_CACHE_DIR`).

Every cache gravyflow keeps lives under that one root, `GRAVYFLOW_CACHE_DIR` if it is set, else `~/.gravyflow/cache`: GWOSC timelines and events, Gravity Spy files, the glitch, event and partner caches (`glitch_cache_v3.h5`, `event_cache_v3.h5`, `partner_cache_v3.h5`), real-noise segment caches (`segments/`) and the GWOSC strain downloads (`astropy/`). Earlier versions kept noise segment caches in `./generator_data` or beside the package. gravyflow does not move them; it warns once when it finds one there, naming where to move it to reuse it. Glitch and event caches of an earlier format (`glitch_cache.h5`, `event_cache.h5`) are never read, nor deleted.

## 6. Randomness: one seed tree

Every random draw comes from one seed tree, rooted at a dataset's `seed` (default `gf.Defaults.seed`) and addressed
by the split (`group`), the epoch, the batch index and what the draw is for. A batch is rebuilt from its address alone,
and the splits never share a stream. Within a split, datasets of one seed draw the same streams (common random
numbers), so a noise-only dataset and a signal dataset share their noise, as an ablation wants. Datasets that must be
independent, such as the classes of a classifier, take distinct names (`gf.Dataset(..., name="noise")`), each a
scope of the tree, or are the pools of one `gf.ComposedDataset`. Old seeds give new data. Every draw is bit-identical between rebuilds; on CPU,
XLA's multi-threaded kernels can still round the projected strain differently from one call to the next, in the last
bits of float32. For bit-identical batches, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` before importing
gravyflow (the results then differ, also in the last bits, from multi-threaded ones). The replaced seed arguments (`Distribution(seed=)`,
`generate(seed=)`, `reseed`, `Network(seed=)`, `Curriculum(seed=)` and the like) still work for one release, with a
`DeprecationWarning`. These changes have no such period:

- `ds[k]` is batch k, and `next(iter(ds))` is always batch 0. Step through batches with `next(ds)`, or index them.
- Under `model.fit`, a training split takes Keras' epoch: pass `callbacks=[ds.epoch_callback()]`. Without it, Keras
  epoch e would draw the batches of epoch e + 1, and a warning says so. With it, `fit(steps_per_epoch=)` must be
  unset or the dataset's length (it is refused otherwise), and a second `fit` continues only with `initial_epoch`:
  without it, it starts again at epoch 0 and serves the first's batches.
- `gf.Model` gives its training dataset `steps_per_epoch = num_examples_per_epoc // num_examples_per_batch` from its
  training config, and refuses `dataset_args` with another `steps_per_epoch`; `Model.train` passes `fit` no
  `steps_per_epoch` for a dataset with a length. Its Keras model's output is named (a dict, as the dataset's targets
  are), and it is saved to `model_path / "model.keras"` (`gf.model_file`): Keras 3 saves whole models only as
  `.keras` files.
- `ds[k]` and `ds.epoch` take integers only; a float, bool or string was truncated to an integer batch.
- `gf.project_wave` takes no seed and needs every sky angle; nothing is drawn there.
- `Curriculum` has no progress counter: `current_epoch` is gone, and `at(epoch)` gives the distribution of an epoch.
- `Distribution.sample(n, rng)` returns an ndarray.
- Splits are repartitioned by one `gf.SplitSpec` (`docs/acquisition.md`, section 2): GPS time is cut into absolute
  8192 s blocks from GPS 0, each hashed to test (10%), validate (10%) or train (80%), with 128 s between splits. Every
  route (a dataset, a composed dataset's pools, `NoiseObtainer`, an obtainer called directly, the validator) reads
  the data obtainer's `split_spec`, so a GPS time is in one split whatever reads it. A transient record takes its
  footprint's split; one that crosses an edge between splits is in no split. Noise and records have moved between
  splits: do not evaluate a model trained on an earlier release's split on this release's validate or test split.
  A model's split is its data's: `gf.Model.train` and the recipes' `TrainingRun.train` read it from their
  training and validation data (a gravyflow dataset names its `split_spec` and `group`), refuse data that names
  none or is split by two specs, and record the fingerprint and the split that selected the model in `run.json`
  beside the model. The record belongs to the weights: every model gravyflow trains or loads carries it
  (`TrainingRun.train`'s, `RunDirectory.load_model`'s, a `gf.Model`'s), weights that carry one never train on
  another split, whatever `force_retrain` says, and the validator reads it from the model, refusing a model split
  by another spec and the split that selected it.
- Every entry point needs a named split: `group='train'`, `'validate'` or `'test'`. `'all'` and any other name raise
  `KeyError`. `gf.DEFAULT_SPLIT_FRACTIONS`, `NoiseObtainer(groups=)`, `get_valid_segments(groups=)` and
  `TransientIndex.assign_groups` are gone: give the data obtainer `split_spec=gf.SplitSpec(fractions=...)`, or another
  `version` for an independent split (cross-validation).
- Real events are evaluation data: every confident event is in `test`, named events too, and an obtainer of events
  alone refuses `'train'` and `'validate'` unless it is built with `train_on_real_events=True`.
- An obtainer called for a split serves that split, called directly or through a `NoiseObtainer`; called again for
  another, it rebuilds its segments (it served the first split's, and a fresh real-noise obtainer called directly
  served nothing). A generator it gave out for the first split then raises `RuntimeError`.
- `TransientIndex.assign_splits` and `TransientDataObtainer.build_feature_index` take no seed.
- A glitch's example reads each requested detector the glitch was not seen in at a partner time of its own, in the
  record's split and block, never at the glitch's time, and labels that detector noise (`docs/acquisition.md`,
  Transient detectors and partners); it read every detector over the glitch's window. A record with no partner time
  is not served, and is counted under `"no partner"`. Partners have their own cache, `partner_cache_v3.h5`.
- The model recipes take one root seed: `TrainingRun.train`'s `training_data` is called with the seed alone (it
  took the seed and the first epoch), and `split_seeds`, `TrainingRun.train_seed` and
  `GeorgeHuerta2017.dataset(first_epoch=)` / `dataset_regression(first_epoch=)` are gone; resume sets `ds.epoch`.
- A recipe run is reproducible from its seed: `fit` serves each epoch's batches in index order (`shuffle=False`;
  Keras' shuffle drew from Python's unseeded global random), and `TrainingRun.train` builds the model under
  `keras_draws`, so its initial weights and dropout seeds come from the seed tree. Each checkpoint's record keeps its
  dropout seed state, which Keras does not save, so a resumed run trains as an uninterrupted one (bit for bit with
  the single-threaded `XLA_FLAGS` above; otherwise the whitened strain's last bits differ). A model with dropout
  saved in a run directory before this release records no seed state, and loading it (to resume or to validate) is
  refused by name: train it again.
- `gf.Model`'s binary output is a logit, not a sigmoid probability: it trains with
  `BinaryCrossentropy(from_logits=True)` and reports `BinaryAccuracy` at a threshold of 0, and the validator (which
  refused a probability output) scores it. `Model.validate(validate_args, config, model_path)` takes a
  `ValidationConfig` and returns the `Validator`. A checkpoint that exists but cannot be loaded raises `ValueError`;
  `Model.load` returned an untrained model in its place.

## 7. Units: physical strain

Every array gravyflow reads, returns or caches is physical strain: noise of every type but white (which has no
detector, and unit variance), injections, `ONSOURCE`, `OFFSOURCE` and `INJECTIONS`, the HRSS and HPEAK targets and
labels, and the transient caches. Whitened outputs and SNRs are dimensionless, as before. The square of strain
(about 1e-42) is below float32's normal range, so each step that squares data (PSD estimation, whitening, the SNR,
the matched filter, HRSS and HPEAK, the rolling Pearson correlation) divides by the data's peak first and restores
it after; whitened data, correlations and SNRs are identical, to float32 rounding, at any input amplitude. A PSD
array given to `gf.matched_filter_fft` or `gf.optimal_snr` is normalised by its median in float64, so give it as
float64: a float32 PSD of strain has already underflowed, and is refused.

Where the result itself is a square of strain, float32 cannot hold it:

- `gf.psd()` and `gf.estimate_psd()` return a `PowerSpectrum`, which keeps the normalised PSD and its amplitude scale
  apart; `.value()` gives the PSD in float64 NumPy, and `.frequencies` its grid.
- `gf.spectrogram()` and `gf.generate_spectrogram()` refuse an input whose power would underflow, such as raw strain,
  with a `ValueError`: take the spectrogram of whitened data (`SPECTROGRAM_ONSOURCE` does).
- A network cannot learn from raw strain: its gradients and normalisation statistics underflow. `gf.WhitenPass`, the
  genetic search's ablation against whitening, therefore scales each example to unit RMS over its detectors and
  samples; a model of your own on `ONSOURCE` must normalise its input too.
- WHITE noise is dimensionless, so an HRSS or HPEAK target (a strain amplitude) cannot be injected into it: the
  dataset (and a `gf.FeaturePool`) refuses the combination. Scale to SNR in WHITE noise; the HRSS and HPEAK labels of those injections are
  then in WHITE noise's units, not strain.

Earlier releases multiplied strain by an internal scale, `gf.Defaults.scale_factor`. It is gone:

- `ONSOURCE`, `OFFSOURCE` and `INJECTIONS` are no longer multiplied by that scale. A model trained on raw `ONSOURCE`
  (through `gf.WhitenPass`, which now scales each example to unit RMS) must be retrained; models on whitened inputs,
  such as the shipped examples, are unaffected.
- `gf.psd()` returns a `PowerSpectrum` instead of `(frequencies, psd)`: `spectrum.frequencies, spectrum.value()`.
- `scale_factor=` arguments (`gf.Dataset`, `gf.ComposedDataset`, `gf.NoiseObtainer`, an obtainer called directly,
  `gf.InjectionGenerator` and its `add_injections_to_onsource`, `gf.generate_strain_plot`, `gf.BatchConfig`,
  `colored_noise_generator`) are ignored for one release, with a `DeprecationWarning`. `gf.scale_to_snr(return_scale_factor=)` is `return_factor=`, with the same warning.
- `gf.Defaults.scale_factor` is removed: reading it raises `AttributeError`, and `gf.Defaults.set(scale_factor=)`
  raises one that says why. `IFOData.scale` is removed.
- A validator checkpoint of format 3 is refused (format 4 holds physical strain); validate the model again.
- A waveform's reference time (t = 0: a burst's peak, a CBC's merger) is always on a sample, sample N // 2 of its
  buffer, as in LAL's burst waveforms. A buffer of odd length put it half a sample off the grid, which lowered a
  sampled peak by cos(pi f / fs) (0.953 at 200 Hz and 2048 Hz).

## 8. Test Gravyflow (optional)

GravyFlow includes PyTest for testing its functionality. To run tests:

```bash
pytest gravyflow
```

Note: Tests may fail due to unavailable GPU memory if GPUs are currently under heavy use.
