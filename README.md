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

Every cache gravyflow keeps lives under that one root, `GRAVYFLOW_CACHE_DIR` if it is set, else `~/.gravyflow/cache`: GWOSC timelines and events, Gravity Spy files, the glitch and event caches (`glitch_cache_v3.h5`, `event_cache_v3.h5`), real-noise segment caches (`segments/`) and the GWOSC strain downloads (`astropy/`). Earlier versions kept noise segment caches in `./generator_data` or beside the package. gravyflow does not move them; it warns once when it finds one there, naming where to move it to reuse it. Glitch and event caches of an earlier format (`glitch_cache.h5`, `event_cache.h5`) are never read, nor deleted.

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
- Splits are repartitioned. Every entry point (a dataset, a composed dataset's pools, `NoiseObtainer`, an obtainer
  called directly, `TransientIndex.assign_groups()`, the validator) splits by `gf.DEFAULT_SPLIT_FRACTIONS`: 80%
  train, 10% validate, 10% test. They split 98/1/1 (an obtainer, and the datasets
  reading one directly) or 89/10/1 (`NoiseObtainer`), so one record could be in train by one route and in validate
  by another. A transient record's split is now a hash of its GPS key alone; it was the key and the seed. Records
  have moved between splits: do not evaluate a model trained on an earlier release's split on this release's
  validate or test split.
- Split fractions other than `gf.DEFAULT_SPLIT_FRACTIONS` (its keys, in its order, with its values) raise
  `ValueError` wherever they are given (`NoiseObtainer(groups=)`, `get_valid_segments(groups=)`,
  `TransientIndex.assign_groups(groups)`): they were taken silently by one route, which could then train on another
  route's validate or test records. Leave them unset; configurable fractions return with a split specification
  that every route reads.
- An obtainer called for a split serves that split, called directly or through a `NoiseObtainer`; called again for
  another, it rebuilds its segments (it served the first split's, and a fresh real-noise obtainer called directly
  served nothing). A generator it gave out for the first split then raises `RuntimeError`.
- `TransientIndex.assign_groups` and `TransientDataObtainer.build_feature_index` take no seed.
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

## 7. Test Gravyflow (optional)

GravyFlow includes PyTest for testing its functionality. To run tests:

```bash
pytest gravyflow
```

Note: Tests may fail due to unavailable GPU memory if GPUs are currently under heavy use.
