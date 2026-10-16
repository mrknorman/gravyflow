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

## 4. Setup permissions:

Follow these guides for setting up permissions to access real data:

https://computing.docs.ligo.org/guide/auth/scitokens/
https://computing.docs.ligo.org/guide/auth/kerberos/

## 5. Public data sources

Real noise and glitches use public data only; no credentials are needed:

- Segments and observing-run bounds come from the GWOSC timeline and run API (https://gwosc.org). The default selection is `DATA` with the `BURST_CAT1` and `BURST_CAT2` vetoes applied, without CBC, burst, stochastic or detchar hardware injections (continuous-wave injections, which run throughout, are kept). Real noise also excludes every GW event GWOSC publishes (its whole signal in band; see `docs/acquisition.md`) and every Gravity Spy glitch (its whole extent).
- Glitches come from the Gravity Spy machine-learning classifications: Glanzer et al. 2023, *Class. Quantum Grav.* 40 065004, released on Zenodo (doi:10.5281/zenodo.5649212, v1.0.0, CC-BY-4.0). Please cite both if you use them. Each file is downloaded on first use and cached in `~/.gravyflow/cache` (override with `GRAVYFLOW_CACHE_DIR`).

## 6. Randomness: one seed tree

Every random draw comes from one seed tree, rooted at a dataset's `seed` (default `gf.Defaults.seed`) and addressed
by the split (`group`), the epoch, the batch index and what the draw is for. A batch is rebuilt from its address alone,
and the splits never share a stream. Old seeds give new data. Every draw is bit-identical between rebuilds; on CPU,
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
- `ds[k]` and `ds.epoch` take integers only; a float, bool or string was truncated to an integer batch.
- `gf.project_wave` takes no seed and needs every sky angle; nothing is drawn there.
- `Curriculum` has no progress counter: `current_epoch` is gone, and `at(epoch)` gives the distribution of an epoch.
- `Distribution.sample(n, rng)` returns an ndarray.
- Splits are repartitioned. Every entry point that is given no fractions (a dataset, a composed dataset's pools,
  `NoiseObtainer`, an obtainer called directly, `TransientIndex.assign_groups()`, the validator) splits by
  `gf.DEFAULT_SPLIT_FRACTIONS`: 80% train, 10% validate, 10% test. They split 98/1/1 (an obtainer, and the datasets
  reading one directly) or 89/10/1 (`NoiseObtainer`), so one record could be in train by one route and in validate
  by another. A transient record's split is now a hash of its GPS key alone; it was the key and the seed. Records
  have moved between splits: do not evaluate a model trained on an earlier release's split on this release's
  validate or test split.
- An obtainer called for a split serves that split, called directly or through a `NoiseObtainer`; called again for
  another, it rebuilds its segments (it served the first split's, and a fresh real-noise obtainer called directly
  served nothing). A generator it gave out for the first split then raises `RuntimeError`.
- `TransientIndex.assign_groups` and `TransientDataObtainer.build_feature_index` take no seed.
- The model recipes take one root seed: `TrainingRun.train`'s `training_data` is called with the seed alone (it
  took the seed and the first epoch), and `split_seeds`, `TrainingRun.train_seed` and
  `GeorgeHuerta2017.dataset(first_epoch=)` / `dataset_regression(first_epoch=)` are gone; resume sets `ds.epoch`.

## 7. Test Gravyflow (optional)

GravyFlow includes PyTest for testing its functionality. To run tests:

```bash
pytest gravyflow
```

Note: Tests may fail due to unavailable GPU memory if GPUs are currently under heavy use.
