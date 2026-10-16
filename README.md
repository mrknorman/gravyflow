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
and the splits never share a stream. Old seeds give new data. The replaced seed arguments (`Distribution(seed=)`,
`generate(seed=)`, `reseed`, `Network(seed=)`, `Curriculum(seed=)` and the like) still work for one release, with a
`DeprecationWarning`. These changes have no such period:

- `ds[k]` is batch k, and `next(iter(ds))` is always batch 0. Step through batches with `next(ds)`, or index them.
- Under `model.fit`, a training split takes Keras' epoch: pass `callbacks=[ds.epoch_callback()]`. Without it, Keras
  epoch e would draw the batches of epoch e + 1, and a warning says so.
- `gf.project_wave` takes no seed and needs every sky angle; nothing is drawn there.
- `Curriculum` has no progress counter: `current_epoch` is gone, and `at(epoch)` gives the distribution of an epoch.
- `Distribution.sample(n, rng)` returns an ndarray.

## 7. Test Gravyflow (optional)

GravyFlow includes PyTest for testing its functionality. To run tests:

```bash
pytest gravyflow
```

Note: Tests may fail due to unavailable GPU memory if GPUs are currently under heavy use.
