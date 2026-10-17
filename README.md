# GravyFlow

JAX and Keras 3 tools for gravitational-wave data science: simulated and real detector noise, waveform injections,
whitening and SNR, datasets for training, matched filtering, and example models with their validation.

# Installation Guide

## 1. Install GravyFlow

GravyFlow needs Python 3.13 or later. It is not yet on PyPI: install it from the repository, into a new environment
(a virtual environment, or conda):

```bash
python3.13 -m venv gravyflow-env
source gravyflow-env/bin/activate
pip install "gravyflow[realdata,viz] @ git+https://github.com/mrknorman/gravyflow"
```

The core install (`gravyflow` alone) simulates noise and injections, builds datasets and trains models, offline. The
extras add the rest:

| Extra | Adds | For |
|---|---|---|
| `realdata` | gwpy, gwdatafind, gwosc, requests, pandas | real noise, glitches and events (GWOSC, Gravity Spy, GWDataFind) |
| `viz` | bokeh, panel, holoviews, datashader | plots, the validator and its dashboard, and the example models' recipe |
| `notebooks` | `realdata`, `viz`, ipykernel, matplotlib | the notebooks and tutorials in `example_notebooks/` |
| `test` | `realdata`, `viz`, pytest, pytest-cov, lalsuite, pycbc | the test suite |
| `cuda` | JAX's CUDA 12 plugin and NVIDIA's libraries (Linux x86_64) | NVIDIA GPUs |

To work on GravyFlow itself, install a clone in editable mode with the test extra:

```bash
git clone https://github.com/mrknorman/gravyflow
cd gravyflow
pip install -e ".[test]"
```

The package ships the bundled noise curves, template banks and the smaller trained models. The two deep models of
George and Huerta (2017), 66 MB each, are attached to each release instead: fetch one once with
`gravyflow.src.resources.download_weights(name)`, which checks it against its recorded SHA-256 and keeps it in the
cache (below); `pretrained()` finds it there.

## 2. Choose Keras's JAX backend

GravyFlow runs on Keras 3's JAX backend. `import gravyflow` changes nothing in your process (no environment
variable, warnings filter, log level or library setting), so choose the backend yourself, before Keras is imported:

```bash
export KERAS_BACKEND=jax
```

or, in Python, before importing Keras or GravyFlow:

```python
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
import gravyflow as gf
```

Using a GravyFlow name without it raises an `ImportError` that says so. JAX chooses its own platform (a GPU when the
`cuda` extra finds one). Set `JAX_PLATFORMS=cpu` to force the CPU, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` when
several processes share a GPU: GravyFlow no longer sets either.

`import gravyflow` is instant: each name imports its module when it is first used. The public names are listed in
[`docs/api.md`](docs/api.md); everything under `gravyflow.src` is internal.

## 3. Quick start: a synthetic dataset

Two detectors' coloured noise with binary black hole injections (IMRPhenomD, masses 5–95 M☉, optimal SNR 8–15, in
half the examples), whitened. It needs no extra and no network:

```python
import os
os.environ.setdefault("KERAS_BACKEND", "jax")  # gravyflow runs on Keras's JAX backend

import gravyflow as gf

uniform = lambda low, high: gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)

dataset = gf.Dataset(
    seed=1,
    group="train",
    noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.L1, gf.IFO.H1]),
    waveform_generators=[gf.CBCGenerator(
        mass_1_msun=uniform(5.0, 95.0),
        mass_2_msun=uniform(5.0, 95.0),
        scaling_method=gf.ScalingMethod(uniform(8.0, 15.0), gf.ScalingTypes.SNR),
        injection_chance=0.5,
    )],
    input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
    output_variables=[gf.ReturnVariables.INJECTION_PRESENT],
    num_examples_per_batch=32,
    steps_per_epoch=100,
)
inputs, outputs = dataset[0]
print(inputs["WHITENED_ONSOURCE"].shape)  # (32, 2, 2048): examples, detectors, samples (1 s at 2048 Hz)
```

A dataset is a Keras `PyDataset`: pass it to `model.fit`. Each batch is a pure function of its seed and index.

## 4. Quick start: a CNN on a laptop CPU

One command trains the CNN of Gabbard *et al.* (2017) on simulated binary black hole signals (IMRPhenomD, masses
5–95 M☉, optimal SNR 8–15) in whitened coloured Gaussian noise, then validates it. It runs offline, on a CPU, with
XLA's CPU kernels single-threaded, and needs the `viz` extra:

```bash
export KERAS_BACKEND=jax
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
python -m gravyflow.src.model.examples.train_gabbard_2017 --output-dir ~/gravyflow_runs/gabbard_2017_cpu --preset cpu \
  && python -m gravyflow.src.model.examples.validate_gabbard_2017 --run-dir ~/gravyflow_runs/gabbard_2017_cpu
```

- **Training** (about 3 minutes, 198 s measured, on an Apple M-series CPU): the paper's network at 2048 Hz (41,842 weights), 10 epochs of
  300 batches of 32 one-second windows, Nadam at 0.002; the best epoch by validation loss is saved as `model.keras`.
  The run directory keeps a checkpoint per epoch: the same command resumes an interrupted run, and refuses a
  directory holding a run of another configuration.
- **Validation** (about 4 minutes, 246 s measured): 100,000 noise windows (100,000 s of background) and 100,000 injections at SNR
  0–15, from the validator's own seeds. It prints the score threshold for each false-alarm rate (in Hz, with its
  achieved rate), the efficiency by SNR bin at that threshold, and the ROC AUC, and writes the dashboard
  `validation/validation.html` in the run directory. With the default `NoiseProfile.O3_ACTUAL` noise, expect an AUC near 0.978 and, at a false-alarm rate
  of 1e-3 Hz, an efficiency near 0.75 at SNR 7–8 and 0.97 at SNR 9–10.
- **Reproducible from the seed** (`--seed`): the same seed gives the same batches, in the same order, from the same
  initial weights, and with the single-threaded kernels above the same weights, epoch by epoch. XLA's multi-threaded
  CPU kernels round the whitened strain differently in its last bits from call to call (see section 7); training
  amplifies the difference, so a run without the flag is not reproducible, sometimes fails to train at all, and warns.

Notebooks 07 and 08 in `example_notebooks/` walk through the same run.

## 5. Setup permissions

Public data (the default, section 5) needs no credentials. Reading local frames (`gf.LocalFrames`, section 5) needs
the credentials of the cluster that holds them; follow these guides:

https://computing.docs.ligo.org/guide/auth/scitokens/
https://computing.docs.ligo.org/guide/auth/kerberos/

## 6. Public data sources

Real noise and glitches use public data only by default; no credentials are needed:

- Strain comes from the obtainer's data source, named explicitly (`source=`, default `gf.Defaults.source`):
  `gf.GWOSC()`, each run's bulk GWOSC strain release, by default, or `gf.LocalFrames(frametypes=..., channels=...)`
  for frames found by gwdatafind. A source never falls back to another, and never switches the frame type it names:
  a fetch that fails raises `gf.StrainUnavailable`, naming the source and the cause. On a LIGO cluster, pass
  `source=gf.LocalFrames(...)` to read its frames; without it, strain is the public GWOSC release.
- Segments and observing-run bounds come from the GWOSC timeline and run API (https://gwosc.org). The default selection is `DATA` with the `BURST_CAT1` and `BURST_CAT2` vetoes applied, without CBC, burst, stochastic or detchar hardware injections (continuous-wave injections, which run throughout, are kept). Real noise also excludes every GW event GWOSC publishes (its whole signal in band; see `docs/acquisition.md`) and every Gravity Spy glitch (its whole extent).
- Glitches come from the Gravity Spy machine-learning classifications: Glanzer et al. 2023, *Class. Quantum Grav.* 40 065004, released on Zenodo (doi:10.5281/zenodo.5649212, v1.0.0, CC-BY-4.0). Please cite both if you use them. Each file is downloaded on first use and cached in `~/.gravyflow/cache` (override with `GRAVYFLOW_CACHE_DIR`).

Every cache gravyflow keeps lives under that one root, `GRAVYFLOW_CACHE_DIR` if it is set, else `~/.gravyflow/cache`: GWOSC timelines and events, Gravity Spy files, and the strain store (`store/`), which keeps the source's own files of real strain, raw, each fetched once, named by the source they come from so two sources never share one. Noise, glitches, events and their partners are all read from it; nothing else keeps strain, and gwpy's download cache is off. Strain caches of earlier versions (real-noise segments in `segments/`, `./generator_data` or beside the package; glitch, event and partner caches, `*_cache_v*.h5`; GWOSC downloads in `astropy/`) are never read; gravyflow names each once, with its size, and never moves or deletes them.

The data source changes these, with no transition period:

- Strain no longer comes from local frames when gwdatafind finds them, with GWOSC as the fallback: it comes from the
  source alone. On a host with local frames, the default is now public GWOSC strain; pass `gf.LocalFrames(...)` for
  the frames.
- `ObservingRunData` takes no channels or frame types (`ObservingRunData(name, start, end, state_flags, gwosc_runs)`),
  and obtainers have no `frame_types` or `channels`: `gf.LocalFrames` names them, per GWOSC run.
- The source is in every obtainer's fingerprint, so a checkpoint or validation written before is refused as stale,
  and caches written before (real-noise segments, and glitch, event and partner caches) are not read again.
- Glitches, events and partners are read from the strain store, as noise is: whole source files, fetched once and
  shared by every rate and configuration. The glitch, event and partner caches, `TransientCache`, the
  `force_acquisition` argument and `scripts/check_glitch_cache.py` are gone. A transient whose strain the source
  does not deliver is refused by name (`TransientUnavailable`, a `gf.StrainUnavailable`, naming the `exclusions`
  entry that plans around it), never skipped and counted as `"no data"`. `START_GPS_TIME` of a transient window is
  exact, its centre the sample nearest the transient on the absolute grid; a request may read up to 48.2 s each
  side of a transient, the resampler's margin counted. A saved `TransientIndex` keeps each record's SNR, peak
  frequency, duration, confidence and event parameters; an index saved before is refused, to be built again.

## 7. Randomness: one seed tree

Every random draw comes from one seed tree, rooted at a dataset's `seed` (default `gf.Defaults.seed`) and addressed
by the split (`group`), the epoch, the batch index and what the draw is for. A batch is rebuilt from its address alone,
and the splits never share a stream. Within a split, datasets of one seed draw the same streams (common random
numbers), so a noise-only dataset and a signal dataset share their noise, as an ablation wants. Datasets that must be
independent, such as the classes of a classifier, take distinct names (`gf.Dataset(..., name="noise")`), each a
scope of the tree, or are the pools of one `gf.ComposedDataset`. Old seeds give new data. Every draw is bit-identical between rebuilds; on CPU,
XLA's multi-threaded kernels can still round whitened and projected strain and Welch spectra differently from one call
to the next, in the last bits of float32. For bit-identical batches, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`
before starting Python: importing gravyflow starts JAX, which reads the flag then (the results differ, also in the last
bits, from multi-threaded ones). The test suite sets it itself, and a recipe training run warns without it. The remaining replaced seed arguments
(`generate(seed=)`, `InjectionGenerator(seed=)`, `PoolSampler(seed=)` and the like) still work for one release, with
a `DeprecationWarning`. These changes have no such period:

- `ds[k]` is batch k, and `next(iter(ds))` is always batch 0. Step through batches with `next(ds)`, or index them.
- Under `model.fit`, a training split takes Keras' epoch: pass `callbacks=[ds.epoch_callback()]` and
  `shuffle=False`. Without the callback, Keras epoch e would draw the batches of epoch e + 1, and a warning says so.
  With it, `fit(steps_per_epoch=)` must be unset or the dataset's length (it is refused otherwise), a second `fit`
  continues only with `initial_epoch` (without it, it starts again at epoch 0 and serves the first's batches), and a
  batch asked for out of order is refused: `fit`'s default shuffle reorders batches with Python's unseeded global
  random, which no seed repeats. Every batch is already a fresh seeded draw.
- Real noise drawn at random (`NoiseType.REAL`, `SamplingMode.RANDOM`) is addressed by index too: batch k of epoch
  e is the same bytes however and in whatever order it is asked for, so validate and test serve the same batches
  every epoch, and each training epoch its own. Its batches are a plan over the split's segments, from the
  availability frozen when the dataset is built (part of every fingerprint): each pass visits the segments in a new
  order, each for its share of batches, and each batch draws its examples from eight batches' segments of the pass,
  so no segment feeds a run of batches alone (one fed hundreds). Every batch has exactly its size: an example whose
  detectors' windows cannot be separated is refused by name (`SeparationRefused`), never dropped. The batches after
  the one asked for are read ahead on background threads (`prefetch_depth=2`, `prefetch_workers=2` on the obtainer;
  they never change a batch). `ds.prepare(epochs=E)` (or `batches=N`) fetches everything those batches read before a
  long run, and lists every contradiction between the timeline and the source at once (`gf.Unprepared`, with the
  `exclusions` that plan around them); a refusal from a batch reaches the caller with its own type, not a bare
  `Exception`. The old `ds.prepare(onsource, offsource)`, the model inputs of raw windows, is `ds.model_inputs`.
- `gf.Model` gives its training dataset `steps_per_epoch = num_examples_per_epoc // num_examples_per_batch` from its
  training config, and refuses `dataset_args` with another `steps_per_epoch`; `Model.train` passes `fit` no
  `steps_per_epoch` for a dataset with a length. Its Keras model's output is named (a dict, as the dataset's targets
  are), and it is saved to `model_path / "model.keras"` (`gf.model_file`): Keras 3 saves whole models only as
  `.keras` files.
- `ds[k]` and `ds.epoch` take integers only; a float, bool or string was truncated to an integer batch.
- `gf.project_wave` takes no seed and needs every sky angle; nothing is drawn there.
- `Curriculum` has no progress counter: `current_epoch` is gone, and `at(epoch)` gives the distribution of an epoch.
- A `gf.ComposedDataset` builds its batches as `gf.Dataset` does, so they meet the batch contract: example b is row b
  of its pool's batch (it was the pool's next unused row, so a seed's batches have changed); `POOL_LABEL` and the
  transient labels are float32, the transient labels per detector, with noise's (`DATA_LABEL` NOISE, -1 for the
  types) for a pool of synthetic noise, which took the pool's label; `START_GPS_TIME` is NaN for synthetic noise; and
  masks, injections and every label have a row per example, for every pool's generators. Given no `ifos`, it takes
  the detectors its pools' sources share (it took L1).
- An `IncoherentGenerator` labels each parameter with its components' common value, NaN where they differ (it gave
  component 0's), and has one mask value per example; with it, `InjectionGenerator.generate` gives (G, B, I, 2, S)
  injections, a coherent generator's repeated in each detector.
- `Distribution` and `Curriculum` are frozen specs, holding no random state: `Distribution.sample(n, rng)` needs a
  NumPy Generator and returns an ndarray (an integer UNIFORM includes its max); `Curriculum.sample(n, rng, epoch)`
  draws from `at(epoch)`. Their `seed=`, `rng`, `reseed`, rng-less `sample`, and the Curriculum's `steps_per_epoch=`,
  `configure`, `reset` and current `min_`/`max_` are gone. Change a field with `dataclasses.replace`.
- A `Network` is stateless geometry: `Network(seed=)` and `Network.rng` are gone, and random geometry needs `rng=`.
  A generator given a network config path draws it from its own seed-tree stream, so one config gives one network.
  Waveform generators have no `reseed`.
- `gf.get_glitch_times_with_labels(balanced=True)` oversamples from a seed-tree stream (`streams=`), not NumPy's
  global RNG.
- Splits are repartitioned by one `gf.SplitSpec` (`docs/acquisition.md`, section 2): GPS time is cut into absolute
  8192 s blocks from GPS 0, each hashed to test (10%), validate (10%) or train (80%), with 128 s between splits. Every
  route (a dataset, a composed dataset's pools, `NoiseObtainer`, an obtainer called directly, the validator) reads
  the data obtainer's `split_spec`, so a GPS time is in one split whatever reads it. A transient record takes its
  footprint's split; one that crosses an edge between splits is in no split. Noise and records have moved between
  splits: do not evaluate a model trained on an earlier release's split on this release's validate or test split.
  A model's split is its data's: `gf.Model.train` and the recipes' `TrainingRun.train` read it from their
  training and validation data (a gravyflow dataset names its `split_spec` and `group`), refuse data that names
  none or is split by two specs (synthetic noise, WHITE or COLORED, reads no real data: a gravyflow dataset of it
  alone, or the recipe's batches of one, has `split_spec` None, is split-neutral, and combines with real data of
  any spec; a model of synthetic data alone records the default spec; anything else with `split_spec` None names
  no split), and record the fingerprint and the split that selected the model in `run.json`
  beside the model. The record belongs to the weights: every model gravyflow trains or loads carries it
  (`TrainingRun.train`'s, `RunDirectory.load_model`'s, a `gf.Model`'s), weights that carry one never train on
  another split, whatever `force_retrain` says, and the validator reads it from the model, refusing a model split
  by another spec and the split that selected it. The validator's split is that of every stage that reads real
  data: its noise's, and with `score_real_events=True` the real events', read under the noise's split, or beside
  synthetic noise under the default spec, so a model of another spec is refused before any data is read.
- Every entry point needs a named split: `group='train'`, `'validate'` or `'test'`. `'all'` and any other name raise
  `KeyError`. `gf.DEFAULT_SPLIT_FRACTIONS`, `NoiseObtainer(groups=)`, `get_valid_segments(groups=)` and
  `TransientIndex.assign_groups` are gone: give the data obtainer `split_spec=gf.SplitSpec(fractions=...)`, or another
  `version` for an independent split (cross-validation).
- Real events are evaluation data: every confident event is in `test`, named events too, and an obtainer of events
  alone refuses `'train'` and `'validate'` unless it is built with `train_on_real_events=True`, which needs
  `coincidence=gf.Coincidence.COINCIDENT`. Data labels are read by their kind, never their number: give
  `gf.DataLabel`, `gf.GlitchType` or `gf.EventConfidence` members (a bare number is refused, since 2 is
  `DataLabel.EVENTS` and `GlitchType.CHIRP` alike), and `GlitchType.CHIRP` asks for chirp glitches, never
  events. A noise obtainer refuses `DataLabel.EVENTS`: noise vetoes every
  event. An event is read at its time on the detectors whose data holds it, the others at partner times in the
  nearest test block of its observing run, so every read of an event is test time.
  A `validate` or `test` pass over transients serves each once, then ends.
- Detectors are read at one time only under `coincidence=gf.Coincidence.COINCIDENT`, or for an event seen
  in several; every real-noise batch says which examples are (`ReturnVariables.COINCIDENT`), and a dataset
  returns it when asked (0 for simulated noise).
- The event list's cache is format 4 (`event_catalogue_v4.json`, fetched again once): it adds the 90% bounds of
  the component masses, and each event carries `mass_ambiguous`, True where either component's interval crosses
  the 3 solar-mass type threshold.
- An obtainer called for a split serves that split, called directly or through a `NoiseObtainer`; called again for
  another, it rebuilds its segments (it served the first split's, and a fresh real-noise obtainer called directly
  served nothing). A generator it gave out for the first split then raises `RuntimeError`.
- `TransientIndex.assign_splits` and `TransientDataObtainer.build_feature_index` take no seed.
- A glitch's example reads each requested detector the glitch was not seen in at a partner time of its own, in the
  record's split and block, never at the glitch's time, and labels that detector noise (`docs/acquisition.md`,
  Transient detectors and partners); it read every detector over the glitch's window. A record with no partner time
  is not served, and is counted under `"no partner"`.
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
- A dataset's batches can be written once and read back: `ds.materialise(root, batches=N)` (or `k_range=`) writes
  them under `root` as a set named by its manifest's sha256, which records `ds.fingerprint` (the sha256 of the
  dataset's canonical spec, seed root, split spec, seed-tree and data versions, and the gravyflow, numpy, jax and
  scipy versions), the platform (`XLA_FLAGS` and the backend, recorded, never hashed), the batch contract, the split,
  the epoch and the range. Each batch is an uncompressed `.npz` named by its sha256; files are created once and never
  replaced, so an interrupted run resumes and writers that disagree are refused (`MaterialiseConflict`).
  `ds.load_materialised(set)` serves `ds[k]` from the set wherever it holds the address, verifying every read, and
  `seed_metadata["materialised"]` names it; another spec, split or contract, a missing shard, an edited or foreign
  file is refused by name. The recipe's `materialise(batches, n, root)` is this: a fixed validation set read from
  disk, so a resumed run reads it rather than drawing it again.
- `gf.Model`'s binary output is a logit, not a sigmoid probability: it trains with
  `BinaryCrossentropy(from_logits=True)` and reports `BinaryAccuracy` at a threshold of 0, and the validator (which
  refused a probability output) scores it. `Model.validate(validate_args, config, model_path)` takes a
  `ValidationConfig` and returns the `Validator`. A checkpoint that exists but cannot be loaded raises `ValueError`;
  `Model.load` returned an untrained model in its place. `Model.train(force_retrain=False)` resumes the run in the
  model's directory: a checkpoint there with no history is refused by name, never discarded (only
  `force_retrain=True` starts a new run over it). The model search scores a generation from its validators' saved
  results and draws from `search/` streams of the seed tree, so its draws (and a seed's genomes) have changed.

## 8. Units: physical strain

Every array gravyflow reads, returns or caches is physical strain: noise of every type but white (which has no
detector, and unit variance), injections, `ONSOURCE`, `OFFSOURCE` and `INJECTIONS`, the HRSS and HPEAK targets and
labels, and the strain store. Whitened outputs and SNRs are dimensionless, as before. The square of strain
(about 1e-42) is below float32's normal range, so each step that squares data (PSD estimation, whitening, the SNR,
the matched filter, HRSS and HPEAK, the rolling Pearson correlation) divides by the data's peak first and restores
it after; whitened data, correlations and SNRs are identical, to float32 rounding, at any input amplitude. A PSD
array given to `gf.matched_filter_fft` or `gf.optimal_snr` is normalised by its median in float64, so give it as
float64: a float32 PSD of strain has already underflowed, and is refused.

Where the result itself is a square of strain, float32 cannot hold it:

- `gf.psd()`, the one public PSD, returns a `PowerSpectrum`, which keeps the normalised PSD and its amplitude scale
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
- A waveform's reference time (t = 0: a burst's peak, a CBC's merger, the peak of its envelope) is always on a
  sample, sample N // 2 of its buffer, as in LAL's burst waveforms. A buffer of odd length put it half a sample off
  the grid, which lowered a sampled peak by cos(pi f / fs) (0.953 at 200 Hz and 2048 Hz).

## 9. Test Gravyflow (optional)

The suite needs the `test` extra (`pip install -e ".[test]"` in a clone) and runs offline, in temporary roots: it
never reads or writes your cache. From the clone:

```bash
pytest gravyflow
```

Tests that need GWOSC or Zenodo are marked `network` and run only with `--network`. Tests may fail for lack of GPU
memory when GPUs are under heavy use.
