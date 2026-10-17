"""
Synthetic streams: from one root, the train, validate and test splits, the
validator's stages and the pools of a composed dataset draw disjoint strain and
independent labels, whatever order they are built and drawn in; under Keras'
fit, each epoch serves its own addresses; and datasets share their noise only
as common random numbers allow, by name.

Synthetic noise has no GPS time, so every check here reads the content
(`fingerprints`): 64-sample windows for strain, rank correlation at every
offset and exact values for labels, and a recursive state digest for caller objects.

Augmentation reaches only the training split: those checks read the fake
world's coded strain (`provenance`), whose every row says whether it was
sign-flipped or time-reversed, and where it was read.
"""

import hashlib
import itertools
import warnings

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.tests.leakage import fixtures as fx
from gravyflow.tests.leakage.fingerprints import (
    assert_rows_disjoint, assert_streams_independent, assert_unchanged, concatenate_streams,
    label_streams, state_items, strain_prints,
)
from gravyflow.tests.leakage.provenance import DecodeRefused, decode_row, footprints
from gravyflow.tests.leakage.tap import DatasetTap, EpochWindows

RV, WP, ST = gf.ReturnVariables, gf.WaveformParameters, gf.ScalingTypes
UNIFORM = gf.DistributionType.UNIFORM

FS = 256.0
ONSOURCE_S, OFFSOURCE_S, CROP_S = 1.0, 4.0, 0.5
IFOS = [gf.IFO.H1, gf.IFO.L1]
SPLITS = ("train", "validate", "test")
ROOTS = (None, 0, 42)
NOISE = (gf.NoiseType.WHITE, gf.NoiseType.COLORED)

STRAIN = [RV.ONSOURCE, RV.OFFSOURCE]
LABELS = [ST.SNR, ST.HRSS, ST.HPEAK, RV.CENTRAL_TIME, WP.RIGHT_ASCENSION_RADIANS,
          WP.DECLINATION_RADIANS, WP.POLARIZATION_RADIANS]
# Label streams need at least 64 examples per set.
LABEL_BATCH, LABEL_BATCHES = 32, 2


def uniform(lo, hi):
    return gf.Distribution(min_=lo, max_=hi, type_=UNIFORM)


def burst(snr=None, name=None):
    """A white-noise burst, injected into every example, at a random SNR."""
    kwargs = {} if name is None else {"name": name}
    return gf.WNBGenerator(
        scaling_method=gf.ScalingMethod(value=snr or uniform(8.0, 20.0), type_=ST.SNR),
        injection_chance=1.0,
        duration_seconds=uniform(0.1, 0.3),
        min_frequency_hertz=uniform(20.0, 40.0),
        max_frequency_hertz=uniform(60.0, 110.0),
        **kwargs,
    )


def settings(batch=8, **overrides):
    values = dict(
        sample_rate_hertz=FS, onsource_duration_seconds=ONSOURCE_S,
        offsource_duration_seconds=OFFSOURCE_S, crop_duration_seconds=CROP_S,
        num_examples_per_batch=batch, ifos=IFOS, steps_per_epoch=16,
    )
    values.update(overrides)
    return values


def dataset(noise_type, root, split, generators=None, outputs=(), batch=8, obtainer=None,
            inputs=STRAIN, **kwargs):
    return gf.Dataset(
        seed=root, group=split,
        noise_obtainer=obtainer or gf.NoiseObtainer(noise_type=noise_type, ifos=IFOS),
        waveform_generators=[burst()] if generators is None else generators,
        input_variables=list(inputs), output_variables=list(outputs),
        **settings(batch, **kwargs),
    )


def composed(noise_type, root, split, pools=None, outputs=(RV.POOL_LABEL,), batch=8):
    pools = pools or [
        gf.FeaturePool(name="noise", label=0, probability=0.5,
                       noise_obtainer=gf.NoiseObtainer(noise_type=noise_type, ifos=IFOS)),
        gf.FeaturePool(name="bursts", label=1, probability=0.5,
                       noise_obtainer=gf.NoiseObtainer(noise_type=noise_type, ifos=IFOS),
                       injection_generators=[burst()]),
    ]
    return gf.ComposedDataset(
        pools=pools, seed=root, group=split, input_variables=STRAIN,
        output_variables=list(outputs), **settings(batch),
    )


def batches(ds, count, start=0):
    return [ds[k] for k in range(start, start + count)]


def strain_of(drawn, keys=STRAIN):
    """Strain fingerprints of a list of (x, y) batches."""
    arrays = {key.name: np.concatenate([np.asarray(x[key.name], np.float32) for x, _ in drawn])
              for key in keys}
    return strain_prints(arrays)


def labels_of(drawn, keys=LABELS):
    """Per-example label streams of a list of (x, y) batches."""
    per_batch = []
    for _, y in drawn:
        n = np.shape(y[keys[0].name])[0]
        per_batch.append(label_streams({key.name: y[key.name] for key in keys}, n))
    return concatenate_streams(per_batch)


def batch_digest(batch):
    """sha256 of every array of one (x, y) batch, by key."""
    x, y = batch
    h = hashlib.sha256()
    for part in (x, y):
        for key in sorted(part):
            value = np.ascontiguousarray(np.asarray(part[key]))
            h.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------- strain rows

@pytest.mark.parametrize("build", ["dataset", "composed"])
@pytest.mark.parametrize("root", ROOTS, ids=lambda r: f"root={r}")
@pytest.mark.parametrize("noise_type", NOISE, ids=lambda n: n.name)
def test_split_rows_disjoint(noise_type, root, build):
    """From one root, no strain window of one split appears in another."""
    if build == "dataset":
        keys = [*STRAIN, RV.INJECTIONS]
        prints = {split: strain_of(batches(dataset(noise_type, root, split, inputs=keys), 2), keys)
                  for split in SPLITS}
    else:
        prints = {split: strain_of(batches(composed(noise_type, root, split), 2)) for split in SPLITS}
    assert_rows_disjoint(prints)


# ------------------------------------------------------------- label streams

@pytest.mark.parametrize("root", ROOTS, ids=lambda r: f"root={r}")
@pytest.mark.parametrize("noise_type", NOISE, ids=lambda n: n.name)
def test_label_streams_independent(noise_type, root, monkeypatch):
    """
    From one root, no label stream of one split, nor of the validator's
    injection stage, is shared with another's, in value, in rank or at any
    offset; and two generators in one dataset draw independent skies.
    """
    streams = {
        split: labels_of(batches(dataset(noise_type, root, split, outputs=LABELS, batch=LABEL_BATCH),
                                 LABEL_BATCHES))
        for split in SPLITS
    }
    streams["validator"] = validator_injection_streams(noise_type, root, monkeypatch)
    assert_streams_independent(streams)

    two = dataset(noise_type, root, "train", generators=[burst(), burst()],
                  outputs=[WP.RIGHT_ASCENSION_RADIANS], batch=LABEL_BATCH)
    ra = labels_of(batches(two, LABEL_BATCHES), [WP.RIGHT_ASCENSION_RADIANS])
    assert_streams_independent({"generator 0": {"RA": ra["RIGHT_ASCENSION_RADIANS[0]"]},
                                "generator 1": {"RA": ra["RIGHT_ASCENSION_RADIANS[1]"]}})


def logit_model(rate=FS):
    """A fixed linear model returning logits, at ``rate``: the validator only needs scores."""
    import keras
    return keras.Sequential([
        keras.Input(shape=(len(IFOS), int(rate * ONSOURCE_S)), name=RV.WHITENED_ONSOURCE.name),
        keras.layers.Flatten(),
        keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros"),
    ])


def validator_injection_streams(noise_type, root, monkeypatch):
    """
    The label streams the validator's injection stage draws, given training's
    dataset arguments and root, every detector's central time among them:
    tapped from its dataset, under the splits' own names.
    """
    from gravyflow.src.validate import ValidationConfig
    from gravyflow.src.validate.bank import ValidationBank

    args = dict(
        seed=root, noise_obtainer=gf.NoiseObtainer(noise_type=noise_type, ifos=IFOS),
        waveform_generators=[burst()], input_variables=[RV.WHITENED_ONSOURCE],
        **settings(LABEL_BATCH),
    )
    config = ValidationConfig(seed=gf.Defaults.seed if root is None else root,
                              num_examples=LABEL_BATCH * LABEL_BATCHES, batch_size=LABEL_BATCH)
    tap = DatasetTap(monkeypatch, LABELS)
    ValidationBank(logit_model(), args, config).generate()
    (stage,) = tap.take()
    assert stage.group == "test" and len(stage.batches) == LABEL_BATCHES
    return concatenate_streams(label_streams({key.name: batch[key.name] for key in LABELS}, LABEL_BATCH)
                               for batch in stage.batches)


# ----------------------------------------------------------------- ordering

def shared_objects(noise_type):
    """Caller objects shared by every split: an obtainer and a generator with a curriculum."""
    curriculum = gf.Curriculum(start=uniform(40.0, 50.0), end=uniform(8.0, 12.0), num_epochs=4)
    return {"obtainer": gf.NoiseObtainer(noise_type=noise_type, ifos=IFOS),
            "generator": burst(snr=curriculum)}


def split_dataset(split, objects, noise_type, root=42):
    return dataset(noise_type, root, split, generators=[objects["generator"]], obtainer=objects["obtainer"],
                   outputs=[ST.SNR], steps_per_epoch=2)


def two_epochs(ds):
    """Digests of batches 0 and 1 of epochs 0 and 1."""
    digests = []
    for _ in range(2):
        digests += [batch_digest(ds[k]) for k in range(2)]
        ds.on_epoch_end()
    return digests


@pytest.mark.parametrize("noise_type", NOISE, ids=lambda n: n.name)
def test_order_independent(noise_type):
    """
    Each split's batches, over two epochs, are the same whatever the order the
    splits are built and drawn in, alone or interleaved, and using the shared
    caller objects changes none of them.
    """
    solo = {split: two_epochs(split_dataset(split, shared_objects(noise_type), noise_type))
            for split in SPLITS}
    mismatches = []
    for order in itertools.permutations(SPLITS):
        objects = shared_objects(noise_type)
        before = {name: state_items(obj, name) for name, obj in objects.items()}
        built = {split: split_dataset(split, objects, noise_type) for split in order}
        for split in order:
            if two_epochs(built[split]) != solo[split]:
                mismatches.append(f"order {order}: {split}")
        assert_unchanged(before, objects)

    # Interleaved: one batch of each split in turn; the evaluation splits see
    # an epoch end after every batch, which must not move them.
    objects = shared_objects(noise_type)
    built = {split: split_dataset(split, objects, noise_type) for split in SPLITS}
    interleaved = {split: [] for split in SPLITS}
    for _ in range(2):
        for k in range(2):
            for split in SPLITS:
                interleaved[split].append(batch_digest(built[split][k]))
                if split != "train":
                    built[split].on_epoch_end()
        built["train"].on_epoch_end()
    for split in SPLITS:
        if interleaved[split] != solo[split]:
            mismatches.append(f"interleaved: {split}")
    assert not mismatches, f"Batches that depend on build or draw order: {mismatches}"


def test_evaluation_targets_are_fixed_at_the_curriculum_end():
    """Validate's targets are the same at epochs 0 and 3, drawn from the curriculum's end distribution."""
    objects = shared_objects(gf.NoiseType.WHITE)
    validate = split_dataset("validate", objects, gf.NoiseType.WHITE)
    first = np.asarray(validate[0][1][ST.SNR.name])
    for _ in range(3):
        validate.on_epoch_end()
    later = np.asarray(validate[0][1][ST.SNR.name])
    np.testing.assert_array_equal(first, later)
    assert np.all((first >= 8.0) & (first <= 12.0)), f"targets outside the end distribution: {first}"


# ------------------------------------------------------------ Keras' epochs

FIT_STEPS, FIT_VALIDATION_STEPS, FIT_EPOCHS = 3, 2, 3


def one_layer_model(target):
    """A compiled one-layer model from the onsource window to ``target``."""
    import keras

    inputs = keras.Input(shape=(len(IFOS), int(FS * ONSOURCE_S)), name=RV.ONSOURCE.name)
    outputs = keras.layers.Dense(1, name=target.name)(keras.layers.Flatten()(inputs))
    model = keras.Model(inputs, {target.name: outputs})
    model.compile(optimizer="sgd", loss={target.name: "mse"})
    return model


def fit_serves(monkeypatch, shuffle, epochs=FIT_EPOCHS, initial_epoch=0):
    """
    Fit a one-layer model to a training split, validating on a validate
    split, and return what fit asked of them: the serves before its first
    epoch began, and each epoch's serves, as (split, epoch, k).
    """
    import keras

    tap = DatasetTap(monkeypatch, ())
    target = RV.INJECTION_PRESENT
    train, validate = (
        dataset(gf.NoiseType.WHITE, 42, split, generators=[burst()], outputs=[target],
                inputs=[RV.ONSOURCE], steps_per_epoch=steps)
        for split, steps in (("train", FIT_STEPS), ("validate", FIT_VALIDATION_STEPS)))
    model = one_layer_model(target)
    windows = EpochWindows(tap)
    model.fit(train, validation_data=validate, epochs=epochs, initial_epoch=initial_epoch, shuffle=shuffle,
              callbacks=[train.epoch_callback(), windows.callback()], verbose=0)
    return windows.take()


@pytest.mark.global_rng
def test_keras_fit_addresses(monkeypatch):
    """
    Under ``model.fit``, Keras epoch e serves the training addresses
    (train, e, k), each k once, and the validation addresses (validate, 0, k),
    each k once, in every epoch; a run resumed with ``initial_epoch`` serves
    the same addresses in the epochs it runs.

    Before its first epoch, Keras asks for training batches to learn their
    structure; they are not among any epoch's steps. They come from the
    dataset's epoch before Keras sets it, 0, even in a resumed run.
    """
    for initial_epoch in (0, 1):
        before, epochs = fit_serves(monkeypatch, False, initial_epoch=initial_epoch)
        run = f"fit from epoch {initial_epoch}"
        assert sorted(epochs) == list(range(initial_epoch, FIT_EPOCHS)), f"{run}: epochs {sorted(epochs)}"
        for epoch, served in epochs.items():
            assert sorted(a for a in served if a[0] == "train") == [
                ("train", epoch, k) for k in range(FIT_STEPS)], f"{run}, Keras epoch {epoch}: {served}"
            assert sorted(a for a in served if a[0] == "validate") == [
                ("validate", 0, k) for k in range(FIT_VALIDATION_STEPS)], f"{run}, Keras epoch {epoch}: {served}"
        assert set(before) <= {("train", 0, k) for k in range(FIT_STEPS)}, (
            f"{run}, before its first epoch: {before}")


@pytest.mark.global_rng
def test_keras_shuffle_is_refused(monkeypatch):
    """
    ``fit``'s shuffle, its default, reorders a dataset's batches with Python's unseeded
    global random, which no seed repeats: a training dataset given Keras' epoch refuses
    the first batch asked for out of order, by name, before it is served.
    """
    import random

    random.seed(3)  # Keras' shuffle draws from here: seeded, so the refusal is repeatable
    with pytest.raises(ValueError, match=r"Keras asked for batch \d+ where batch \d+ comes next.*shuffle=False"):
        fit_serves(monkeypatch, True)


@pytest.mark.global_rng
def test_after_fit_any_order_is_served():
    """
    The out-of-order refusal holds only while ``fit`` runs: after it, the same
    dataset is served to ``evaluate`` and ``predict``, and indexed by hand in any
    order; a later ``fit`` that shuffles is still refused.
    """
    import random

    target = RV.INJECTION_PRESENT
    train = dataset(gf.NoiseType.WHITE, 42, "train", generators=[burst()], outputs=[target],
                    inputs=[RV.ONSOURCE], steps_per_epoch=FIT_STEPS)
    model = one_layer_model(target)
    callbacks = [train.epoch_callback()]
    model.fit(train, epochs=1, shuffle=False, callbacks=callbacks, verbose=0)
    model.evaluate(train, verbose=0)
    model.predict(train, verbose=0)
    for k in reversed(range(FIT_STEPS)):
        train[k]
    random.seed(3)  # Keras' shuffle draws from here: seeded, so the refusal is repeatable
    with pytest.raises(ValueError, match=r"Keras asked for batch \d+ where batch \d+ comes next"):
        model.fit(train, epochs=FIT_EPOCHS, shuffle=True, callbacks=callbacks, verbose=0)


# -------------------------------------------------------------------- pools

def test_pools_disjoint():
    """
    No strain or label row is shared between the pools of a composed dataset,
    between its splits, or between roots r and r + 1; two pools may share
    caller objects. The legacy pool sampler's deprecation is an error.
    """
    noise = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS)
    shared_burst = burst()

    def pools():
        return [
            gf.FeaturePool(name="noise", label=0, probability=1.0, noise_obtainer=noise),
            gf.FeaturePool(name="bursts-a", label=1, probability=1.0, noise_obtainer=noise,
                           injection_generators=[shared_burst]),
            gf.FeaturePool(name="bursts-b", label=2, probability=1.0,
                           noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS),
                           injection_generators=[burst()]),
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sets = {}
        for root in (42, 43, 44):
            for split in ("train", "test"):
                drawn = batches(composed(gf.NoiseType.WHITE, root, split, pools=pools(), batch=16), 2)
                for label in (0, 1, 2):
                    sets[f"root {root} {split} pool {label}"] = strain_prints(_select(drawn, label))
    assert_rows_disjoint(sets)


def _select(drawn, label):
    """The strain of one pool's examples in composed batches; refuses a pool that served none."""
    keep = np.concatenate([np.asarray(y[RV.POOL_LABEL.name]).reshape(-1) == label for _, y in drawn])
    if np.count_nonzero(keep) < 2:
        raise AssertionError(f"Pool {label} served {np.count_nonzero(keep)} examples; the check would be empty.")
    return {key.name: np.concatenate([np.asarray(x[key.name], np.float32) for x, _ in drawn])[keep]
            for key in STRAIN}


# ------------------------------------------------------ common random numbers

def test_common_random_numbers():
    """
    As ruled: datasets with the same root, split and name share their noise
    (common random numbers across ablations); datasets with different names
    share none. A name is one scope of the seed tree: it cannot hold a slash.
    """
    def noise_only(name):
        return gf.Dataset(seed=42, group="train", name=name,
                          noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS),
                          input_variables=[RV.OFFSOURCE], **settings())

    def with_bursts(name):
        return gf.Dataset(seed=42, group="train", name=name,
                          noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS),
                          waveform_generators=[burst()], input_variables=[RV.OFFSOURCE], **settings())

    same = [np.asarray(ds[0][0][RV.OFFSOURCE.name]) for ds in (noise_only("ablation"), with_bursts("ablation"))]
    np.testing.assert_array_equal(same[0], same[1])

    distinct = {name: strain_prints({"OFFSOURCE": np.asarray(ds[0][0][RV.OFFSOURCE.name], np.float32)})
                for name, ds in (("noise", noise_only("noise")), ("signal", with_bursts("signal")))}
    assert_rows_disjoint(distinct)

    with pytest.raises(ValueError, match="dataset name"):
        noise_only("noise/signal")


# --------------------------------------------------------------- augmentation

#: Sign and time reversal on every example: any row they reach is visibly reversed.
REVERSALS = (gf.SignReversal(probability=1.0), gf.TimeReversal(probability=1.0))
WORLD_IFOS = ("H1", "L1")


def world_settings(batch=8, rate=256.0):
    """The fake world's window and detectors, at its rate (256 Hz unless built at another)."""
    return dict(
        sample_rate_hertz=rate, onsource_duration_seconds=fx.ONSOURCE_SECONDS,
        crop_duration_seconds=fx.CROP_SECONDS, offsource_duration_seconds=fx.OFFSOURCE_SECONDS,
        num_examples_per_batch=batch, ifos=[gf.IFO[name] for name in WORLD_IFOS], steps_per_epoch=16,
    )


def augmented_world():
    """Real noise in every split: one O3 piece across a train|test edge, one across validate|train."""
    fixture, _ = fx.two_splits_per_run(("O3",), origin="O3")
    return fixture


#: The rate real events are read at: the validator's world is served at it.
EVENT_RATE = 4096.0


def validator_world():
    """
    The augmented world at the rate real events are read at, with one confident
    event in test: 100 s past the train|test edge, inside its piece.
    """
    from gravyflow.tests.leakage.fakes import EventRecord

    fixture, _ = fx.two_splits_per_run(("O3",), rate=EVENT_RATE, origin="O3")
    event = EventRecord("GWREAL", fx.boundary("O3", "train", "test", run="O3") + 100.4, 2.0)
    return fixture.with_catalogues(events=[event])


def world_dataset(obtainer, split, outputs=(), batch=8):
    """A dataset of the world's real noise, read through ``obtainer``."""
    return gf.Dataset(noise_obtainer=obtainer, group=split, seed=7, input_variables=STRAIN,
                      output_variables=[RV.START_GPS_TIME, *outputs], **world_settings(batch))


def pool_dataset(obtainer, split):
    """A composed dataset whose one pool reads the world's real noise through ``obtainer``."""
    pool = gf.FeaturePool(name="noise", label=0, probability=1.0, noise_obtainer=obtainer)
    return gf.ComposedDataset(pools=[pool], seed=7, group=split, input_variables=STRAIN,
                              output_variables=[RV.POOL_LABEL], **world_settings())


def recorded(onsource, offsource, fixture, padded=False, adjacent=True):
    """
    The footprints of rows of the world's strain as recorded: refuses a reversed
    or negated row. A served onsource is cropped; a raw one (``padded``) is not.
    A real event's offsource is not ``adjacent`` to its onsource.
    """
    return footprints(np.asarray(onsource), np.asarray(offsource), WORLD_IFOS, fixture.encoding,
                      fx.window(fixture.encoding.sample_rate_hertz), onsource_cropped=not padded,
                      adjacent=adjacent)


def reversals_of(row, ifo_name, fixture) -> int:
    """The SignReversal and TimeReversal bits a row of the world's strain carries."""
    row = np.asarray(row, np.float64)
    bits = 0
    if np.all(row < 0):
        row, bits = -row, bits | gf.SignReversal.bit
    try:
        decode_row(row, ifo_name, fixture.encoding)
    except DecodeRefused:
        decode_row(row[::-1], ifo_name, fixture.encoding)
        bits |= gf.TimeReversal.bit
    return bits


def test_augmentation_stays_in_split(world, monkeypatch):
    """
    Augmentation never reaches evaluation data. One obtainer, with sign and time
    reversal at probability 1, serves train, validate and test, through a dataset
    and a composed dataset's pool, and the validator: every evaluation row is the
    strain as recorded, forward and positive, and every training row is that
    strain negated and reversed. The validator's rows are every window it scores,
    served or read raw and conditioned (`model_inputs`): its time-slid noise in both
    splits, and a real event in test, read at the rate real events are read at.
    Real events refuse time reversal and random shift.
    """
    from gravyflow.src.validate import ValidationConfig
    from gravyflow.src.validate.bank import ValidationBank

    fixture = augmented_world()
    with world(fixture):
        noise = fx.noise_obtainer(fixture, augmentations=list(REVERSALS))
        for split in SPLITS:
            for build in (world_dataset, pool_dataset):
                x, _ = build(noise, split)[0]
                onsource, offsource = (np.asarray(x[key.name]) for key in STRAIN)
                if split == "train":
                    with pytest.raises(DecodeRefused, match="sign-flipped"):
                        recorded(onsource, offsource, fixture)
                    onsource, offsource = (-rows[..., ::-1] for rows in (onsource, offsource))
                recorded(onsource, offsource, fixture)

    fixture = validator_world()
    with world(fixture):
        noise = fx.noise_obtainer(fixture, augmentations=list(REVERSALS))
        tap = DatasetTap(monkeypatch, STRAIN)
        args = dict(noise_obtainer=noise, input_variables=[RV.WHITENED_ONSOURCE],
                    **world_settings(rate=EVENT_RATE))
        for group, events in (("validate", 0), ("test", 1)):
            config = ValidationConfig(group=group, num_examples=16, batch_size=8)
            bank = ValidationBank(logit_model(EVENT_RATE), args, config)
            bank.generate_noise()
            bank.generate_real_events([gf.ObservingRun.O3])
            assert [event["status"] for event in bank.real_events] == ["scored"] * events
            (stage,) = tap.take()
            assert stage.group == group
            for batch in stage.batches:
                recorded(batch[RV.ONSOURCE.name], batch[RV.OFFSOURCE.name], fixture)
            # Conditioned in the order scored: the background's windows, then the events'.
            for k, windows in enumerate(stage.prepared):
                recorded(windows[RV.ONSOURCE.name], windows[RV.OFFSOURCE.name], fixture, padded=True,
                         adjacent=k < len(stage.prepared) - events)
            # Every window scored was read here.
            seen = sum(len(rows[RV.ONSOURCE.name]) for rows in (*stage.batches, *stage.prepared))
            assert seen == len(bank.far_scores) + events, (group, seen)

    for labels in ([gf.DataLabel.EVENTS], [gf.DataLabel.GLITCHES, gf.EventConfidence.CONFIDENT]):
        for augmentation in (gf.TimeReversal(), gf.RandomShift()):
            with pytest.raises(ValueError, match="real events"):
                gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=labels,
                                   augmentations=[augmentation])
            # Events by number are refused too: a number names no kind (GlitchType.CHIRP is 2 as well).
            with pytest.raises(ValueError, match="not a data label"):
                gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.EVENTS.value],
                                   augmentations=[augmentation])


def test_composed_pools_augment_only_their_own_training_examples(world):
    """
    A composed dataset augments each pool's noise by that pool's own source, for
    the dataset's split. In train, every example carries its own pool's
    augmentations, drawn per example and flagged in AUGMENTATIONS bit for bit; in
    validate and test, every row is the strain as recorded, flagged 0.
    """
    fixture = augmented_world()
    specs = {  # POOL_LABEL: (probability, the pool's augmentations)
        0: (0.4, list(REVERSALS)),
        1: (0.3, [gf.SignReversal(probability=0.5), gf.TimeReversal(probability=0.5)]),
        2: (0.3, None),
    }
    with world(fixture):
        pools = [gf.FeaturePool(name=f"pool_{label}", label=label, probability=probability,
                                noise_obtainer=fx.noise_obtainer(fixture, augmentations=spec))
                 for label, (probability, spec) in specs.items()]
        for split in SPLITS:
            ds = gf.ComposedDataset(pools=pools, seed=7, group=split, input_variables=STRAIN,
                                    output_variables=[RV.POOL_LABEL, RV.AUGMENTATIONS], **world_settings(16))
            seen = {label: set() for label in specs}
            for k in range(3):
                x, y = ds[k]
                flags, labels = np.asarray(y[RV.AUGMENTATIONS.name]), np.asarray(y[RV.POOL_LABEL.name])
                assert flags.dtype == np.float32 and flags.shape == (16,), (split, flags.dtype, flags.shape)
                for b in range(16):
                    bits = {reversals_of(x[key.name][b, d], name, fixture)
                            for key in STRAIN for d, name in enumerate(WORLD_IFOS)}
                    assert bits == {int(flags[b])}, (split, k, b, bits, flags[b])
                    seen[int(labels[b])].add(int(flags[b]))
            if split == "train":
                assert seen[0] == {3} and seen[2] == {0}, seen
                assert len(seen[1]) > 1, f"pool 1 drew one choice for every example: {seen}"
            else:
                assert seen == {0: {0}, 1: {0}, 2: {0}}, (split, seen)


def test_a_default_batch_is_unaugmented(world):
    """No augmentation by default: a training batch is the strain as recorded, flagged 0."""
    fixture = augmented_world()
    with world(fixture):
        noise = fx.noise_obtainer(fixture, augmentations=None)
        ds = world_dataset(noise, "train")
        for k in range(4):
            x, _ = ds[k]
            recorded(x[RV.ONSOURCE.name], x[RV.OFFSOURCE.name], fixture)
        _, y = world_dataset(noise, "train", outputs=[RV.AUGMENTATIONS])[0]
        np.testing.assert_array_equal(y[RV.AUGMENTATIONS.name], np.zeros(8, np.float32))


def test_each_example_draws_once_for_both_windows_and_is_flagged(world):
    """
    Each example draws each augmentation once, from its own stream: its onsource
    and offsource, on every detector, are reversed together or not at all;
    examples of one batch differ; and AUGMENTATIONS flags exactly what was applied.
    """
    fixture = augmented_world()
    with world(fixture):
        noise = fx.noise_obtainer(fixture, augmentations=[
            gf.SignReversal(probability=0.5), gf.TimeReversal(probability=0.5)])
        ds = world_dataset(noise, "train", outputs=[RV.AUGMENTATIONS], batch=16)
        seen = []
        for k in range(2):
            x, y = ds[k]
            flags = np.asarray(y[RV.AUGMENTATIONS.name])
            assert flags.dtype == np.float32 and flags.shape == (16,)
            for b in range(16):
                bits = {reversals_of(x[key.name][b, d], name, fixture)
                        for key in STRAIN for d, name in enumerate(WORLD_IFOS)}
                assert bits == {int(flags[b])}, (k, b, bits, flags[b])
                seen.append(int(flags[b]))
        assert set(seen) == {0, 1, 2, 3}, seen


def test_injections_are_never_augmented(world):
    """
    Augmentation transforms the raw strain before injection: a signal injected
    into negated strain is the same signal, never negated with it.
    """
    fixture = augmented_world()
    outputs = [RV.INJECTIONS]
    with world(fixture):
        drawn = {}
        for name, spec in (("plain", []), ("negated", [gf.SignReversal(probability=1.0)])):
            noise = fx.noise_obtainer(fixture, augmentations=spec)
            ds = gf.Dataset(noise_obtainer=noise, group="train", seed=7, input_variables=STRAIN,
                            output_variables=outputs, waveform_generators=[burst()], **world_settings())
            x, y = ds[0]
            drawn[name] = np.asarray(x[RV.ONSOURCE.name], np.float64), np.asarray(y[RV.INJECTIONS.name])
    (plain, injection), (negated, same_injection) = drawn["plain"], drawn["negated"]
    np.testing.assert_array_equal(injection, same_injection)
    signal = injection[:, 0].astype(np.float64)
    # The strain cancels; the signal, injected into both as it is, doubles.
    np.testing.assert_allclose(negated + plain, 2.0 * signal, atol=0.01 * np.max(np.abs(signal)))


def test_toggling_an_augmentation_moves_no_window(world):
    """
    Augmentation is drawn apart from sampling, and does not change how densely
    windows are drawn: adding augmentations changes no window drawn.
    """
    fixture = augmented_world()
    specs = {
        "none": [],
        "sign": [gf.SignReversal(probability=0.5)],
        "sign and noise": [gf.SignReversal(probability=0.5), gf.AddNoise(probability=0.5, amplitude=0.1)],
    }
    with world(fixture):
        starts = {}
        for name, spec in specs.items():
            # A low saturation, so a row serves a few batches and the reader moves between rows.
            noise = fx.noise_obtainer(fixture, augmentations=spec, saturation=0.125)
            ds = world_dataset(noise, "train")
            starts[name] = [np.asarray(ds[k][1][RV.START_GPS_TIME.name]) for k in range(12)]
    rows = {tuple(np.floor(start[0] / fx.BLOCK_SECONDS)) for start in starts["none"]}
    assert len(rows) > 1, "the batches compared must come from more than one row"
    for name in specs:
        for k, start in enumerate(starts[name]):
            np.testing.assert_array_equal(start, starts["none"][k], err_msg=f"{name}, batch {k}")


# ------------------------------------------------ real noise, addressed by index

#: A low saturation, so the odd pieces' pass is about twenty batches and passes turn over.
REAL_SATURATION = 0.05
REAL_STEPS = 6
#: Two training epochs' addresses, as (epoch, k).
REAL_ADDRESSES = [(epoch, k) for epoch in range(2) for k in range(REAL_STEPS)]


def real_dataset(fixture, split="train", steps=REAL_STEPS, **overrides):
    """A dataset of the fixture's real noise, drawn at random."""
    return fx.dataset(fixture, split, steps_per_epoch=steps, saturation=REAL_SATURATION, **overrides)


def outcomes(ds, addresses):
    """Each address's batch digest, or its refusal by type and message, in the order given."""
    served = {}
    for epoch, k in addresses:
        ds.epoch = epoch
        try:
            served[(epoch, k)] = ("served", batch_digest(ds[k]))
        except gf.StrainUnavailable as error:
            served[(epoch, k)] = ("refused", type(error).__name__, str(error))
    return served


def slow_fetches(harness):
    """Each fetch of the world's source waits a few milliseconds, by its tile."""
    import time

    fetch = harness.source.fetch

    def slow(detector, sample_rate_hertz, interval):
        time.sleep(0.002 * (int(interval[0]) // 64 % 5))
        return fetch(detector, sample_rate_hertz, interval)

    harness.monkeypatch.setattr(harness.source, "fetch", slow)


def refuse_tiles(harness, starts):
    """The world's source refuses every fetch of the tiles from GPS ``starts`` (all, if None)."""
    from gravyflow.src.dataset.store import read

    fetch = harness.source.fetch

    def refusing(detector, sample_rate_hertz, interval):
        if starts is None or float(interval[0]) in starts:
            raise gf.StrainUnavailable(harness.source, detector, interval, "planted outage")
        return fetch(detector, sample_rate_hertz, interval)

    harness.monkeypatch.setattr(harness.source, "fetch", refusing)
    harness.monkeypatch.setattr(read, "FETCH_BACKOFF_SECONDS", 0.0)


def tile_at(gps, tile_seconds=64.0):
    """The GPS start of the world's tile holding ``gps``."""
    return float(np.floor(gps / tile_seconds) * tile_seconds)


#: The orders a variant asks for the addresses in.
ORDERS = {
    "in order": lambda addresses: list(addresses),
    "reversed": lambda addresses: list(reversed(addresses)),
    "shuffled": lambda addresses: [addresses[i] for i in np.random.default_rng(5).permutation(len(addresses))],
}
#: A fresh dataset, asked in an order, reading ahead as set (the obtainer's prefetch knobs).
PURITY_VARIANTS = {
    "reversed": ("reversed", {}),
    "shuffled": ("shuffled", {}),
    "read on demand": ("in order", dict(prefetch_depth=0)),
    "one thread": ("in order", dict(prefetch_workers=1)),
    "deep on eight threads": ("shuffled", dict(prefetch_depth=4, prefetch_workers=8)),
}


def purity_mismatches(world, fixture, refused_starts=()):
    """
    The variants whose batches differ from the reference's: each a fresh dataset in its
    own cold world, with slow fetches (and ``refused_starts``' tiles refused), asked
    for ``REAL_ADDRESSES`` in its order. Returns the reference outcomes and the mismatches.
    """
    def run(order, knobs):
        with world(fixture) as harness:
            slow_fetches(harness)
            if refused_starts:
                refuse_tiles(harness, set(refused_starts))
            served = outcomes(real_dataset(fixture, **knobs), ORDERS[order](REAL_ADDRESSES))
            return served, harness.source.snapshot()[0]

    reference, fetches = run("in order", {})
    mismatches = {}
    for name, (order, knobs) in PURITY_VARIANTS.items():
        served, _ = run(order, knobs)
        differ = [address for address in REAL_ADDRESSES if served[address] != reference[address]]
        if differ:
            mismatches[name] = differ
    return reference, fetches, mismatches


def test_real_batches_are_pure_functions_of_their_address(world):
    """
    Real noise drawn at random is addressed by index: ``ds[k]`` at epoch e is the same
    bytes from a fresh object, in any call order, read on demand or ahead at any depth
    on one thread or eight, with fetch latency; and reading ahead runs, on the
    prefetch threads.
    """
    reference, fetches, mismatches = purity_mismatches(world, fx.odd_pieces())
    assert not mismatches, f"batches that depend on how they were asked for: {mismatches}"
    assert all(outcome[0] == "served" for outcome in reference.values())
    ahead = [fetch for fetch in fetches if fetch.thread.startswith("gfstore-prefetch-")]
    assert ahead, f"nothing was read ahead: fetch threads {sorted({fetch.thread for fetch in fetches})}"


def test_real_refusals_are_pure_functions_of_their_address(world):
    """
    With tiles that the source refuses, every address is the same batch or the same
    named refusal however it is asked for: a read ahead that fails refuses its batch
    by the same name when it is asked for, and never moves another batch.
    """
    fixture = fx.odd_pieces()
    long_piece = fixture.segments["H1"][2]
    refused = [tile_at(long_piece[0] + 640.0), tile_at(long_piece[0] + 1280.0)]
    reference, _, mismatches = purity_mismatches(world, fixture, refused)
    assert not mismatches, f"batches that depend on how they were asked for: {mismatches}"
    kinds = [outcome[0] for outcome in reference.values()]
    assert "served" in kinds and "refused" in kinds, f"the planted outage must refuse some batches only: {kinds}"
    for outcome in reference.values():
        if outcome[0] == "refused":
            assert outcome[1] == "TimelineContradiction" and "planted outage" in outcome[2]


def _pieces_of(gps, pieces):
    """The index of the piece each GPS time lies in."""
    index = np.searchsorted(pieces[:, 0], gps, side="right") - 1
    assert np.all((index >= 0) & (gps < pieces[np.maximum(index, 0), 1])), "a window outside every piece"
    return index


#: The longest run of consecutive batches one piece may feed alone.
LONGEST_LONE_RUN = 2


def test_real_batches_mix_their_rows(world):
    """
    Every batch has exactly B examples, drawn across the split's rows: no piece feeds
    a run of more than ``LONGEST_LONE_RUN`` consecutive batches alone (one segment fed
    hundreds), and every piece due a batch a pass is served.
    """
    fixture = fx.odd_pieces()
    pieces = fixture.segments["H1"]
    with world(fixture):
        ds = real_dataset(fixture, steps=60)
        batches = [np.asarray(ds[k][1][RV.START_GPS_TIME.name]) for k in range(60)]
    assert all(batch.shape == (8, len(WORLD_IFOS)) for batch in batches)
    alone = [set(_pieces_of(batch[:, 0], pieces).tolist()) for batch in batches]
    run = longest = 0
    for k, used in enumerate(alone):
        run = run + 1 if len(used) == 1 and k and used == alone[k - 1] else int(len(used) == 1)
        longest = max(longest, run)
    assert longest <= LONGEST_LONE_RUN, f"one piece fed {longest} consecutive batches alone"
    assert set().union(*alone) >= {0, 1, 2}, f"pieces served: {sorted(set().union(*alone))}"


def test_training_epochs_differ_and_evaluation_epochs_repeat(world):
    """
    Each training epoch draws its own batches; validate and test serve the same
    batches every epoch, whatever came before.
    """
    fixture = augmented_world()
    with world(fixture):
        train = outcomes(real_dataset(fixture, "train"), REAL_ADDRESSES)
        same = [k for k in range(REAL_STEPS) if train[(0, k)] == train[(1, k)]]
        assert not same, f"training batches {same} repeat in the next epoch"
        for split in ("validate", "test"):
            ds = real_dataset(fixture, split)
            first = [batch_digest(ds[k]) for k in range(REAL_STEPS)]
            ds.on_epoch_end()
            ds.on_epoch_end()
            assert [batch_digest(ds[k]) for k in reversed(range(REAL_STEPS))][::-1] == first, split


def test_a_refusal_reaches_the_caller_by_type_and_exclusion(world):
    """
    A dataset passes a read's refusal on as it was raised: a ``TimelineContradiction``
    keeps its type and names the ``exclusions`` entry that plans around it.
    """
    fixture = fx.odd_pieces()
    with world(fixture) as harness:
        refuse_tiles(harness, None)
        with pytest.raises(gf.StrainUnavailable) as refused:
            real_dataset(fixture)[0]
    assert type(refused.value).__name__ == "TimelineContradiction"
    detector, span, reason = refused.value.exclusion
    assert detector in WORLD_IFOS and span[1] - span[0] == 64.0 and "could not deliver" in reason
    assert "planted outage" in str(refused.value)


def test_availability_is_frozen_and_fingerprinted_when_the_dataset_is_built(world):
    """
    A dataset freezes its noise's readable and noise-admissible time as it is built:
    the frozen sets are in the obtainer's specification, so in its fingerprint, and a
    timeline changed afterwards moves none of its batches; a dataset built after the
    change has another fingerprint.
    """
    from gravyflow.src.validate.provenance import config_fingerprint
    from .fakes import write_timelines

    fixture = fx.odd_pieces()
    addresses = [(0, k) for k in range(REAL_STEPS)]
    without_long_piece = {ifo: np.delete(segments, 2, axis=0) for ifo, segments in fixture.segments.items()}
    with world(fixture):
        built = real_dataset(fixture)
        reference = outcomes(real_dataset(fixture), addresses)
        write_timelines(without_long_piece, fixture.flag_gaps)
        assert outcomes(built, addresses) == reference
        after = real_dataset(fixture)
        moved = outcomes(after, addresses)
    reader = built.noise_obtainer.ifo_data_obtainer
    frozen = reader.specification()["availability"]
    np.testing.assert_array_equal(frozen.of(gf.IFO.H1).readable, fixture.segments["H1"])
    assert config_fingerprint(reader) != config_fingerprint(after.noise_obtainer.ifo_data_obtainer)
    assert moved != reference


def test_prepare_reads_ahead_then_the_batches_read_nothing(world):
    """
    ``ds.prepare(epochs=E)`` commits every tile the first E epochs need; those batches
    then fetch nothing. It names its one argument.
    """
    fixture = fx.odd_pieces()
    with world(fixture) as harness:
        ds = real_dataset(fixture)
        assert ds.prepare(epochs=2) == len(REAL_ADDRESSES)
        prepared = len(harness.source.snapshot()[0])
        assert prepared > 0
        outcomes(ds, REAL_ADDRESSES)
        assert len(harness.source.snapshot()[0]) == prepared
        with pytest.raises(ValueError, match="one of batches=N or epochs=E"):
            ds.prepare()


def test_prepare_lists_every_contradiction_at_once(world):
    """
    ``ds.prepare`` meets every contradiction its batches would, and refuses once,
    naming each with the ``exclusions`` entries that plan around them all.
    """
    fixture = fx.odd_pieces()
    long_piece = fixture.segments["H1"][2]
    refused = {tile_at(long_piece[0] + 640.0), tile_at(long_piece[0] + 1280.0)}
    with world(fixture) as harness:
        refuse_tiles(harness, refused)
        with pytest.raises(gf.Unprepared) as unprepared:
            real_dataset(fixture).prepare(epochs=2)
    spans = {exclusion[1][0] for exclusion in unprepared.value.exclusions}
    assert spans == refused, f"contradictions named: {unprepared.value.exclusions}"
    assert len(unprepared.value.contradictions) == len(unprepared.value.exclusions) >= 2
    assert "exclusions=" in str(unprepared.value)
