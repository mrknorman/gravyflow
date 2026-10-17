"""
Determinism: one seed tree, batches addressed by index, splits independent.

Every random draw comes from a stream of a gf.SeedTree addressed by (split,
consumer, epoch, k). These tests check the tree itself, that the objects a
caller builds hold no random state, and that datasets are pure functions of
their address with independent splits.
"""

import hashlib
import pickle
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.rng import DIRECT_SPLIT, platform, resolve_streams

SAMPLE_RATE = 512.0
BATCH = 4


def snr_uniform(low=8.0, high=20.0):
    return gf.ScalingMethod(
        gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM),
        gf.ScalingTypes.SNR,
    )


def wnb(**kwargs):
    return gf.WNBGenerator(
        scaling_method=kwargs.pop("scaling_method", snr_uniform()),
        duration_seconds=gf.Distribution(min_=0.1, max_=0.4, type_=gf.DistributionType.UNIFORM),
        min_frequency_hertz=gf.Distribution(min_=40.0, max_=80.0, type_=gf.DistributionType.UNIFORM),
        max_frequency_hertz=gf.Distribution(min_=120.0, max_=200.0, type_=gf.DistributionType.UNIFORM),
        **kwargs,
    )


def dataset(seed=None, group="train", generators=None, noise=None, **kwargs):
    return gf.Dataset(
        seed=seed,
        group=group,
        noise_obtainer=noise or gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1, gf.IFO.H1]),
        waveform_generators=[wnb()] if generators is None else generators,
        sample_rate_hertz=SAMPLE_RATE,
        offsource_duration_seconds=4.0,
        num_examples_per_batch=BATCH,
        steps_per_epoch=kwargs.pop("steps_per_epoch", 8),
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=[gf.ScalingTypes.SNR, gf.WaveformParameters.RIGHT_ASCENSION_RADIANS],
        **kwargs,
    )


def digest(batch):
    x, y = batch
    h = hashlib.sha256()
    for part in (x, y):
        for key in sorted(part):
            h.update(key.encode())
            h.update(np.ascontiguousarray(np.asarray(part[key])).tobytes())
    return h.hexdigest()


class TestSeedTree:

    def test_same_path_same_stream(self):
        a = gf.SeedTree(42).stream("train", "noise/L1/on", 0, 3).random(4)
        b = gf.SeedTree(42).stream("train", "noise/L1/on", 0, 3).random(4)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("other", [
        (43, "train", "noise/L1/on", 0, 3),
        (42, "validate", "noise/L1/on", 0, 3),
        (42, "train", "noise/H1/on", 0, 3),
        (42, "train", "noise/L1/off", 0, 3),
        (42, "train", "noise/L1/on", 1, 3),
        (42, "train", "noise/L1/on", 0, 4),
        (42, "train", "noise/L1/on", None, 3),
        (42, "train", "noise/L1/on", 0, None),
    ])
    def test_every_coordinate_moves_the_stream(self, other):
        base = gf.SeedTree(42).stream("train", "noise/L1/on", 0, 3).random(4)
        root, *path = other
        assert not np.array_equal(base, gf.SeedTree(root).stream(*path).random(4))

    def test_seed_zero_is_its_own_root(self):
        # B155: seed 0 was mapped to the default seed.
        assert gf.SeedTree(0).root == 0
        assert gf.SeedTree(None).root == gf.Defaults.seed
        assert gf.SeedTree(None).source == "default" and gf.SeedTree(0).source == "user"
        zero = gf.SeedTree(0).stream("train", "pool/assign").random(4)
        default = gf.SeedTree(None).stream("train", "pool/assign").random(4)
        assert not np.array_equal(zero, default)

    @pytest.mark.parametrize("seed", [-1, 1.5, "1", True])
    def test_bad_roots_are_refused(self, seed):
        with pytest.raises((TypeError, ValueError)):
            gf.SeedTree(seed)

    @pytest.mark.parametrize("coordinate", [1.0, 1.7, True, "1", np.float32(1.0)])
    def test_non_integer_coordinates_are_refused(self, coordinate):
        # An epoch or k of 1.7, True or "1" aliased batch 1.
        tree = gf.SeedTree(1)
        with pytest.raises(TypeError):
            tree.stream("train", "pool/assign", coordinate, 0)
        with pytest.raises(TypeError):
            tree.stream("train", "pool/assign", 0, coordinate)

    def test_integer_coordinates_of_any_kind_agree(self):
        tree = gf.SeedTree(1)
        a = tree.stream("train", "pool/assign", np.int64(2), np.uint8(3)).random(2)
        np.testing.assert_array_equal(a, tree.stream("train", "pool/assign", 2, 3).random(2))

    def test_the_direct_split_is_reserved(self):
        with pytest.raises(ValueError, match="reserved"):
            gf.SeedTree(1).batch(DIRECT_SPLIT, 0, 0)
        with pytest.raises(ValueError, match="reserved"):
            dataset(group=DIRECT_SPLIT)

    @pytest.mark.parametrize("consumer", ["noise", "noisy/L1/on", "wave/x/param", "sky/a/b"])
    def test_unregistered_consumers_are_refused(self, consumer):
        with pytest.raises(ValueError):
            gf.SeedTree(1).stream("train", consumer)

    def test_scoped_consumers_are_registered(self):
        tree = gf.SeedTree(1)
        for consumer in ("pool/noise/noise/L1/on", "validator/far/noise/H1/off",
                         "wave/0:CBCGenerator/comp/1/param/mass_1_msun", "wave/g/param/spin_1_in/2",
                         "inject/g/shift", "scale/g", "pol/g"):
            tree.stream("train", consumer)

    def test_sub_tree_never_meets_its_parent(self):
        tree = gf.SeedTree(42)
        assert not np.array_equal(
            tree.stream("test", "noise/L1/on", 0, 0).random(4),
            tree.sub("validator/far").stream("test", "noise/L1/on", 0, 0).random(4),
        )

    def test_keys_are_distinct_across_many_batches(self):
        # B154: 30-bit chained seeds repeated; tree keys are 64-bit and addressed.
        tree = gf.SeedTree(7)
        words = {
            tuple(np.asarray(tree.seed_sequence("train", "noise/L1/on", e, k).generate_state(2)))
            for e in range(4) for k in range(2500)
        }
        assert len(words) == 10000

    def test_stream_is_the_same_in_another_process(self):
        here = gf.SeedTree(42).stream("validate", "sky/g", 2, 9).random(3).tolist()
        code = textwrap.dedent("""
            import gravyflow as gf
            print(gf.SeedTree(42).stream("validate", "sky/g", 2, 9).random(3).tolist())
        """)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip().splitlines()[-1] == str(here)


class TestBitReproducibleKernels:
    """
    The suite's oracles compare bits, which needs XLA's CPU kernels to give the
    same bits for the same inputs: the session runs them single-threaded
    (conftest), as the documented CPU run does.
    """

    def test_the_session_runs_single_threaded_kernels(self):
        assert platform.single_threaded_cpu(), platform.xla_flags()
        assert not platform.multi_threaded_cpu()

    def test_whitening_is_bit_reproducible_between_calls(self):
        # Multi-threaded kernels whiten these arrays to one of two results,
        # differing by up to 1e-5 of the peak, about once in 500 calls.
        rng = np.random.default_rng(26)
        offsource = rng.normal(size=(4, 2, 16 * 2048)).astype(np.float32)
        onsource = rng.normal(size=(4, 2, 2 * 2048)).astype(np.float32)
        digests = {
            hashlib.sha256(np.asarray(gf.whiten(onsource, offsource, 2048.0)).tobytes()).hexdigest()
            for _ in range(1000)
        }
        assert len(digests) == 1


class TestStatelessObjects:

    def test_distribution_is_not_changed_by_sampling(self):
        distribution = gf.Distribution(min_=1.0, max_=3.0, type_=gf.DistributionType.NORMAL, mean=2.0, std=1.0)
        before = (distribution.min_, distribution.max_, distribution.mean, distribution.std)
        distribution.sample(16, np.random.default_rng(0))
        assert (distribution.min_, distribution.max_, distribution.mean, distribution.std) == before

    def test_integer_uniform_reaches_its_max(self):
        # B157: rng.uniform then int() never gave the maximum.
        distribution = gf.Distribution(min_=1, max_=3, type_=gf.DistributionType.UNIFORM, dtype=int)
        samples = distribution.sample(2000, np.random.default_rng(0))
        assert set(samples.tolist()) == {1, 2, 3}

    def test_sample_without_rng_is_deprecated(self):
        with pytest.warns(DeprecationWarning):
            gf.Distribution(value=1.0).sample(1)

    def test_direct_generate_is_deterministic(self):
        # B156: generate() without a seed crashed for WNB with injection_chance.
        generator = wnb(injection_chance=0.5)
        a, _ = generator.generate(8, SAMPLE_RATE, 1.0)
        b, _ = generator.generate(8, SAMPLE_RATE, 1.0)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_incoherent_components_draw_independently(self):
        # B33: every component was seeded the same.
        generator = gf.IncoherentGenerator([wnb(), wnb()])
        waveforms, _ = generator.generate(4, SAMPLE_RATE, 1.0)
        waveforms = np.asarray(waveforms)
        assert not np.array_equal(waveforms[:, 0], waveforms[:, 1])

    def direct_labels(self, k, streams_of=None):
        """Batch k of a direct InjectionGenerator, injected with the streams ``streams_of`` gives."""
        generator = wnb(network=[gf.IFO.L1, gf.IFO.H1])
        injector = gf.InjectionGenerator([generator])
        batches = injector(sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0,
                           crop_duration_seconds=0.5, num_examples_per_batch=BATCH)
        for _ in range(k + 1):
            injections, mask, parameters = next(batches)
        noise = np.random.default_rng(0)
        onsource = noise.normal(size=(BATCH, 2, int(2.0 * SAMPLE_RATE))).astype(np.float32)
        offsource = noise.normal(size=(BATCH, 2, int(4.0 * SAMPLE_RATE))).astype(np.float32)
        _, _, labels = injector.add_injections_to_onsource(
            injections, mask, onsource, offsource,
            parameters_to_return=[gf.ScalingTypes.SNR, gf.WaveformParameters.RIGHT_ASCENSION_RADIANS],
            onsource_duration_seconds=1.0, injection_parameters=parameters,
            streams=None if streams_of is None else streams_of(injector, k),
        )
        return {key: np.asarray(value) for key, value in labels.items()}

    def test_direct_batches_draw_their_own_sky_and_targets(self):
        # A direct iterator advanced k, but the injection layer drew every
        # batch's sky and targets from k=None.
        # The labels also carry every parameter the generator drew, some of
        # them constant; the injection layer's own draws are the sky, the
        # polarisation and the SNR target.
        first, second = (self.direct_labels(k, lambda injector, k: injector.streams(k)) for k in (0, 1))
        drawn_here = [gf.ScalingTypes.SNR, gf.WaveformParameters.RIGHT_ASCENSION_RADIANS,
                      gf.WaveformParameters.DECLINATION_RADIANS, gf.WaveformParameters.POLARIZATION_RADIANS]
        for key in drawn_here:
            assert not np.array_equal(first[key], second[key]), key

    def test_injecting_without_streams_is_deprecated(self):
        with pytest.warns(DeprecationWarning, match="streams"):
            self.direct_labels(0)

    def test_incoherent_components_share_their_presence(self):
        # Each component drew its own injection chance, so an example could
        # carry a signal in one detector only and still be labelled injected.
        incoherent = gf.IncoherentGenerator([wnb(injection_chance=0.5), wnb(injection_chance=0.5)])
        waveforms, _ = incoherent.generate(64, SAMPLE_RATE, 1.0)
        present = np.any(np.asarray(waveforms) != 0.0, axis=(2, 3))  # (examples, components)
        assert np.array_equal(present[:, 0], present[:, 1])
        assert 0 < present[:, 0].sum() < 64

    def test_random_network_geometry_draws_from_its_rng(self):
        # The geometry was drawn through the deprecated rng-less sample().
        def uniform(low, high):
            return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)

        geometry = {
            "num_detectors": 2, "latitude_radians": uniform(-1.0, 1.0),
            "longitude_radians": uniform(0.0, 6.0), "y_angle_radians": uniform(0.0, 6.0),
            "x_length_meters": 4000.0, "y_length_meters": 4000.0, "height_meters": 0.0,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            a = gf.Network(geometry, rng=np.random.default_rng(3))
            b = gf.Network(geometry, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(np.asarray(a.location), np.asarray(b.location))
        assert geometry["num_detectors"] == 2  # the caller's dict is not changed
        with pytest.warns(DeprecationWarning, match="no rng"):
            gf.Network(geometry)

    def test_network_needs_every_sky_angle(self):
        network = gf.Network([gf.IFO.L1])
        with pytest.raises(ValueError, match="polarization"):
            network.project_wave(np.zeros((2, 2, 64), np.float32), SAMPLE_RATE, 0.1, 0.2, None)

    def test_curriculum_has_no_counter(self):
        curriculum = gf.Curriculum(
            start=gf.Distribution(min_=50.0, max_=60.0, type_=gf.DistributionType.UNIFORM),
            end=gf.Distribution(min_=5.0, max_=10.0, type_=gf.DistributionType.UNIFORM),
            num_epochs=3,
        )
        rng = np.random.default_rng(0)
        for _ in range(5):
            curriculum.sample(4, rng, epoch=0)
        assert curriculum.at(0).min_ == 50.0
        assert curriculum.at(None).min_ == 5.0


class TestDeprecations:
    """The replaced seed API warns for one release, where a user will see it."""

    SCRIPT = textwrap.dedent("""
        import gravyflow as gf
        uniform = dict(type_=gf.DistributionType.UNIFORM)
        gf.Distribution(min_=1.0, max_=2.0, seed=1, **uniform)
        gf.Distribution(min_=1.0, max_=2.0, **uniform).sample(2)
        curriculum = gf.Curriculum(
            start=gf.Distribution(min_=80.0, max_=100.0, **uniform),
            end=gf.Distribution(min_=5.0, max_=15.0, **uniform),
            seed=1, steps_per_epoch=10,
        )
        curriculum.sample(2)
        curriculum.min_
        gf.WNBGenerator(
            duration_seconds=0.2, min_frequency_hertz=40.0, max_frequency_hertz=200.0,
        ).generate(2, 512.0, 1.0, seed=1)
        gf.Network([gf.IFO.L1], seed=1)
    """)

    def test_deprecations_show_under_default_filters(self, tmp_path):
        # A fixed stacklevel pointed most of these warnings at library code,
        # where Python's default filters drop a DeprecationWarning.
        import os
        script = tmp_path / "user_script.py"
        script.write_text(self.SCRIPT)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
        env["PYTHONPATH"] = os.pathsep.join(
            [str(__import__("pathlib").Path(gf.__file__).parents[1]), env.get("PYTHONPATH", "")]
        )
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        for name in ("Distribution(seed=)", "Distribution.sample() without an rng", "Curriculum(seed=)",
                     "Curriculum(steps_per_epoch=)", "Curriculum.sample() without an rng",
                     "Curriculum.min_", "WNBGenerator.generate(seed=)", "Network(seed=)"):
            assert f"{name} is deprecated" in out.stderr, name
            assert "user_script.py" in out.stderr

    def test_renamed_functions_warn_and_still_run(self):
        import jax
        from gravyflow.src.dataset.noise import noise as gf_noise

        args = (2, SAMPLE_RATE, 1.0, np.full(2, 0.3), np.full(2, 40.0), np.full(2, 120.0))
        with pytest.warns(DeprecationWarning, match="gf.wnb"):
            by_seed = gf.wnb(*args, 5)
        np.testing.assert_array_equal(by_seed, gf.wnb(*args, key=jax.random.PRNGKey(5)))

        generator = wnb(injection_chance=0.5)
        waveforms = np.ones((16, 2, 8), np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            masked = generator.apply_injection_chance(waveforms, 3)
        # One warning, for the call made: none for generate(seed=), never called.
        deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1 and "apply_injection_chance" in deprecations[0], deprecations
        np.testing.assert_array_equal(
            masked, generator.apply_injection_chance(
                waveforms, resolve_streams(None, 3, f"wave/{generator.stream_label}"))
        )

        with pytest.warns(DeprecationWarning, match="white_noise_generator"):
            white = next(gf_noise.white_noise_generator(BATCH, [gf.IFO.L1], 1.0, 0.5, 2.0, SAMPLE_RATE, 1))
        with pytest.warns(DeprecationWarning, match="colored_noise_generator"):
            colored = next(gf_noise.colored_noise_generator(BATCH, 1.0, 0.5, 2.0, [gf.IFO.L1], SAMPLE_RATE, None, 1))
        for batch in (white, colored):
            assert np.shape(batch[gf.ReturnVariables.ONSOURCE]) == (BATCH, 1, int(2.0 * SAMPLE_RATE))

    def test_deprecated_curriculum_sampling_keeps_its_old_meaning(self):
        # It drew the end distribution, the same values every call.
        curriculum = gf.Curriculum(
            start=gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM),
            end=gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
        )
        with pytest.warns(DeprecationWarning):
            first, second = curriculum.sample(3), curriculum.sample(3)
        assert not np.array_equal(first, second)
        assert np.all((first >= 80.0) & (first <= 100.0))
        with pytest.warns(DeprecationWarning):
            assert (curriculum.min_, curriculum.max_) == (80.0, 100.0)


class TestIndexedDatasets:

    def test_batch_is_a_pure_function_of_its_index(self):
        ds = dataset(seed=11)
        fifth = digest(ds[5])
        _ = [ds[k] for k in range(5)]
        assert digest(ds[5]) == fifth
        assert digest(dataset(seed=11)[5]) == fifth

    def test_iteration_and_indexing_agree(self):
        ds = dataset(seed=11)
        by_index = [digest(ds[k]) for k in range(3)]
        cursor = dataset(seed=11)
        assert [digest(next(cursor)) for _ in range(3)] == by_index
        assert [digest(batch) for batch, _ in zip(dataset(seed=11), range(3))] == by_index

    def test_index_out_of_range(self):
        with pytest.raises(IndexError):
            dataset(seed=1, steps_per_epoch=2)[2]

    @pytest.mark.parametrize("coordinate", [1.7, True, "1", np.float32(1.0)])
    def test_non_integer_indices_and_epochs_are_refused(self, coordinate):
        # ds[1.7], ds[True] and ds["1"] were all batch 1; so was ds.epoch = 1.6.
        ds = dataset(seed=1, steps_per_epoch=2)
        with pytest.raises(TypeError):
            ds[coordinate]
        with pytest.raises(TypeError):
            ds.epoch = coordinate
        ds.epoch = np.int64(1)
        assert ds.epoch == 1 and digest(ds[np.int64(1)]) == digest(ds[1])

    def test_training_epochs_are_fresh_and_evaluation_fixed(self):
        train, val = dataset(seed=3), dataset(seed=3, group="validate")
        first_train, first_val = digest(train[0]), digest(val[0])
        train.on_epoch_end()
        val.on_epoch_end()
        assert digest(train[0]) != first_train
        assert digest(val[0]) == first_val
        repeat = dataset(seed=3, epoch_mode=gf.EpochMode.REPEAT)
        repeat.on_epoch_end()
        assert digest(repeat[0]) == first_train

    def test_a_target_list_runs_on_across_training_epochs(self):
        # Batch k took targets [k B, (k + 1) B) in every epoch, so a list longer
        # than one epoch's examples was never reached past its first epoch.
        steps = 2
        targets = 8.0 + np.arange(2 * steps * BATCH)
        method = gf.ScalingMethod(targets, gf.ScalingTypes.SNR)

        def snrs(ds):
            return np.concatenate([np.asarray(ds[k][1]["SNR"]).ravel() for k in range(steps)])

        train = dataset(seed=1, steps_per_epoch=steps, generators=[wnb(scaling_method=method)])
        first = snrs(train)
        train.on_epoch_end()
        np.testing.assert_allclose(np.concatenate([first, snrs(train)]), targets, rtol=1e-5)
        evaluation = dataset(seed=1, group="validate", steps_per_epoch=steps,
                             generators=[wnb(scaling_method=method)])
        evaluation.on_epoch_end()
        np.testing.assert_allclose(snrs(evaluation), targets[:steps * BATCH], rtol=1e-5)

    def test_unseeded_splits_differ(self):
        # B28 (identical batches), B102 (equal targets), B36 (equal sky).
        train, val = dataset(), dataset(group="validate")
        for k in range(3):
            (x_t, y_t), (x_v, y_v) = train[k], val[k]
            assert not np.array_equal(x_t["ONSOURCE"], x_v["ONSOURCE"])
            assert not np.array_equal(y_t["SNR"], y_v["SNR"])
            assert not np.array_equal(
                y_t["RIGHT_ASCENSION_RADIANS"], y_v["RIGHT_ASCENSION_RADIANS"]
            )

    def test_generators_draw_independent_skies(self):
        # B107: two generators shared sky, polarisation and targets.
        ds = dataset(seed=5, generators=[wnb(), wnb()])
        _, y = ds[0]
        ra = np.asarray(y["RIGHT_ASCENSION_RADIANS"]).squeeze()
        ra = ra if ra.shape[0] == 2 else ra.T  # (generators, batch)
        assert ra.shape == (2, BATCH)
        assert not np.array_equal(ra[0], ra[1])

    def test_caller_objects_are_never_changed(self):
        noise = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1])
        generator = wnb()
        before = pickle.dumps((noise, generator))
        ds = dataset(seed=2, generators=[generator], noise=noise, ifos=[gf.IFO.L1])
        _ = [ds[k] for k in range(3)]
        assert pickle.dumps((noise, generator)) == before
        assert generator.network is None

    def test_validation_does_not_move_training(self):
        # B120: a validation call changed later training batches.
        noise = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1])
        generator = wnb()
        train = dataset(seed=9, generators=[generator], noise=noise, ifos=[gf.IFO.L1])
        reference = digest(train[1])
        val = dataset(seed=9, group="validate", generators=[generator], noise=noise, ifos=[gf.IFO.L1])
        val[0]
        assert digest(train[1]) == reference
        # B119: a rebuilt dataset repeats the first.
        rebuilt = dataset(seed=9, generators=[generator], noise=noise, ifos=[gf.IFO.L1])
        assert digest(rebuilt[1]) == reference

    def test_composed_pools_do_not_share_streams_across_seeds(self):
        # B121: seed 42's pool 1 equalled seed 43's pool 0.
        def composed(seed):
            pools = [
                gf.FeaturePool(name=f"noise_{i}", label=i, probability=0.5,
                               noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]))
                for i in range(2)
            ]
            return gf.ComposedDataset(
                pools=pools, seed=seed, sample_rate_hertz=SAMPLE_RATE, offsource_duration_seconds=4.0,
                num_examples_per_batch=BATCH, steps_per_epoch=4,
                input_variables=[gf.ReturnVariables.ONSOURCE], output_variables=[gf.ReturnVariables.POOL_LABEL],
                group="train",
            )
        a, b = composed(42), composed(43)
        x_a, y_a = a[0]
        x_b, y_b = b[0]
        for i, j in [(i, j) for i in range(BATCH) for j in range(BATCH)]:
            assert not np.array_equal(np.asarray(x_a["ONSOURCE"])[i], np.asarray(x_b["ONSOURCE"])[j])
        assert digest(composed(42)[0]) == digest(a[0])

    def test_seed_metadata_records_the_root(self):
        assert dataset().seed_metadata["source"] == "default"
        meta = dataset(seed=0, group="test").seed_metadata
        assert (meta["root"], meta["source"], meta["split"], meta["epoch_mode"]) == (0, "user", "test", "REPEAT")

    def test_metadata_says_which_noise_is_read_by_address(self):
        # Real, pseudo-real and transient noise are still read in order.
        from gravyflow.src.dataset.noise.noise import index_addressed
        assert dataset().seed_metadata["index_addressed"] is True
        for noise_type in (gf.NoiseType.WHITE, gf.NoiseType.COLORED):
            assert index_addressed(gf.NoiseObtainer(noise_type=noise_type))
        for noise_type in (gf.NoiseType.PSEUDO_REAL, gf.NoiseType.REAL):
            assert not index_addressed(gf.NoiseObtainer(noise_type=noise_type))
        assert not index_addressed(gf.TransientDataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES]))

    def test_library_draws_no_deprecated_paths(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dataset(seed=1)[0]


class TestKerasEpochs:
    """Under ``model.fit``, Keras epoch e serves the dataset's epoch e."""

    CURRICULUM = gf.Curriculum(
        start=gf.Distribution(min_=80.0, max_=100.0, type_=gf.DistributionType.UNIFORM),
        end=gf.Distribution(min_=5.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
        num_epochs=3,
    )

    def fit(self, ds, epochs, initial_epoch=0, bind=True, steps_per_epoch=None):
        """Fit a tiny model; return, per Keras epoch, the (dataset epoch, SNRs) it was served."""
        import keras

        served = {}
        keras_epoch = [None]

        class Served(keras.utils.PyDataset):
            def __len__(self):
                return len(ds)

            def __getitem__(self, k):
                x, y = ds[k]
                served.setdefault(keras_epoch[0], []).append((ds.epoch, np.asarray(y["SNR"])))
                onsource = np.asarray(x["ONSOURCE"], dtype=np.float32)
                return onsource.reshape(len(onsource), -1), np.ones((len(onsource), 1), np.float32)

            def on_epoch_begin(self):
                ds.on_epoch_begin()

            def on_epoch_end(self):
                ds.on_epoch_end()

        class Marker(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                keras_epoch[0] = epoch

        x, _ = ds[0]
        inputs = int(np.prod(np.shape(x["ONSOURCE"])[1:]))
        model = keras.Sequential([
            keras.Input((inputs,)),
            keras.layers.Dense(1, kernel_initializer="zeros"),
        ])
        model.compile(optimizer="sgd", loss="mse")
        callbacks = [ds.epoch_callback(), Marker()] if bind else [Marker()]
        model.fit(Served(), epochs=epochs, initial_epoch=initial_epoch, shuffle=False,
                  steps_per_epoch=steps_per_epoch, verbose=0, callbacks=callbacks)
        return served

    def assert_served_its_own_epoch(self, served, epochs):
        for epoch in epochs:
            low, high = self.CURRICULUM.at(epoch).min_, self.CURRICULUM.at(epoch).max_
            assert served[epoch], f"Keras epoch {epoch} served nothing"
            for data_epoch, snr in served[epoch]:
                assert data_epoch == epoch
                assert np.all((snr >= low) & (snr <= high)), (epoch, snr, low, high)

    def curriculum_dataset(self, seed=4):
        return dataset(seed=seed, steps_per_epoch=3, generators=[wnb(
            scaling_method=gf.ScalingMethod(self.CURRICULUM, gf.ScalingTypes.SNR)
        )])

    def test_fit_serves_each_epoch_its_own_batches(self):
        served = self.fit(self.curriculum_dataset(), epochs=3)
        self.assert_served_its_own_epoch(served, range(3))

    def test_resumed_fit_serves_the_resumed_epoch(self):
        served = self.fit(self.curriculum_dataset(), epochs=3, initial_epoch=2)
        self.assert_served_its_own_epoch(served, [2])

    def test_fit_with_other_steps_per_epoch_is_refused(self):
        # Keras then keeps one iterator across epochs and prefetches, so an
        # epoch trained batches addressed to the epoch before.
        with pytest.raises(ValueError, match="steps_per_epoch"):
            self.fit(self.curriculum_dataset(), epochs=2, steps_per_epoch=2)

    def test_unbound_fresh_split_warns_under_keras(self):
        with pytest.warns(UserWarning, match="epoch_callback"):
            self.fit(self.curriculum_dataset(), epochs=1, bind=False)
