"""
Curriculum: a distribution that moves with the epoch, with no state.

``at(epoch)`` is the distribution of one training epoch and ``at(None)`` the
end distribution, which evaluation splits use.
"""

import numpy as np
import pytest

import gravyflow as gf


def uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


def curriculum(**kwargs):
    return gf.Curriculum(
        start=kwargs.pop("start", uniform(80.0, 100.0)),
        end=kwargs.pop("end", uniform(0.0, 20.0)),
        num_epochs=kwargs.pop("num_epochs", 10),
        **kwargs,
    )


class TestCurriculumBasics:

    def test_schedule_types_exist(self):
        assert {s.name for s in gf.CurriculumSchedule} == {
            "LINEAR", "EXPONENTIAL", "STEP", "COSINE", "CUSTOM"
        }

    def test_basic_init(self):
        c = curriculum()
        assert c.num_epochs == 10
        assert c.schedule == gf.CurriculumSchedule.LINEAR
        assert c.type_ == gf.DistributionType.UNIFORM

    def test_mismatched_distribution_types_raises(self):
        with pytest.raises(ValueError, match="same type"):
            curriculum(end=gf.Distribution(value=5.0, type_=gf.DistributionType.CONSTANT))

    def test_verbose_is_deprecated(self):
        with pytest.warns(DeprecationWarning):
            curriculum(verbose=True)

    @pytest.mark.parametrize("kwargs", [{"seed": 3}, {"steps_per_epoch": 10}])
    def test_the_seed_and_counter_arguments_are_gone(self, kwargs):
        with pytest.raises(TypeError, match=next(iter(kwargs))):
            curriculum(**kwargs)

    @pytest.mark.parametrize("name", ["reset", "configure", "current_epoch", "min_", "max_", "value", "rng"])
    def test_no_counter_or_current_values(self, name):
        # A Curriculum is a schedule: the distribution of an epoch is at(epoch).
        assert not hasattr(curriculum(), name)

    def test_is_frozen(self):
        import dataclasses
        c = curriculum()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.num_epochs = 3

    def test_sample_needs_an_rng(self):
        with pytest.raises(TypeError, match="rng"):
            curriculum().sample(2)


class TestCurriculumSchedules:

    @pytest.mark.parametrize("epoch, expected", [(0, 0.0), (9, 1.0), (20, 1.0), (None, 1.0)])
    def test_linear_endpoints(self, epoch, expected):
        assert curriculum().progress(epoch) == expected

    def test_linear_midpoint(self):
        assert curriculum().progress(4) == pytest.approx(4 / 9)

    def test_step_schedule(self):
        c = curriculum(schedule=gf.CurriculumSchedule.STEP)
        assert c.progress(4) == 0.0
        assert c.progress(5) == 1.0

    def test_exponential_and_cosine_are_monotone(self):
        for schedule in (gf.CurriculumSchedule.EXPONENTIAL, gf.CurriculumSchedule.COSINE):
            c = curriculum(schedule=schedule)
            values = [c.progress(e) for e in range(10)]
            assert values[0] == pytest.approx(0.0)
            assert all(b >= a for a, b in zip(values, values[1:]))

    def test_custom_schedule(self):
        c = curriculum(schedule=gf.CurriculumSchedule.CUSTOM,
                       custom_schedule_fn=lambda epoch, n: 0.0 if epoch < 5 else 1.0)
        assert c.progress(3) == 0.0
        assert c.progress(5) == 1.0

    def test_custom_schedule_without_function_raises(self):
        with pytest.raises(ValueError, match="custom_schedule_fn"):
            curriculum(schedule=gf.CurriculumSchedule.CUSTOM).progress(1)

    def test_single_epoch_stays_at_start(self):
        assert curriculum(num_epochs=1).progress(5) == 0.0


class TestCurriculumAt:

    def test_min_max_interpolation(self):
        c = curriculum()
        assert (c.at(0).min_, c.at(0).max_) == (80.0, 100.0)
        assert 35.0 < c.at(4).min_ < 45.0 and 55.0 < c.at(4).max_ < 65.0
        assert (c.at(9).min_, c.at(9).max_) == (0.0, 20.0)
        assert (c.at(None).min_, c.at(None).max_) == (0.0, 20.0)

    def test_constant_value_interpolation(self):
        c = curriculum(start=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
                       end=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT))
        assert c.at(0).value == 100.0
        assert c.at(9).value == 10.0

    def test_samples_in_expected_range(self):
        c = curriculum(end=uniform(5.0, 15.0))
        rng = np.random.default_rng(0)
        assert np.all((c.sample(100, rng, epoch=0) >= 80.0) & (c.sample(100, rng, epoch=0) <= 100.0))
        late = c.sample(100, rng, epoch=9)
        assert np.all((late >= 5.0) & (late <= 15.0))

    def test_sampling_changes_nothing(self):
        c = curriculum()
        rng = np.random.default_rng(0)
        for _ in range(50):
            c.sample(4, rng, epoch=3)
        assert c.at(0).min_ == 80.0

    def test_resolve(self):
        c = curriculum()
        assert gf.Curriculum.resolve(c, 0).min_ == 80.0
        plain = uniform(1.0, 2.0)
        assert gf.Curriculum.resolve(plain, 0) is plain


class TestCurriculumDatasetIntegration:

    def dataset(self, c, group="train", generators=1, **kwargs):
        def generator():
            return gf.WNBGenerator(
                scaling_method=gf.ScalingMethod(c, gf.ScalingTypes.SNR),
                duration_seconds=gf.Distribution(min_=0.1, max_=0.3, type_=gf.DistributionType.UNIFORM),
                min_frequency_hertz=gf.Distribution(min_=40.0, max_=60.0, type_=gf.DistributionType.UNIFORM),
                max_frequency_hertz=gf.Distribution(min_=100.0, max_=150.0, type_=gf.DistributionType.UNIFORM),
            )
        return gf.Dataset(
            seed=1, group=group,
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
            waveform_generators=[generator() for _ in range(generators)], sample_rate_hertz=512.0,
            offsource_duration_seconds=4.0,
            num_examples_per_batch=8, steps_per_epoch=2,
            input_variables=[gf.ReturnVariables.ONSOURCE], output_variables=[gf.ScalingTypes.SNR],
            **kwargs,
        )

    def test_training_follows_the_epoch(self):
        c = curriculum(start=uniform(80.0, 100.0), end=uniform(5.0, 15.0), num_epochs=3)
        ds = self.dataset(c)
        early = np.asarray(ds[0][1]["SNR"])
        assert np.all((early >= 80.0) & (early <= 100.0))
        ds.on_epoch_end()
        ds.on_epoch_end()
        late = np.asarray(ds[0][1]["SNR"])
        assert np.all((late >= 5.0) & (late <= 15.0))

    def test_evaluation_uses_the_end_distribution(self):
        c = curriculum(start=uniform(80.0, 100.0), end=uniform(5.0, 15.0), num_epochs=3)
        val = self.dataset(c, group="validate")
        targets = np.asarray(val[0][1]["SNR"])
        assert np.all((targets >= 5.0) & (targets <= 15.0))

    def test_validation_does_not_move_training(self):
        # B37: validation sampled the shared curriculum and advanced it.
        c = curriculum(num_epochs=3)
        train, val = self.dataset(c), self.dataset(c, group="validate")
        reference = np.asarray(train[1][1]["SNR"])
        for k in range(2):
            val[k]
        np.testing.assert_array_equal(np.asarray(train[1][1]["SNR"]), reference)

    def test_a_shared_curriculum_is_read_at_the_epoch_by_every_generator(self):
        # A Curriculum shared by several generators advanced once per
        # generator per batch.
        c = curriculum(start=uniform(80.0, 100.0), end=uniform(5.0, 15.0), num_epochs=3)
        ds = self.dataset(c, generators=2)
        first = np.asarray(ds[0][1]["SNR"])
        assert first.shape == (8, 2) and np.all((first >= 80.0) & (first <= 100.0))
        np.testing.assert_array_equal(np.asarray(ds[0][1]["SNR"]), first)
        ds.on_epoch_end()
        middle = np.asarray(ds[1][1]["SNR"])
        assert np.all((middle >= 42.5) & (middle <= 57.5))

    def test_draws_do_not_move_the_schedule(self):
        # Progress counted sample() calls, so Keras' build peek and shape
        # probes advanced it.
        c = curriculum(num_epochs=3)
        probed, fresh = self.dataset(c), self.dataset(c)
        for _ in range(10):
            probed[0], probed[1]
        np.testing.assert_array_equal(np.asarray(probed[1][1]["SNR"]), np.asarray(fresh[1][1]["SNR"]))
        assert probed.epoch == 0

    def test_resumed_run_sets_the_epoch(self):
        c = curriculum(start=uniform(80.0, 100.0), end=uniform(5.0, 15.0), num_epochs=3)
        ds = self.dataset(c)
        ds.epoch = 2
        late = np.asarray(ds[0][1]["SNR"])
        assert np.all((late >= 5.0) & (late <= 15.0))


class TestCurriculumProgressCallback:

    def test_prints_the_epoch_range(self, capsys):
        callback = gf.CurriculumProgressCallback(curriculum(), name="SNR")
        callback.on_train_begin()
        callback.on_epoch_begin(9)
        out = capsys.readouterr().out
        assert "Epoch 10: SNR range [0.0, 20.0] (progress: 100.0%)" in out

    def test_quiet(self, capsys):
        callback = gf.CurriculumProgressCallback(curriculum(), verbose=False)
        callback.on_epoch_begin(0)
        assert capsys.readouterr().out == ""
