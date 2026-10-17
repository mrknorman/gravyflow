"""
The dataset: batches of white noise, with and without injections, their processing,
and the refusal of configurations that can serve nothing.

Real-noise datasets are served offline in ``test_public_sources`` (GWOSC's timeline,
Gravity Spy and strain faked) and in ``test_transient_mode`` (real events).
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.dataset import GravyflowDataset

RV = gf.ReturnVariables
INJECTIONS = gf.tests.PATH / "example_injection_parameters"
RATE = 1024.0


def white_dataset(seed=42, batch=4, generators=None, ifos=(gf.IFO.L1,), inputs=(RV.ONSOURCE,),
                  outputs=(RV.OFFSOURCE,), **kwargs):
    arguments = dict(
        seed=seed, sample_rate_hertz=RATE, onsource_duration_seconds=1.0,
        offsource_duration_seconds=2.0,  # three Welch segments: the PSD minimum
        crop_duration_seconds=0.5, num_examples_per_batch=batch,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=list(ifos)),
        waveform_generators=generators, input_variables=list(inputs), output_variables=list(outputs),
        steps_per_epoch=10, group="train",
    )
    arguments.update(kwargs)
    return GravyflowDataset(**arguments)


def test_an_epoch_is_steps_per_epoch_batches():
    assert len(white_dataset(steps_per_epoch=7)) == 7


def test_white_noise_batches_follow_their_seed():
    """Shapes are the cropped onsource and the offsource; the seed alone fixes the draw."""
    inputs, outputs = white_dataset()[0]
    onsource, offsource = (np.asarray(inputs[RV.ONSOURCE.name]), np.asarray(outputs[RV.OFFSOURCE.name]))
    # The onsource is cropped to 1.0 s (1024 samples); the offsource is 2.0 s.
    assert onsource.shape == (4, 1, 1024) and offsource.shape == (4, 1, 2048)
    assert np.all(np.isfinite(onsource)) and len({row.tobytes() for row in onsource}) == 4
    again, _ = white_dataset()[0]
    other, _ = white_dataset(seed=43)[0]
    np.testing.assert_array_equal(np.asarray(again[RV.ONSOURCE.name]), onsource)
    assert not np.array_equal(np.asarray(other[RV.ONSOURCE.name]), onsource)


def test_an_injection_is_added_to_the_noise_drawn_without_it():
    """The noise stream is the seed's whatever is injected: onsource less injection is that noise."""
    wnb = gf.WNBGenerator(
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        min_frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        max_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
        scaling_method=gf.ScalingMethod(value=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
                                        type_=gf.ScalingTypes.SNR),
    )
    inputs, _ = white_dataset(batch=2, generators=[wnb], inputs=(RV.ONSOURCE, RV.INJECTIONS))[0]
    noise, _ = white_dataset(batch=2)[0]
    injections = np.asarray(inputs[RV.INJECTIONS.name])
    # (batch, generators, detectors, onsource samples)
    assert injections.shape == (2, 1, 1, 1024) and np.all(np.any(injections != 0, axis=-1))
    signal = np.asarray(inputs[RV.ONSOURCE.name]) - np.asarray(noise[RV.ONSOURCE.name])
    # One float32 addition per sample: half an ulp of the sum's scale.
    scale = np.abs(np.asarray(inputs[RV.ONSOURCE.name])).max()
    np.testing.assert_allclose(signal, injections[:, 0], rtol=0, atol=np.finfo(np.float32).eps * scale)


@pytest.mark.parametrize("name, ifos", [
    ("phenom_d_parameters.json", [gf.IFO.L1]),
    ("wnb_parameters.json", [gf.IFO.L1, gf.IFO.H1]),
])
def test_example_generators_inject_into_every_example(name, ifos):
    """The example configurations, every example injected: a mask, a signal and finite whitened data."""
    generator = gf.WaveformGenerator.load(
        path=INJECTIONS / name, network=ifos,
        scaling_method=gf.ScalingMethod(value=gf.Distribution(min_=8.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
                                        type_=gf.ScalingTypes.SNR),
    )
    generator.injection_chance = 1.0
    inputs, _ = white_dataset(batch=4, generators=[generator], ifos=ifos,
                              inputs=(RV.WHITENED_ONSOURCE, RV.INJECTION_MASKS, RV.INJECTIONS))[0]
    np.testing.assert_array_equal(np.asarray(inputs[RV.INJECTION_MASKS.name]), np.ones((4, 1)))  # (batch, generators)
    injections = np.asarray(inputs[RV.INJECTIONS.name])
    assert injections.shape[:3] == (4, 1, len(ifos)) and np.all(np.any(injections != 0, axis=-1))
    whitened = np.asarray(inputs[RV.WHITENED_ONSOURCE.name])
    assert whitened.shape == (4, len(ifos), 1024) and np.all(np.isfinite(whitened))


def test_processing_whitens_and_makes_a_spectrogram():
    inputs, _ = white_dataset(batch=2, inputs=(RV.WHITENED_ONSOURCE, RV.SPECTROGRAM_ONSOURCE), outputs=())[0]
    whitened = np.asarray(inputs[RV.WHITENED_ONSOURCE.name])
    spectrogram = np.asarray(inputs[RV.SPECTROGRAM_ONSOURCE.name])
    assert whitened.shape == (2, 1, 1024) and np.all(np.isfinite(whitened))
    assert spectrogram.shape[:2] == (2, 1) and np.all(np.isfinite(spectrogram)) and np.all(spectrogram >= 0)


def test_dataset_empty_segment_list():
    """
    REAL noise with no readable time is refused by name, offline, rather than hang or
    serve nothing: the timeline (``get_all_segment_times``) holds nothing.
    """

    class EmptySegmentsObtainer(gf.NoiseDataObtainer):
        """A noise obtainer whose timeline holds no data."""

        def get_all_segment_times(self, ifo):
            return np.empty((0, 2))

        def remove_unwanted_segments(self, ifo, valid_segments):
            return valid_segments  # nothing to veto: no catalogue is read

    ifo_obtainer = EmptySegmentsObtainer(
        observing_runs=gf.ObservingRun.O3,
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
    )
    noise_obtainer = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=[gf.IFO.L1]
    )
    # Refused as the dataset is built, when its availability is frozen, or at its first batch.
    with pytest.raises(ValueError, match="no valid segments|no row long enough"):
        GravyflowDataset(
            noise_obtainer=noise_obtainer,
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.OFFSOURCE],
            steps_per_epoch=1,
            group="train"
        )[0]


def test_a_dataset_with_no_return_variables_is_refused():
    with pytest.raises(ValueError, match="No return variables requested"):
        GravyflowDataset(group="train")


@pytest.mark.parametrize("name, value", [
    ("sample_rate_hertz", -100.0),
    ("onsource_duration_seconds", -1.0),
    ("num_examples_per_batch", 0),
])
def test_a_configuration_that_can_serve_nothing_is_refused_by_name(name, value):
    """A non-positive rate, duration or batch size is refused as the dataset is built, naming it."""
    with pytest.raises(ValueError, match=name):
        white_dataset(**{name: value})
