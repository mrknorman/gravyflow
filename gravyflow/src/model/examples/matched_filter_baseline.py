"""
Matched-filter baseline model.

A CBC template-bank matched filter wrapped as an example model, so that it can
be scored on exactly the data the neural-network examples see, by the same
validator. It takes the whitened onsource window (the example models' input)
and the offsource the dataset whitened it with, whitens its templates with the
same filter, and returns two logits whose difference is
(network SNR - snr_threshold) / temperature: the validator ranks by the SNR.

The baseline is a reference for comparison, so it is configured to see what
the models see and no more: the dataset's window, crop, band and PSD
estimator, a bank of non-spinning, 22-mode templates over the injection's mass
range with a minimal match of 0.97, and the merger times the dataset's
placement can produce. It has no signal-consistency veto, so on real noise
with glitches it is penalised against a veto-equipped search.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import keras
from keras.models import Model

import gravyflow as gf
from gravyflow.src.model.examples.base import (
    ExampleModel, MODEL_INPUT, ValidationDatasetConfig, cbc_dataset_args, onsource_samples,
    validation_dataset_args as _validation_dataset_args,
)
from gravyflow.src.detection.matched_filter import MatchedFilterLayer

# The second input: the offsource the whitening PSD is estimated from.
BACKGROUND_INPUT = gf.ReturnVariables.OFFSOURCE


@dataclass
class MatchedFilterBaselineConfig:
    """
    Configuration of the matched-filter baseline and of its dataset.

    The window, crop, offsource and band must be the dataset's: the filter
    whitens its templates with the filter the dataset designed for them.
    """
    sample_rate_hertz: float = 2048.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    crop_duration_seconds: float = 0.5
    spectral_config: Optional[gf.SpectralConfig] = None
    ifos: List[gf.IFO] = field(default_factory=lambda: [gf.IFO.L1])

    # The bank: non-spinning IMRPhenomD over both component masses.
    mass_range_msun: Tuple[float, float] = (5.0, 95.0)
    minimal_match: float = 0.97
    # Merger times searched, in seconds from the window's start. None: where
    # the dataset's placement can put a merger (`search_window_seconds`).
    search_window_seconds: Optional[Tuple[float, float]] = None

    # The logits: (rho - snr_threshold) / temperature.
    snr_threshold: float = 8.0
    temperature: float = 2.0

    # Memory: examples and templates per step.
    batch_chunk: int = 16
    template_chunk: int = 8

    # The dataset (for detection with INJECTION_PRESENT labels).
    injection_chance: float = 0.5
    snr_min: float = 8.0
    snr_max: float = 20.0
    batch_size: int = 32


class MatchedFilterBaseline(ExampleModel):
    """
    The matched-filter baseline as an example model. No training: `compile_model`
    returns the model unchanged and `pretrained` builds it.

        >>> model = MatchedFilterBaseline.model()
        >>> dataset = MatchedFilterBaseline.dataset(seed=1)
        >>> logits = model.predict_on_batch(dataset[0][0])
    """

    config = MatchedFilterBaselineConfig()

    @classmethod
    def inputs(cls, config: Optional[MatchedFilterBaselineConfig] = None) -> list:
        """The model's input variables, in order."""
        return [MODEL_INPUT, BACKGROUND_INPUT]

    @classmethod
    def search_window_seconds(cls, dataset_args: dict) -> Tuple[float, float]:
        """
        The merger times a dataset's placement can produce in any of its
        detectors, (start, end) in seconds from the window's start: the union
        over its waveform generators of each one's placement range, widened by
        its detectors' delays from the geocentre. A noise-only dataset (no
        waveform generators) places nothing, so it is refused: the search
        window must then be set in the config.
        """
        window = float(dataset_args["onsource_duration_seconds"])
        generators = dataset_args.get("waveform_generators") or []
        if not isinstance(generators, (list, tuple)):
            generators = [generators]
        if not generators:
            raise ValueError(
                "The dataset has no waveform generators, so no placement range to search: set "
                "MatchedFilterBaselineConfig.search_window_seconds.")
        noise_network = gf.Network(list(dataset_args["noise_obtainer"].ifos))
        ranges = [g.reference_time_window_seconds(window, noise_network if g.network is None else g.network)
                  for g in generators]
        return min(lo for lo, _ in ranges), max(hi for _, hi in ranges)

    @classmethod
    def model(cls, config: Optional[MatchedFilterBaselineConfig] = None, num_ifos: Optional[int] = None,
              dataset_args: Optional[dict] = None, **kwargs) -> Model:
        """
        WHITENED_ONSOURCE (detectors, samples) and OFFSOURCE (detectors,
        samples) in, logits out. Unless the config sets the search window, it
        is the placement range of `dataset_args` (default the config's own
        dataset), the data the baseline will be scored on.
        """
        config = config or cls.config
        search_window = config.search_window_seconds
        if search_window is None:
            search_window = cls.search_window_seconds(dataset_args or cls.dataset_args(config=config))
        num_ifos = len(config.ifos) if num_ifos is None else num_ifos
        if num_ifos != len(config.ifos):
            raise ValueError(f"num_ifos ({num_ifos}) differs from the config's ifos ({len(config.ifos)}).")
        onsource = keras.Input(
            shape=(num_ifos, onsource_samples(config.sample_rate_hertz, config.onsource_duration_seconds)),
            name=MODEL_INPUT.name, dtype="float32")
        offsource = keras.Input(
            shape=(num_ifos, onsource_samples(config.sample_rate_hertz, config.offsource_duration_seconds)),
            name=BACKGROUND_INPUT.name, dtype="float32")
        layer = MatchedFilterLayer(
            mass_range=config.mass_range_msun,
            minimal_match=config.minimal_match,
            sample_rate_hertz=config.sample_rate_hertz,
            onsource_duration_seconds=config.onsource_duration_seconds,
            crop_duration_seconds=config.crop_duration_seconds,
            spectral_config=config.spectral_config,
            search_window_seconds=search_window,
            ifos=config.ifos,
            snr_threshold=config.snr_threshold,
            temperature=config.temperature,
            batch_chunk=config.batch_chunk,
            template_chunk=config.template_chunk,
            name="matched_filter",
        )
        outputs = layer({MODEL_INPUT.name: onsource, BACKGROUND_INPUT.name: offsource})
        return Model(inputs=[onsource, offsource], outputs=outputs, name="MatchedFilterBaseline")

    @classmethod
    def dataset_args(cls, noise_type: gf.NoiseType = gf.NoiseType.COLORED,
                     config: Optional[MatchedFilterBaselineConfig] = None, **kwargs) -> dict:
        """Arguments of a BBH dataset (masses uniform over the bank's range) in the model's input contract."""
        config = config or cls.config
        if noise_type != gf.NoiseType.COLORED:
            raise ValueError("The example dataset uses coloured noise; pass a noise_obtainer for other noise.")
        return cbc_dataset_args(
            sample_rate_hertz=config.sample_rate_hertz,
            onsource_duration_seconds=config.onsource_duration_seconds,
            offsource_duration_seconds=config.offsource_duration_seconds,
            mass_distribution_type=gf.DistributionType.UNIFORM,
            mass_range_msun=config.mass_range_msun,
            snr=gf.Distribution(min_=config.snr_min, max_=config.snr_max, type_=gf.DistributionType.UNIFORM),
            injection_chance=config.injection_chance,
            batch_size=config.batch_size,
            output_variables=[gf.ReturnVariables.INJECTION_PRESENT],
            noise_obtainer=gf.NoiseObtainer(noise_type=noise_type, ifos=list(config.ifos)),
            input_variables=cls.inputs(config),
            crop_duration_seconds=config.crop_duration_seconds,
            **({"spectral_config": config.spectral_config} if config.spectral_config is not None else {}),
            **kwargs,
        )

    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[MatchedFilterBaselineConfig] = None,
        seed: Optional[int] = None,
        group: str = None,
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """A dataset in the model's input contract, with ``INJECTION_PRESENT`` labels."""
        return gf.Dataset(**cls.dataset_args(noise_type, config, seed=seed, group=group,
                                             steps_per_epoch=steps_per_epoch, **kwargs))

    @classmethod
    def validation_dataset_args(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[ValidationDatasetConfig] = None,
        baseline: Optional[MatchedFilterBaselineConfig] = None,
        **kwargs
    ) -> dict:
        """
        The example models' common validation data (`validation_dataset_args`),
        with the offsource as a second input. The baseline's window must match.
        """
        cfg = config or ValidationDatasetConfig()
        baseline = baseline or cls.config
        mismatched = [
            name for name, a, b in [
                ("sample_rate_hertz", cfg.sample_rate_hertz, baseline.sample_rate_hertz),
                ("onsource_duration_seconds", cfg.onsource_duration_seconds, baseline.onsource_duration_seconds),
                ("offsource_duration_seconds", cfg.offsource_duration_seconds, baseline.offsource_duration_seconds),
            ] if float(a) != float(b)
        ]
        if mismatched:
            raise ValueError(f"The validation data and the baseline differ in {mismatched}.")
        return _validation_dataset_args(
            noise_type=noise_type, config=cfg, input_variables=cls.inputs(baseline),
            crop_duration_seconds=baseline.crop_duration_seconds,
            **({"spectral_config": baseline.spectral_config} if baseline.spectral_config is not None else {}),
            **kwargs)

    @classmethod
    def compile_model(cls, model: Model, **kwargs) -> Model:
        """No-op: the matched filter has nothing to train."""
        return model

    @classmethod
    def pretrained(cls, weights_path=None) -> Model:
        """The model itself: its 'weights' are the template bank."""
        return cls.model()
