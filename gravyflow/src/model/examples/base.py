"""
The shared interface and data contract of the example models.

Every example model takes ``WHITENED_ONSOURCE``, shaped (detectors, samples)
per example: whitening is done in the data path, not inside the model. Every
classifier returns logits (no softmax): the validator ranks logits, because
saturated probabilities tie.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import keras
import numpy as np

import gravyflow as gf
from gravyflow.src.resources import weights

# The batch contract every example model is built on.
MODEL_INPUT = gf.ReturnVariables.WHITENED_ONSOURCE


def onsource_samples(sample_rate_hertz: float, onsource_duration_seconds: float) -> int:
    return int(round(sample_rate_hertz * onsource_duration_seconds))


def model_input(sample_rate_hertz: float, onsource_duration_seconds: float, num_ifos: int = 1):
    """The Keras input of an example model: (detectors, samples), float32."""
    return keras.Input(
        shape=(num_ifos, onsource_samples(sample_rate_hertz, onsource_duration_seconds)),
        name=MODEL_INPUT.name,
        dtype="float32",
    )


def cbc_dataset_args(
    *,
    sample_rate_hertz: float,
    onsource_duration_seconds: float,
    offsource_duration_seconds: float,
    mass_distribution_type: gf.DistributionType,
    mass_range_msun: Sequence[float],
    snr: "gf.Distribution | gf.Curriculum",
    injection_chance: float,
    batch_size: int,
    output_variables: list,
    noise_obtainer: Optional[gf.NoiseObtainer] = None,
    **kwargs,
) -> dict:
    """
    Arguments of a dataset of non-spinning BBH injections (IMRPhenomD) in noise
    (coloured L1 noise by default), returning the example models' input contract.

    Inclination is uniform in [0, π]: the library has no uniform-in-cos(ι)
    prior, which the papers use.
    """
    lo, hi = mass_range_msun
    masses = gf.Distribution(min_=lo, max_=hi, type_=mass_distribution_type)
    return {
        "sample_rate_hertz": sample_rate_hertz,
        "onsource_duration_seconds": onsource_duration_seconds,
        "offsource_duration_seconds": offsource_duration_seconds,
        "noise_obtainer": noise_obtainer or gf.NoiseObtainer(
            noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.L1]),
        "waveform_generators": gf.CBCGenerator(
            mass_1_msun=masses,
            mass_2_msun=masses,
            inclination_radians=gf.Distribution(min_=0.0, max_=np.pi, type_=gf.DistributionType.UNIFORM),
            scaling_method=gf.ScalingMethod(value=snr, type_=gf.ScalingTypes.SNR),
            injection_chance=injection_chance,
            spin_1_in=(0.0, 0.0, 0.0),
            spin_2_in=(0.0, 0.0, 0.0),
        ),
        "num_examples_per_batch": batch_size,
        "input_variables": [MODEL_INPUT],
        "output_variables": output_variables,
        **kwargs,
    }


class ExampleModel(ABC):
    """
    Abstract base class of the example GW detection models.

    Subclasses define ``config`` (a configuration dataclass), ``weights_name``
    (the file of their shipped weights in ``res/model``) and the methods below.
    """

    config = None
    weights_name: Optional[str] = None

    @classmethod
    @abstractmethod
    def model(cls, config=None, num_ifos: int = 1, **kwargs) -> keras.Model:
        """The uncompiled network: ``WHITENED_ONSOURCE`` in, logits out."""

    @classmethod
    @abstractmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config=None,
        seed: Optional[int] = None,
        group: str = None,
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """A dataset in the model's input contract, with ``INJECTION_PRESENT`` labels."""

    @classmethod
    @abstractmethod
    def compile_model(cls, model: keras.Model, **kwargs) -> keras.Model:
        """Compile with the model's optimiser and a loss that takes logits."""

    @classmethod
    def pretrained(cls, weights_path: Optional[Path] = None) -> keras.Model:
        """
        Load trained weights, by default the example's own (``gravyflow.src.resources.weights``:
        shipped in the package, or a release download that ``gravyflow.fetch`` put in the store).

        Raises:
            FileNotFoundError: no weights at the path.
            ValueError: the weights were trained for another input contract (a
                model that whitens inside itself, or returns probabilities);
                retrain them with the example's training script.
        """
        if weights_path is None:
            if cls.weights_name is None:
                raise FileNotFoundError(f"{cls.__name__} has no trained weights.")
            weights_path = weights(cls.weights_name)
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"No trained weights at {weights_path}.")
        model = keras.models.load_model(str(weights_path))
        inputs = [t.name for t in model.inputs]
        activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", None)
        if inputs != [MODEL_INPUT.name] or activation in ("softmax", "sigmoid"):
            raise ValueError(
                f"The weights at {weights_path} were trained for an older input contract "
                f"(inputs {inputs}, output activation {activation}); the example models take "
                f"{MODEL_INPUT.name} and return logits. Retrain with the example's training script."
            )
        return model


@dataclass
class ValidationDatasetConfig:
    """
    The common validation data of the example models, so that they are
    compared on one benchmark. The validator sets the SNR distribution and the
    seed itself; this sets everything else.
    """
    sample_rate_hertz: float = 2048.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    ifos: List[gf.IFO] = field(default_factory=lambda: [gf.IFO.L1])
    mass_range_msun: Sequence[float] = (5.0, 95.0)
    snr_min: float = 8.0
    snr_max: float = 20.0
    batch_size: int = 32


def validation_dataset_args(
    noise_type: gf.NoiseType = gf.NoiseType.COLORED,
    config: Optional[ValidationDatasetConfig] = None,
    seed: Optional[int] = None,
    **kwargs
) -> dict:
    """
    Dataset arguments for validating an example model (`gf.Validator`).

    ``seed`` matters only when the arguments build a dataset directly: the
    validator replaces it with seeds of its own. The arguments build only a
    held-out split: ``test`` unless ``group`` names another, never ``train``.
    """
    cfg = config or ValidationDatasetConfig()
    group = gf.SplitSpec.check_name(kwargs.pop("group", "test"))
    if group == "train":
        raise ValueError("Validation arguments build a held-out split, never 'train'.")
    ifo_data_obtainer = None
    if noise_type == gf.NoiseType.REAL:
        ifo_data_obtainer = gf.IFODataObtainer(
            observing_runs=[gf.ObservingRun.O3],
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.NOISE]
        )
    return cbc_dataset_args(
        sample_rate_hertz=cfg.sample_rate_hertz,
        onsource_duration_seconds=cfg.onsource_duration_seconds,
        offsource_duration_seconds=cfg.offsource_duration_seconds,
        mass_distribution_type=gf.DistributionType.UNIFORM,
        mass_range_msun=cfg.mass_range_msun,
        snr=gf.Distribution(min_=cfg.snr_min, max_=cfg.snr_max, type_=gf.DistributionType.UNIFORM),
        injection_chance=0.5,
        batch_size=cfg.batch_size,
        output_variables=[gf.ReturnVariables.INJECTION_PRESENT],
        noise_obtainer=gf.NoiseObtainer(
            noise_type=noise_type, ifos=cfg.ifos, ifo_data_obtainer=ifo_data_obtainer),
        seed=seed,
        group=group,
        **kwargs,
    )
