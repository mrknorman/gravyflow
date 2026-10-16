"""
Gabbard et al. 2017: a binary black hole detection CNN.

Paper: "Matching matched filtering with deep networks for gravitational-wave astronomy"
Authors: H. Gabbard, M. Williams, F. Hayes, C. Messenger
arXiv: 1712.06041v2
DOI: 10.1103/PhysRevLett.120.141103

The example follows the paper as closely as the library allows:

- **Network (Table I):** six ELU convolutional layers, max-pooling after layers
  2, 4 and 6 (sizes 8, 6, 4), two 64-unit ELU dense layers with 50% dropout,
  and two output neurons. They return logits; the paper's softmax is applied
  by the loss (and by whoever wants probabilities), not by the network.
- **Training:** Nadam, learning rate 0.002, β1 = 0.9, β2 = 0.999, ε = 1e-8, on
  the two-class cross-entropy, as in the paper.
- **Data:** 1 s of whitened strain from one detector at 8192 Hz, BBH masses
  5–95 M☉ with a log prior, non-spinning IMRPhenomD, half the examples noise
  only. The model takes ``WHITENED_ONSOURCE``: whitening is in the data path.

Where it departs from the paper:

- inclination is uniform in [0, π], not uniform in cos ι (no such prior yet);
- the merger is placed by the library's injection placement, not uniformly in
  the [0.75, 0.95] fraction of the window;
- the CPU preset (`Gabbard2017Config.cpu`) samples at 2048 Hz.
"""

from dataclasses import dataclass, replace
from typing import Optional

import keras
from keras.layers import Conv1D, Dense, Dropout, ELU, Flatten, MaxPooling1D, Permute
from keras.models import Model

import gravyflow as gf
from gravyflow.src.model.examples.base import ExampleModel, cbc_dataset_args, model_input


@dataclass(frozen=True)
class Gabbard2017Config:
    """The paper's data and network settings (Table I and "Simulation details")."""
    # Sampling
    sample_rate_hertz: float = 8192.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0

    # Component masses 5-95 Msun, log prior
    mass_1_min_msun: float = 5.0
    mass_1_max_msun: float = 95.0

    # Optimal SNR of the injections in training
    snr_min: float = 8.0
    snr_max: float = 15.0

    injection_chance: float = 0.5  # half signal+noise, half noise only
    batch_size: int = 32

    # Table I: (filters, kernel, max-pool size or None) per convolutional layer
    conv_configs: tuple = (
        (8, 64, None),
        (8, 32, 8),
        (16, 32, None),
        (16, 16, 6),
        (32, 16, None),
        (32, 16, 4),
    )
    dense_units: int = 64
    dropout_rate: float = 0.5
    num_output_classes: int = 2

    # Nadam, as in the paper
    learning_rate: float = 0.002

    @classmethod
    def cpu(cls) -> "Gabbard2017Config":
        """The paper's settings at 2048 Hz: a network of 41,842 weights that trains on a CPU."""
        return replace(cls(), sample_rate_hertz=2048.0)


class Gabbard2017(ExampleModel):
    """
    Gabbard et al. 2017 binary black hole detection model.

    Usage:
        >>> from gravyflow.src.model.examples import Gabbard2017, Gabbard2017Config
        >>> config = Gabbard2017Config.cpu()
        >>> model = Gabbard2017.compile_model(Gabbard2017.model(config), config)
        >>> dataset = Gabbard2017.dataset(config=config, seed=1, group="train")

    The training script, ``train_gabbard_2017``, runs the whole recipe.
    """

    config = Gabbard2017Config()
    weights_name = "gabbard_2017.keras"

    @classmethod
    def model(cls, config: Optional[Gabbard2017Config] = None, num_ifos: int = 1) -> keras.Model:
        """The Table I network: ``WHITENED_ONSOURCE`` (detectors, samples) in, two logits out."""
        cfg = config or cls.config
        inputs = model_input(cfg.sample_rate_hertz, cfg.onsource_duration_seconds, num_ifos)
        x = Permute((2, 1), name="permute")(inputs)  # (samples, detectors) for Conv1D
        for number, (filters, kernel_size, pool_size) in enumerate(cfg.conv_configs, start=1):
            x = Conv1D(filters, kernel_size, padding="valid", name=f"Conv1D_{number}")(x)
            x = ELU(name=f"ELU_{number}")(x)
            if pool_size is not None:
                x = MaxPooling1D(pool_size=pool_size, strides=pool_size, padding="same",
                                 name=f"MaxPool_{number}")(x)
        x = Flatten(name="Flatten")(x)
        for number in (7, 8):
            x = Dense(cfg.dense_units, name=f"Dense_{number}")(x)
            x = ELU(name=f"ELU_{number}")(x)
            x = Dropout(cfg.dropout_rate, name=f"Dropout_{number}")(x)
        logits = Dense(cfg.num_output_classes, name="output")(x)
        return Model(inputs={inputs.name: inputs}, outputs=logits, name="Gabbard2017")

    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[Gabbard2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Training or evaluation data in the paper's configuration, with
        ``INJECTION_PRESENT`` labels, in coloured L1 noise or, with
        ``noise_type=NoiseType.REAL``, public GWOSC L1 noise.
        """
        cfg = config or cls.config
        return gf.GravyflowDataset(
            **cbc_dataset_args(
                noise_obtainer=gf.NoiseObtainer(noise_type=noise_type, ifos=[gf.IFO.L1]),
                sample_rate_hertz=cfg.sample_rate_hertz,
                onsource_duration_seconds=cfg.onsource_duration_seconds,
                offsource_duration_seconds=cfg.offsource_duration_seconds,
                mass_distribution_type=gf.DistributionType.LOG,
                mass_range_msun=(cfg.mass_1_min_msun, cfg.mass_1_max_msun),
                snr=gf.Distribution(min_=cfg.snr_min, max_=cfg.snr_max, type_=gf.DistributionType.UNIFORM),
                injection_chance=cfg.injection_chance,
                batch_size=cfg.batch_size,
                output_variables=[gf.ReturnVariables.INJECTION_PRESENT],
            ),
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

    @classmethod
    def compile_model(cls, model: keras.Model, config: Optional[Gabbard2017Config] = None) -> keras.Model:
        """Nadam (lr 0.002, β1 0.9, β2 0.999, ε 1e-8) on the two-class cross-entropy of logits."""
        cfg = config or cls.config
        model.compile(
            optimizer=keras.optimizers.Nadam(
                learning_rate=cfg.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
        )
        return model
