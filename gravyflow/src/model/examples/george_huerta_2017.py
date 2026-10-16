"""
George & Huerta 2017: Deep Filtering for gravitational-wave detection.

Paper: "Deep Neural Networks to Enable Real-time Multimessenger Astrophysics"
Authors: Daniel George, E. A. Huerta
arXiv: 1701.00008v3
Published: Physical Review D 97, 044039 (2018)

Two networks from the paper:

1. `GeorgeHuerta2017Shallow`: three dilated convolutional layers (Fig. 5);
2. `GeorgeHuerta2017Deep`: four dilated convolutional layers (Fig. 6).

Each is a classifier (two logits: noise, signal) or a regressor of the two
component masses (the paper's predictor). The paper trains the predictor first
and transfers its weights to the classifier; ``train_george_huerta_2017``
does that.

The data follow the paper where the library allows: 1 s of whitened strain at
8192 Hz from one detector (``WHITENED_ONSOURCE``: whitening is in the data
path), non-spinning BBHs with masses 5–75 M☉, and the paper's training scheme
of SNR falling from 80–100 to 5–15 (`gf.Curriculum`). The waveforms are
IMRPhenomD, not the paper's EOB, and inclination is uniform in [0, π].
"""

from dataclasses import dataclass, replace
from typing import Optional

import keras
from keras.layers import Activation, Conv1D, Dense, Flatten, MaxPooling1D, Permute
from keras.models import Model

import gravyflow as gf
from gravyflow.src.model.examples.base import ExampleModel, cbc_dataset_args, model_input


@dataclass(frozen=True)
class GeorgeHuerta2017Config:
    """The paper's settings (Section III and Figs. 5 and 6)."""
    # Sampling
    sample_rate_hertz: float = 8192.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0

    # Component masses 5-75 Msun
    mass_1_min_msun: float = 5.0
    mass_1_max_msun: float = 75.0

    # The target SNR range; with the curriculum, training starts at
    # 80-100 and falls linearly to it over snr_curriculum_num_epochs.
    snr_min: float = 5.0
    snr_max: float = 15.0
    use_curriculum: bool = True
    snr_curriculum_start_min: float = 80.0
    snr_curriculum_start_max: float = 100.0
    # Set from the run's length: see `curriculum_epochs`.
    snr_curriculum_num_epochs: int = 10
    snr_curriculum_verbose: bool = False

    injection_chance: float = 0.5
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Fig. 5: (filters, kernel, dilation), max-pool 4 after each; dense 64
    shallow_conv_configs: tuple = ((16, 16, 1), (32, 8, 4), (64, 8, 4))
    shallow_fc_units: tuple = (64,)
    # Fig. 6: (filters, kernel, dilation), max-pool 4 after each; dense 128, 64
    deep_conv_configs: tuple = ((64, 16, 1), (128, 16, 2), (256, 16, 2), (512, 32, 2))
    deep_fc_units: tuple = (128, 64)
    pool_size: int = 4

    num_output_classes: int = 2

    @classmethod
    def cpu(cls) -> "GeorgeHuerta2017Config":
        """
        The paper's settings at 2048 Hz, for the shallow network. The deep
        network's receptive field needs the paper's 8192 samples.
        """
        return replace(cls(), sample_rate_hertz=2048.0)

    def snr(self, curriculum: bool, first_step: int = 0) -> "gf.Distribution | gf.Curriculum":
        """
        The SNR distribution: the curriculum for training, the target range
        otherwise. ``first_step`` is the training step the curriculum starts
        at, so that a resumed run continues it rather than starting it again.
        """
        target = gf.Distribution(min_=self.snr_min, max_=self.snr_max, type_=gf.DistributionType.UNIFORM)
        if not curriculum:
            return target
        return gf.Curriculum(
            start=gf.Distribution(min_=self.snr_curriculum_start_min, max_=self.snr_curriculum_start_max,
                                  type_=gf.DistributionType.UNIFORM),
            end=target,
            num_epochs=self.snr_curriculum_num_epochs,
            schedule=gf.CurriculumSchedule.LINEAR,
            verbose=self.snr_curriculum_verbose,
            name="SNR",
            _current_step=first_step,
        )


def curriculum_epochs(epochs: int) -> int:
    """
    Curriculum length for a run of ``epochs``: the first half, so that the
    second half trains at the target SNR range the model is validated on.

    Raises:
        ValueError: fewer than 4 epochs leave no room for a curriculum.
    """
    if epochs < 4:
        raise ValueError(f"A curriculum needs a run of at least 4 epochs, got {epochs}; "
                         "train without the curriculum instead.")
    return epochs // 2


class _GeorgeHuerta2017(ExampleModel):
    """The two networks' shared data and training; subclasses give the layers."""

    config = GeorgeHuerta2017Config()
    _conv_attr: str
    _fc_attr: str
    _name: str

    @classmethod
    def model(
        cls,
        config: Optional[GeorgeHuerta2017Config] = None,
        num_ifos: int = 1,
        task: str = "classification",
    ) -> keras.Model:
        """
        ``WHITENED_ONSOURCE`` in; two logits (``task="classification"``) or two
        component masses (``task="regression"``) out.
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}.")
        cfg = config or cls.config
        inputs = model_input(cfg.sample_rate_hertz, cfg.onsource_duration_seconds, num_ifos)
        x = Permute((2, 1), name="permute")(inputs)  # (samples, detectors) for Conv1D
        for number, (filters, kernel_size, dilation) in enumerate(getattr(cfg, cls._conv_attr), start=1):
            x = Conv1D(filters, kernel_size, dilation_rate=dilation, name=f"Conv1D_{number}")(x)
            x = MaxPooling1D(pool_size=cfg.pool_size, strides=cfg.pool_size, name=f"MaxPool_{number}")(x)
            x = Activation("relu", name=f"ReLU_{number}")(x)
        x = Flatten(name="Flatten")(x)
        for number, units in enumerate(getattr(cfg, cls._fc_attr), start=1):
            x = Dense(units, name=f"Dense_{number}")(x)
            x = Activation("relu", name=f"ReLU_FC_{number}")(x)
        outputs = Dense(cfg.num_output_classes if task == "classification" else 2, name="output")(x)
        return Model(inputs={inputs.name: inputs}, outputs=outputs, name=cls._name)

    @classmethod
    def _dataset(cls, config, noise_type, seed, group, steps_per_epoch, regression: bool, curriculum: bool,
                 first_epoch: int = 0, **kwargs):
        cfg = config or cls.config
        labels = ([gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN]
                  if regression else [gf.ReturnVariables.INJECTION_MASKS])
        return gf.GravyflowDataset(
            **cbc_dataset_args(
                noise_obtainer=gf.NoiseObtainer(noise_type=noise_type, ifos=[gf.IFO.L1]),
                sample_rate_hertz=cfg.sample_rate_hertz,
                onsource_duration_seconds=cfg.onsource_duration_seconds,
                offsource_duration_seconds=cfg.offsource_duration_seconds,
                mass_distribution_type=gf.DistributionType.UNIFORM,
                mass_range_msun=(cfg.mass_1_min_msun, cfg.mass_1_max_msun),
                snr=cfg.snr(curriculum, first_step=first_epoch * steps_per_epoch),
                injection_chance=1.0 if regression else cfg.injection_chance,
                batch_size=cfg.batch_size,
                output_variables=labels,
            ),
            seed=seed,
            group=group,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[GeorgeHuerta2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 1000,
        curriculum: bool = False,
        first_epoch: int = 0,
        **kwargs
    ) -> gf.GravyflowDataset:
        """
        Detection data with ``INJECTION_MASKS`` labels; with ``curriculum``
        (and ``config.use_curriculum``), the SNR follows the training scheme
        from ``first_epoch``. Coloured L1 noise, or public GWOSC L1 noise with
        ``noise_type=NoiseType.REAL``.
        """
        cfg = config or cls.config
        return cls._dataset(cfg, noise_type, seed, group, steps_per_epoch, regression=False,
                            curriculum=curriculum and cfg.use_curriculum, first_epoch=first_epoch, **kwargs)

    @classmethod
    def dataset_regression(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[GeorgeHuerta2017Config] = None,
        seed: Optional[int] = None,
        group: str = "train",
        steps_per_epoch: int = 1000,
        curriculum: bool = False,
        first_epoch: int = 0,
        **kwargs
    ) -> gf.GravyflowDataset:
        """Every example injected, with ``MASS_1_MSUN`` and ``MASS_2_MSUN`` labels."""
        cfg = config or cls.config
        return cls._dataset(cfg, noise_type, seed, group, steps_per_epoch, regression=True,
                            curriculum=curriculum and cfg.use_curriculum, first_epoch=first_epoch, **kwargs)

    @classmethod
    def compile_model(
        cls,
        model: keras.Model,
        config: Optional[GeorgeHuerta2017Config] = None,
        task: str = "classification",
    ) -> keras.Model:
        """Adam; cross-entropy of logits (classification) or mean absolute error (regression)."""
        cfg = config or cls.config
        optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        if task == "classification":
            model.compile(optimizer=optimizer,
                          loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])
        else:
            model.compile(optimizer=optimizer, loss="mae", metrics=["mse"])
        return model

    @staticmethod
    def transfer_weights(source: keras.Model, target: keras.Model) -> int:
        """
        Copy every weighted layer but the output from ``source`` (the mass
        predictor) to ``target`` (the classifier), by layer name. Returns the
        number of layers copied.

        Raises:
            ValueError: the two networks' layers differ.
        """
        copied = 0
        for layer in source.layers:
            if layer.name == "output" or not layer.weights:
                continue
            target.get_layer(layer.name).set_weights(layer.get_weights())
            copied += 1
        return copied


class GeorgeHuerta2017Shallow(_GeorgeHuerta2017):
    """The shallow network (Fig. 5): three dilated convolutions, one 64-unit dense layer."""
    _conv_attr, _fc_attr, _name = "shallow_conv_configs", "shallow_fc_units", "GeorgeHuerta2017Shallow"
    weights_name = "georgehuerta2017shallow_classification.keras"


class GeorgeHuerta2017Deep(_GeorgeHuerta2017):
    """The deep network (Fig. 6): four dilated convolutions, dense layers of 128 and 64."""
    _conv_attr, _fc_attr, _name = "deep_conv_configs", "deep_fc_units", "GeorgeHuerta2017Deep"
    weights_name = "georgehuerta2017deep_classification.keras"
