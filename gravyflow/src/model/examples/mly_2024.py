"""
MLy: real-time detection of unmodelled gravitational-wave bursts.

Paper: "Toward real-time detection of unmodeled gravitational wave transients
       using convolutional neural networks"
Authors: Vasileios Skliris, Michael R. K. Norman, Patrick J. Sutton
Published: Physical Review D 110, 104034 (2024); arXiv:2009.14611

Two networks, whose scores the pipeline multiplies:

1. `MLyCoincidence` (model 1): three residual blocks over the whitened strain
   of every detector; it finds signals present in more than one detector.
2. `MLyCoherence` (model 2): a strain branch and a branch over the Pearson
   correlation of each detector pair at each lag; it finds signals that are
   correlated across detectors.

The layer layouts, optimisers and batch size follow the authors' training code
(github.com/VasSkliris/mly-methods-paper, ``virgo_rescaling/model{1,2}.py``);
the data follow Section III of the paper where the library allows: 1 s of
whitened strain at 1024 Hz from H1, L1 and V1, whitened with 16 s of
offsource, and white-noise bursts (WNBs) of 0.05-0.9 s, 40-480 Hz wide, scaled
to a network SNR (12-30 for model 1, 10-50 for model 2).

Every network takes ``WHITENED_ONSOURCE`` (model 2 also
``ROLLING_PEARSON_ONSOURCE``) and returns two logits (noise, signal).

Where this example departs from the paper:

- **Training data.** Noise (type 0) and coherent WNBs (type 1) only, in the
  paper's ratio of the two (6:3 for model 1, 10:10 for model 2). The paper also
  trains on single-detector WNBs (type 3, model 1) and incoherent WNBs in each
  detector (types 2a-2c, model 2), labelled noise; the library cannot yet
  generate those.
- **Output.** Two logits with a two-class softmax cross-entropy, where the
  paper has two sigmoid units with binary cross-entropy: the library's
  validator ranks the log-odds ``z1 - z0`` of a two-class softmax. The
  combined score (`MLy.combined_score`) is the log-odds of the product of the
  two models' signal probabilities, which ranks exactly as the paper's product.
- **Correlation lags.** The library's rolling Pearson spans the detectors'
  largest light travel time, every lag from -M to +M samples with M rounded up
  (57 lags for H1, L1 and V1 at 1024 Hz), where the paper uses ±30 ms (60 lags).
  The weight count is the same.
- **Schedule.** Model 2's learning rate is constant (the paper halves it when
  the loss stalls for 5 epochs). Both runs keep the model with the lowest
  validation loss and stop early on it (the paper keeps the best validation
  accuracy and stops on the training loss), and draw new data every step,
  where the paper reuses one fixed set of 150,000 or 300,000 examples.
  Keras 3's Nadam has a fixed momentum schedule (the authors' set
  ``schedule_decay`` to 2e-6).
- **Noise.** Coloured Gaussian noise from the library's detector PSDs, with no
  rescaling of Virgo's PSD (the paper raises it by 4 for model 1 and 32 for
  model 2). The whitening filter is band-limited to the dataset's spectral band
  (20 Hz to Nyquist by default), which stands in for the paper's 20 Hz
  high-pass.
- **WNB band.** The paper gives the bandwidth (uniform over 40-480 Hz) inside a
  20-512 Hz search band, not the lower edge; the library draws the two edges
  independently, so the lower edge is uniform over 20-30 Hz and the upper over
  70-500 Hz, which spans 40-480 Hz of bandwidth (flat over 50-470 Hz). The
  library also redraws every burst whose duration times bandwidth is below
  ``WNBGenerator.min_time_bandwidth`` (4): 0.15% of these draws, all shorter
  than 0.1 s.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import keras
import numpy as np
from keras import ops
from keras.layers import (
    Add, BatchNormalization, Concatenate, Conv1D, Dense, GlobalAveragePooling1D, Permute,
)
from keras.models import Model

import gravyflow as gf
from gravyflow.src.dataset.conditioning.pearson import num_lags
from gravyflow.src.dataset.dataset import get_max_arrival_time_difference
from gravyflow.src.model.examples.base import MODEL_INPUT, ExampleModel, model_input

# Model 2's second input: the Pearson correlation of each detector pair at each lag.
CORRELATION_INPUT = gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE


@dataclass(frozen=True)
class MLyConfig:
    """The paper's settings (Section III) and the authors' layer layouts."""
    # Data
    sample_rate_hertz: float = 1024.0
    onsource_duration_seconds: float = 1.0
    offsource_duration_seconds: float = 16.0
    ifos: Tuple[gf.IFO, ...] = (gf.IFO.H1, gf.IFO.L1, gf.IFO.V1)

    # WNBs: duration, and the band's lower and upper edges (see the module notes)
    wnb_duration_seconds: Tuple[float, float] = (0.05, 0.9)
    wnb_min_frequency_hertz: Tuple[float, float] = (20.0, 30.0)
    wnb_max_frequency_hertz: Tuple[float, float] = (70.0, 500.0)

    # Network SNR of the training signals, and the fraction of examples holding
    # one: the paper's ratio of noise to signal (6N:3N and 10N:10N).
    coincidence_snr: Tuple[float, float] = (12.0, 30.0)
    coherence_snr: Tuple[float, float] = (10.0, 50.0)
    coincidence_injection_chance: float = 1.0 / 3.0
    coherence_injection_chance: float = 0.5

    batch_size: int = 16

    # Model 1: residual blocks (filters, first kernel, second kernel); dense units
    coincidence_blocks: Tuple[Tuple[int, int, int], ...] = ((64, 42, 36), (128, 42, 36), (128, 42, 36))
    coincidence_dense_units: Tuple[int, ...] = (256, 256)
    # Nadam under a triangular cyclic learning rate between these bounds, half
    # a cycle every coincidence_half_cycle_steps steps: the paper's four passes
    # over its 135,000 training examples in batches of 16.
    coincidence_cyclic_learning_rate: Tuple[float, float] = (1e-5, 7e-3)
    coincidence_half_cycle_steps: int = 33_750

    # Model 2: (filters, kernel, stride) of the strain and correlation branches
    coherence_strain_convs: Tuple[Tuple[int, int, int], ...] = ((56, 32, 4), (128, 24, 2), (214, 16, 2))
    coherence_correlation_convs: Tuple[Tuple[int, int, int], ...] = ((64, 8, 1), (64, 5, 1))
    coherence_dense_units: int = 256
    coherence_learning_rate: float = 2e-3

    num_output_classes: int = 2

    @classmethod
    def cpu(cls) -> "MLyConfig":
        """The paper's settings with two detectors, H1 and L1."""
        return replace(cls(), ifos=(gf.IFO.H1, gf.IFO.L1))

    @property
    def correlation_shape(self) -> Tuple[int, int]:
        """``ROLLING_PEARSON_ONSOURCE`` per example: (detector pairs, lags)."""
        n = len(self.ifos)
        lags = num_lags(get_max_arrival_time_difference(self.ifos), self.sample_rate_hertz)
        return n * (n - 1) // 2, lags


@keras.saving.register_keras_serializable(package="gravyflow.mly")
class CyclicLearningRate(keras.optimizers.schedules.LearningRateSchedule):
    """
    A triangular cyclic learning rate (Smith 2017): from ``base`` up to
    ``peak`` over ``half_cycle_steps`` steps, then back down, repeated.
    """

    def __init__(self, base: float, peak: float, half_cycle_steps: int):
        self.base, self.peak, self.half_cycle_steps = float(base), float(peak), int(half_cycle_steps)

    def __call__(self, step):
        step = ops.cast(step, "float32")
        half = float(self.half_cycle_steps)
        cycle = ops.floor(1.0 + step / (2.0 * half))
        distance = ops.abs(step / half - 2.0 * cycle + 1.0)
        return self.base + (self.peak - self.base) * ops.maximum(0.0, 1.0 - distance)

    def get_config(self):
        return {"base": self.base, "peak": self.peak, "half_cycle_steps": self.half_cycle_steps}


def _uniform(bounds: Tuple[float, float]) -> gf.Distribution:
    return gf.Distribution(min_=bounds[0], max_=bounds[1], type_=gf.DistributionType.UNIFORM)


def mly_dataset_args(
    config: MLyConfig,
    snr: Tuple[float, float],
    injection_chance: float,
    input_variables: list,
    noise_type: gf.NoiseType = gf.NoiseType.COLORED,
    **kwargs,
) -> dict:
    """
    Arguments of a dataset of WNBs in noise from ``config.ifos`` (coloured by
    default), with ``INJECTION_PRESENT`` labels.
    """
    return {
        "sample_rate_hertz": config.sample_rate_hertz,
        "onsource_duration_seconds": config.onsource_duration_seconds,
        "offsource_duration_seconds": config.offsource_duration_seconds,
        "noise_obtainer": gf.NoiseObtainer(noise_type=noise_type, ifos=list(config.ifos)),
        "waveform_generators": gf.WNBGenerator(
            duration_seconds=_uniform(config.wnb_duration_seconds),
            min_frequency_hertz=_uniform(config.wnb_min_frequency_hertz),
            max_frequency_hertz=_uniform(config.wnb_max_frequency_hertz),
            scaling_method=gf.ScalingMethod(value=_uniform(snr), type_=gf.ScalingTypes.SNR),
            injection_chance=injection_chance,
        ),
        "num_examples_per_batch": config.batch_size,
        "input_variables": input_variables,
        "output_variables": [gf.ReturnVariables.INJECTION_PRESENT],
        **kwargs,
    }


def _classifier_compile(model: keras.Model, learning_rate) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    return model


class _MLyModel(ExampleModel):
    """The two networks' shared data: subclasses give the layers, inputs and SNRs."""

    config = MLyConfig()
    _snr_attr: str
    _chance_attr: str
    _inputs: Tuple[gf.ReturnVariables, ...]

    @classmethod
    def _dataset_args(cls, config: MLyConfig, noise_type: gf.NoiseType, **kwargs) -> dict:
        return mly_dataset_args(
            config, getattr(config, cls._snr_attr), getattr(config, cls._chance_attr),
            list(cls._inputs), noise_type=noise_type, **kwargs)

    @classmethod
    def dataset(
        cls,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
        config: Optional[MLyConfig] = None,
        seed: Optional[int] = None,
        group: str = None,
        steps_per_epoch: int = 1000,
        **kwargs
    ) -> gf.GravyflowDataset:
        """Training data: noise, and WNBs in the paper's SNR range and ratio."""
        cfg = config or cls.config
        return gf.GravyflowDataset(**cls._dataset_args(
            cfg, noise_type, seed=seed, group=group, steps_per_epoch=steps_per_epoch, **kwargs))

    @classmethod
    def validation_dataset_args(
        cls,
        config: Optional[MLyConfig] = None,
        noise_type: gf.NoiseType = gf.NoiseType.COLORED,
    ) -> dict:
        """
        Dataset arguments for `gf.Validator`, which sets the SNR range, the
        injection chance and the seeds itself.
        """
        return cls._dataset_args(config or cls.config, noise_type)


class MLyCoincidence(_MLyModel):
    """
    Model 1, the coincidence network: three residual blocks of two
    convolutions, global average pooling, two dense layers, two logits.

    Each convolution and dense layer applies ReLU and then batch normalisation;
    each block adds its first convolution's output to its output.
    """

    _snr_attr, _chance_attr = "coincidence_snr", "coincidence_injection_chance"
    _inputs = (MODEL_INPUT,)

    @classmethod
    def model(cls, config: Optional[MLyConfig] = None) -> keras.Model:
        """``WHITENED_ONSOURCE`` (detectors, samples) in; two logits out."""
        cfg = config or cls.config
        inputs = model_input(cfg.sample_rate_hertz, cfg.onsource_duration_seconds, len(cfg.ifos))
        x = Permute((2, 1), name="permute")(inputs)  # (samples, detectors), as the paper
        for n, (filters, first_kernel, second_kernel) in enumerate(cfg.coincidence_blocks, start=1):
            first = Conv1D(filters, first_kernel, padding="same", activation="relu", name=f"block{n}_conv1")(x)
            y = BatchNormalization(name=f"block{n}_bn1")(first)
            y = Conv1D(filters, second_kernel, padding="same", activation="relu", name=f"block{n}_conv2")(y)
            y = BatchNormalization(name=f"block{n}_bn2")(y)
            x = Add(name=f"block{n}_add")([first, y])
        x = GlobalAveragePooling1D(name="pool")(x)
        for n, units in enumerate(cfg.coincidence_dense_units, start=1):
            x = Dense(units, activation="relu", name=f"dense{n}")(x)
            x = BatchNormalization(name=f"dense{n}_bn")(x)
        outputs = Dense(cfg.num_output_classes, name="output")(x)
        return Model(inputs={inputs.name: inputs}, outputs=outputs, name="MLyCoincidence")

    @classmethod
    def compile_model(cls, model: keras.Model, config: Optional[MLyConfig] = None) -> keras.Model:
        """Nadam under the paper's cyclic learning rate; cross-entropy of logits."""
        cfg = config or cls.config
        base, peak = cfg.coincidence_cyclic_learning_rate
        return _classifier_compile(model, CyclicLearningRate(base, peak, cfg.coincidence_half_cycle_steps))


class MLyCoherence(_MLyModel):
    """
    Model 2, the coherence network: a strain branch (three strided
    convolutions) and a correlation branch (two convolutions), each pooled,
    concatenated, then one dense layer and two logits.

    Each convolution and the dense layer apply ReLU and then batch
    normalisation.
    """

    _snr_attr, _chance_attr = "coherence_snr", "coherence_injection_chance"
    _inputs = (MODEL_INPUT, CORRELATION_INPUT)

    @classmethod
    def model(cls, config: Optional[MLyConfig] = None) -> keras.Model:
        """
        ``WHITENED_ONSOURCE`` (detectors, samples) and
        ``ROLLING_PEARSON_ONSOURCE`` (pairs, lags) in; two logits out.

        Raises:
            ValueError: fewer lags than the correlation branch's kernels span.
        """
        cfg = config or cls.config
        if len(cfg.ifos) < 2:
            raise ValueError("The coherence network needs at least two detectors.")
        pairs, lags = cfg.correlation_shape
        span = sum(kernel - 1 for _, kernel, _ in cfg.coherence_correlation_convs) + 1
        if lags < span:
            raise ValueError(f"{lags} correlation lags at {cfg.sample_rate_hertz:g} Hz; the correlation "
                             f"branch needs at least {span}.")

        strain = model_input(cfg.sample_rate_hertz, cfg.onsource_duration_seconds, len(cfg.ifos))
        x = Permute((2, 1), name="strain_permute")(strain)
        for n, (filters, kernel, stride) in enumerate(cfg.coherence_strain_convs, start=1):
            x = Conv1D(filters, kernel, strides=stride, activation="relu", name=f"strain_conv{n}")(x)
            x = BatchNormalization(name=f"strain_bn{n}")(x)
        x = GlobalAveragePooling1D(name="strain_pool")(x)

        correlation = keras.Input(shape=(pairs, lags), name=CORRELATION_INPUT.name, dtype="float32")
        c = Permute((2, 1), name="correlation_permute")(correlation)  # (lags, pairs), as the paper
        for n, (filters, kernel, stride) in enumerate(cfg.coherence_correlation_convs, start=1):
            c = Conv1D(filters, kernel, strides=stride, activation="relu", name=f"correlation_conv{n}")(c)
            c = BatchNormalization(name=f"correlation_bn{n}")(c)
        c = GlobalAveragePooling1D(name="correlation_pool")(c)

        f = Concatenate(name="merge")([x, c])
        f = Dense(cfg.coherence_dense_units, activation="relu", name="dense")(f)
        f = BatchNormalization(name="dense_bn")(f)
        outputs = Dense(cfg.num_output_classes, name="output")(f)
        return Model(inputs={strain.name: strain, correlation.name: correlation}, outputs=outputs,
                     name="MLyCoherence")

    @classmethod
    def compile_model(cls, model: keras.Model, config: Optional[MLyConfig] = None) -> keras.Model:
        """Nadam at the paper's constant learning rate; cross-entropy of logits."""
        cfg = config or cls.config
        return _classifier_compile(model, cfg.coherence_learning_rate)


class MLy:
    """The pipeline: both networks, and their combined score."""

    config = MLyConfig()

    @classmethod
    def model(cls, config: Optional[MLyConfig] = None) -> Tuple[keras.Model, keras.Model]:
        """The two networks: (coincidence, coherence)."""
        cfg = config or cls.config
        return MLyCoincidence.model(cfg), MLyCoherence.model(cfg)

    @staticmethod
    def combined_score(coincidence_logits, coherence_logits) -> np.ndarray:
        """
        The combined detection score, as a log-odds (float64): the paper
        multiplies the two networks' signal probabilities, p = p1·p2; this is
        log(p / (1 - p)), which ranks exactly as p does but does not saturate.

        Args:
            coincidence_logits, coherence_logits: (examples, 2) logits.
        """
        s1, s2 = (np.asarray(o, dtype=np.float64) for o in (coincidence_logits, coherence_logits))
        s1, s2 = s1[:, 1] - s1[:, 0], s2[:, 1] - s2[:, 0]

        def log_sigmoid(x):
            return -np.logaddexp(0.0, -x)

        log_p = log_sigmoid(s1) + log_sigmoid(s2)
        # 1 - p1·p2 = (1 - p1) + p1·(1 - p2)
        log_not_p = np.logaddexp(log_sigmoid(-s1), log_sigmoid(s1) + log_sigmoid(-s2))
        return log_p - log_not_p


__all__ = ["CORRELATION_INPUT", "CyclicLearningRate", "MLy", "MLyCoherence", "MLyCoincidence", "MLyConfig",
           "mly_dataset_args"]
