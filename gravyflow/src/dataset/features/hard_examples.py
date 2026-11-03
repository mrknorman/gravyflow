"""
Hard examples: signals that a detector network cannot receive from any
astrophysical source, each matched to its generator's real signals but for one
violation of what a real signal is.

A ``gf.HardExampleGenerator`` wraps a generator and one violation. It draws its
waveforms as the generator does (its own draws, under its own name, from the
generator's distributions), and the injection layer places, projects, scales,
crops and whitens them exactly as it does the generator's, so a hard example
differs from a real one in its violation alone:

- ``gf.WrongDelay``: one present detector's copy is shifted so that its delay
  from another present detector is that pair's light-travel time plus a margin,
  in the direction that keeps it in the window where only one does;
- ``gf.OneDetector``: the signal is in exactly one present detector, which then
  holds the whole network SNR;
- ``gf.MismatchedWaveforms``: each detector has a waveform of its own, its
  differing draws its own and the rest shared;
- ``gf.TimeReversed``: the waveform runs backwards in time, for a generator
  whose reversal leaves its family (a binary's chirp, a ringdown); a reversed
  sine-Gaussian, white-noise burst, periodic wave or chirplet is one its
  generator makes, so it is refused.

Each example says what it is: ``ReturnVariables.VIOLATION`` gives each
generator's ``gf.ViolationCode`` (NONE for a real signal).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C, ExampleNetworks
from gravyflow.src.dataset.features.injection import (
    SIGNAL_EXTENT, IncoherentGenerator, WaveformGenerator, _common_values, get_support_bounds,
)
from gravyflow.src.rng.consumers import check_name
from gravyflow.src.rng.tree import BatchStreams


#: The least a wrong delay's margin may be, in seconds. The projection's delays
#: are float32, within about 0.1 us of the float64 light travel a margin is
#: added to, so a smaller margin may leave a delay a real source makes.
MIN_MARGIN_SECONDS = 1.0e-6


class ViolationCode(IntEnum):
    """What each example's signal is, as ``ReturnVariables.VIOLATION`` labels it."""
    NONE = 0
    WRONG_DELAY = 1
    ONE_DETECTOR = 2
    MISMATCHED_WAVEFORMS = 3
    TIME_REVERSED = 4


def _distribution(value, name : str) -> "gf.Distribution":
    """``value`` as a gf.Distribution: a float is a constant one."""
    if isinstance(value, gf.Distribution):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return gf.Distribution(value=float(value), type_=gf.DistributionType.CONSTANT)
    raise TypeError(f"{name} is a float or a gf.Distribution, not {value!r}.")


def _draws_one_value(generator : WaveformGenerator, draw : str) -> bool:
    """
    Whether every ``draw`` of ``generator`` is one value: a parameter whose
    distribution's support is a point, every component's for a vector (a
    spin). A burst's ``"noise"``, its random content, never is.
    """
    if draw == "noise":
        return False
    value = getattr(generator, draw)
    for component in value if isinstance(value, (tuple, list)) else (value,):
        low, high = get_support_bounds(component, draw)
        if low != high:
            return False
    return True


@dataclass(frozen=True)
class WrongDelay:
    """
    One present detector's copy shifted so that its delay from another present
    detector is the pair's light-travel time plus ``margin_seconds`` (a float
    or a gf.Distribution, drawn per example; its support finite and at least
    ``MIN_MARGIN_SECONDS``, so every wrong delay exceeds its pair's light
    travel), in either direction.

    The detector, its partner (each uniform among the example's present
    detectors), the direction (either, equally) and the margin are drawn from
    the generator's ``inject/{g}/violation/{detector,partner,sign,margin}``
    streams. The pair is the example's own, so a network drawn per example
    (gf.Detectors) is respected, and ``CENTRAL_TIME`` gives the shifted arrival.

    The example is placed as the generator's signals are, exactly, and the copy
    is kept in the window by its direction: the drawn one, or the other where
    only that keeps the copy (its whole signal, unless the generator allows
    partial signals) in the window. A window too short for that whatever the
    placement, shorter than twice the farthest shift (the network's longest
    light travel plus the largest margin) beside the generator's longest
    signal, is refused by name when the dataset is built.
    """
    margin_seconds : Union[float, "gf.Distribution"]

    code : ClassVar[ViolationCode] = ViolationCode.WRONG_DELAY

    def __post_init__(self):
        margin = _distribution(self.margin_seconds, "WrongDelay's margin_seconds")
        low, high = get_support_bounds(margin, "WrongDelay's margin_seconds")
        if not (np.isfinite(low) and np.isfinite(high) and low >= MIN_MARGIN_SECONDS):
            raise ValueError(
                f"WrongDelay's margin_seconds must be bounded and above {MIN_MARGIN_SECONDS:g} s (what the "
                f"projection's float32 delays resolve), so that every wrong delay exceeds its pair's light travel; "
                f"{margin!r} reaches {low:g} to {high:g} s."
            )
        object.__setattr__(self, "margin_seconds", margin)

    @property
    def max_margin_seconds(self) -> float:
        return float(get_support_bounds(self.margin_seconds, "WrongDelay's margin_seconds")[1])


@dataclass(frozen=True)
class OneDetector:
    """
    The signal in exactly one present detector (uniform among the example's
    present detectors, from ``inject/{g}/violation/detector``), scaled to the
    generator's network SNR target, which it then holds alone. The others
    receive no signal: their DETECTOR_SNR is 0 and their CENTRAL_TIME NaN.
    """
    code : ClassVar[ViolationCode] = ViolationCode.ONE_DETECTOR


@dataclass(frozen=True)
class MismatchedWaveforms:
    """
    A waveform of its own in each detector: what differs, and how much, is
    controlled. Each detector's waveform is the generator's, with the draws
    named in ``differ`` its own (from the generator's streams under
    ``comp/{j}``) and every other draw shared, so the detectors' waveforms
    differ in those draws alone: a parameter (one of the generator's
    ``distributed_attributes``, e.g. the masses, for chirps of other lengths)
    or ``"noise"``, a burst's random content (a WNB's). One of them at least
    changes the waveform's shape (the generator's ``shape_attributes``: a
    binary's masses or spins, a burst's frequencies, length or noise) and can
    take more than one value: waveforms alike but for the amplitude and the
    phase each detector receives them with are what other skies and
    polarisations give, and a draw of one value is the same in every detector,
    so a mismatch in those draws alone is refused. The presence and the
    placement are shared. A parameter that differs is labelled NaN, as an
    IncoherentGenerator's is.

    ``correlation`` (None, or a float or gf.Distribution in [0, 1), drawn per
    example from ``wave/{g}/violation/correlation``) mixes in the shared
    waveform, the one drawn with nothing differing: detector j's is
    sqrt(rho) h_shared + sqrt(1 - rho) h_j, so any two detectors' waveforms
    correlate by rho. For a burst whose content differs it is a burst of the
    same parameters (Gaussian content of the same power). rho = 1 would be the
    coherent signal, a real one, so it is refused; None is rho = 0, the
    waveforms' own.
    """
    differ : Tuple[str, ...]
    correlation : Union[None, float, "gf.Distribution"] = None

    code : ClassVar[ViolationCode] = ViolationCode.MISMATCHED_WAVEFORMS

    def __post_init__(self):
        differ = (self.differ,) if isinstance(self.differ, str) else tuple(self.differ)
        if not differ or not all(isinstance(name, str) for name in differ):
            raise ValueError(
                f"MismatchedWaveforms differs in at least one named draw (a parameter, or 'noise'), not {self.differ!r}."
            )
        object.__setattr__(self, "differ", differ)
        if self.correlation is not None:
            correlation = _distribution(self.correlation, "MismatchedWaveforms' correlation")
            low, high = get_support_bounds(correlation, "MismatchedWaveforms' correlation")
            if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low and high < 1.0):
                raise ValueError(
                    f"MismatchedWaveforms' correlation lies in [0, 1) (1 is the coherent signal, a real one); "
                    f"{correlation!r} reaches {low:g} to {high:g}."
                )
            object.__setattr__(self, "correlation", correlation)


@dataclass(frozen=True)
class TimeReversed:
    """
    The waveform reversed in time about its reference time (a CBC's merger, a
    ringdown's peak) before it is projected, so the reference time is placed
    as a real signal's is: a chirp's frequency falls from its merger, a
    ringdown rises into its peak and stops there.

    Only a generator whose reversed waveform is one it never makes, whatever
    its parameters, declares ``time_reversal_leaves_family`` and takes it:
    gf.CBCGenerator and gf.RingdownGenerator. Any other is refused by name:
    a sine-Gaussian's envelope is symmetric, so reversed it is the same
    wavelet; a white-noise burst's reversal is another such burst; a periodic
    wave's is a periodic wave; a chirplet's is a chirplet from its end
    frequency to its start.
    """
    code : ClassVar[ViolationCode] = ViolationCode.TIME_REVERSED


VIOLATIONS = (WrongDelay, OneDetector, MismatchedWaveforms, TimeReversed)

#: The violations that need two detectors present in every example.
_PAIRED = (WrongDelay, OneDetector, MismatchedWaveforms)


class _MismatchedStreams(BatchStreams):
    """
    A batch's streams as detector ``component`` of a mismatched waveform sees
    them: a consumer of a draw named in ``differ`` (``param/{name}...``, or
    ``noise``) under ``comp/{component}``, every other the shared one. The
    names it routed are recorded in ``routed``.
    """
    __slots__ = ("differ", "component", "routed")

    def __init__(self, streams : BatchStreams, differ : Tuple[str, ...], component : int):
        super().__init__(streams.tree, streams.split, streams.epoch, streams.k, streams.prefix)
        self.differ, self.component, self.routed = differ, component, set()

    def _route(self, consumer : str) -> str:
        parts = consumer.split("/")
        draw = parts[1] if parts[0] == "param" and len(parts) > 1 else parts[0]
        if draw not in self.differ:
            return consumer
        self.routed.add(draw)
        return f"comp/{self.component}/{consumer}"

    def stream(self, consumer : str):
        return super().stream(self._route(consumer))

    def key(self, consumer : str):
        return super().key(self._route(consumer))

    def sub(self, prefix : str):
        raise TypeError("A mismatched waveform's generator draws from its own streams, never a sub-tree of them.")


def _reflect(waveforms):
    """Each waveform reversed in time about its reference sample, N // 2: sample i takes sample 2 (N // 2) - i."""
    num_samples = int(ops.shape(waveforms)[-1])
    flipped = jnp.flip(jnp.asarray(waveforms), axis=-1)
    # flip maps i to N - 1 - i; the reflection about N // 2 is that, delayed by 2 (N // 2) - N + 1 samples.
    delay = 2 * (num_samples // 2) - num_samples + 1
    if delay == 0:
        return flipped
    padding = [(0, 0)] * (flipped.ndim - 1) + [(delay, 0)]
    return jnp.pad(flipped[..., :num_samples - delay], padding)


def _choose(candidates : np.ndarray, rng : np.random.Generator) -> np.ndarray:
    """One detector per example, uniform among its ``candidates`` ((Batch, Detectors), nonzero where allowed)."""
    keys = rng.random(candidates.shape)
    return np.argmax(np.where(candidates > 0, keys, -1.0), axis=1)


class HardExampleGenerator(WaveformGenerator):
    """
    ``generator``'s signals, each with one ``violation`` (gf.WrongDelay,
    gf.OneDetector, gf.MismatchedWaveforms or gf.TimeReversed): matched to the
    generator's real signals in everything else.

    It draws as the generator does, from its distributions, but under its own
    name (``name``, default its position and class), so its examples are
    independent of the generator's own in a dataset beside it. Its scaling
    method, injection chance, padding, sky and reference time are the
    generator's; its network is the generator's, or the dataset's. Placement,
    the SNR target over the present detectors, the window and whitening are
    the injection layer's, as for the generator, so a hard example sits in the
    window as a real signal does (a wrong delay keeps its shifted copy there
    by the copy's direction, gf.WrongDelay).

    The violations that move signal between detectors (all but TimeReversed)
    need two detectors present in every example: a network or a num_present
    that can give fewer is refused by name. TimeReversed takes only a
    generator whose reversal leaves its family (gf.TimeReversed).

    ``ReturnVariables.VIOLATION`` gives each example's ``violation.code``.
    """

    def __init__(self, generator : WaveformGenerator, violation, name : Optional[str] = None):
        if not isinstance(generator, WaveformGenerator) or isinstance(
                generator, (IncoherentGenerator, HardExampleGenerator)):
            raise TypeError(
                f"A HardExampleGenerator wraps a coherent waveform generator (a CBCGenerator, a WNBGenerator...), "
                f"not {generator!r}."
            )
        if not isinstance(violation, VIOLATIONS):
            raise TypeError(
                f"A HardExampleGenerator's violation is one of {', '.join(v.__name__ for v in VIOLATIONS)}, "
                f"not {violation!r}."
            )
        if isinstance(violation, TimeReversed) and not generator.time_reversal_leaves_family:
            raise ValueError(
                f"A {type(generator).__name__}'s waveform reversed in time is one its generator makes (a "
                "sine-Gaussian is the same wavelet, a white-noise burst another such burst, a periodic wave a "
                "periodic wave, a chirplet one from its end to its start), so it is no violation: TimeReversed takes "
                "a generator whose reversal leaves its family (time_reversal_leaves_family: a CBCGenerator, a "
                "RingdownGenerator)."
            )
        if isinstance(violation, MismatchedWaveforms):
            known = set(generator.distributed_attributes or ()) | {"noise"}
            unknown = [draw for draw in violation.differ if draw not in known]
            if unknown:
                raise ValueError(
                    f"MismatchedWaveforms differs in {unknown}, which {type(generator).__name__} does not draw: "
                    f"name its parameters ({', '.join(generator.distributed_attributes or ())}) or 'noise'."
                )
            shape = [draw for draw in violation.differ if draw in generator.shape_attributes]
            if not shape:
                raise ValueError(
                    f"MismatchedWaveforms of a {type(generator).__name__} differs in {list(violation.differ)}, none "
                    "of them a draw that changes the waveform's shape "
                    f"({', '.join(generator.shape_attributes) or 'it names none'}): waveforms alike but for the "
                    "amplitude and the phase each detector receives them with are what other skies and "
                    "polarisations give. Differ in at least one of those."
                )
            if all(_draws_one_value(generator, draw) for draw in shape):
                raise ValueError(
                    f"MismatchedWaveforms of a {type(generator).__name__} differs in {list(violation.differ)}, but "
                    f"each of its draws that changes the shape ({', '.join(shape)}) has one value only, so every "
                    "detector would receive one waveform: a real signal. Give one of them a distribution of more "
                    "than one value."
                )
        self.generator = generator
        self.violation = violation
        self.name = None if name is None else check_name(name, "generator name")
        for attribute in (
            "scaling_method", "injection_chance", "front_padding_duration_seconds",
            "back_padding_duration_seconds", "allow_partially_out_of_window",
            *WaveformGenerator.sky_attributes, "reference_gps_time_seconds",
        ):
            setattr(self, attribute, getattr(generator, attribute))
        self.distributed_attributes = generator.distributed_attributes
        self.network = generator.network

    def specification(self) -> dict:
        """The hard example by content (``utils.fingerprint.canonical`` reads it): its generator, violation, name and network."""
        return {"generator": self.generator, "violation": self.violation, "name": self.name, "network": self.network}

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        # Checked on every assignment, a dataset's binding included.
        network = self.init_network(network, self.geometry_stream())
        if network is not None and isinstance(self.violation, _PAIRED):
            fewest = self._fewest_present(network)
            if fewest < 2:
                raise ValueError(
                    f"A {type(self.violation).__name__} hard example needs two detectors present in every "
                    f"example, but its network can have {fewest}: give it more detectors, or a num_present "
                    "of at least 2."
                )
        self._network = network

    @staticmethod
    def _fewest_present(network) -> int:
        if isinstance(network, gf.Detectors):
            if network.num_present is None:
                return network.num_detectors
            return int(np.ceil(get_support_bounds(network.num_present, "num_present")[0]))
        return int(np.asarray(network.num_detectors).ravel()[0])

    @staticmethod
    def _num_detectors(network) -> int:
        return int(np.asarray(network.num_detectors).ravel()[0])

    def get_max_generated_duration(self):
        return self.generator.get_max_generated_duration()

    def get_signal_extent_seconds(self, parameters):
        """The extent recorded by ``generate``: the generator's, reversed with the waveform, or every detector's."""
        extent = np.asarray(parameters[SIGNAL_EXTENT])
        return extent[..., 0], extent[..., 1]

    def in_band_duration_seconds(self, parameters, lower_frequency_hertz : float):
        """The generator's: a reversed chirp keeps its length, run backwards."""
        return self.generator.in_band_duration_seconds(parameters, lower_frequency_hertz)

    def _reach_seconds(self, network) -> float:
        """
        The farthest a copy is shifted from its partner's arrival, either way:
        for a wrong delay, ``network``'s longest light travel plus the largest
        margin; 0.0 for the other violations, or without a network.
        """
        if network is None or not isinstance(self.violation, WrongDelay):
            return 0.0
        return float(network.max_arrival_time_difference_seconds) + self.violation.max_margin_seconds

    def placement_range_seconds(self, window_seconds : float):
        """
        The generator's. A wrong delay refuses a window that cannot hold its
        shifted copy whatever the placement: one shorter than twice the
        farthest shift beside the generator's longest signal (its reference
        time alone, if the generator allows partial signals).
        """
        reach = self._reach_seconds(self.network)
        if reach > 0.0:
            partial = self.allow_partially_out_of_window
            signal = 0.0 if partial else float(self.generator.get_max_generated_duration())
            if 2.0 * reach + signal > window_seconds:
                held = "its reference time" if partial else f"its signal (up to {signal:g} s long)"
                raise ValueError(
                    f"A WrongDelay hard example shifts its copy up to {reach:g} s from its partner's arrival (the "
                    f"network's longest light travel plus the largest margin), either way, and keeps {held} in the "
                    f"window: that needs a window of {2.0 * reach + signal:g} s, not {window_seconds:g} s. Give a "
                    "smaller margin or a longer window."
                )
        return super().placement_range_seconds(window_seconds)

    def reference_time_window_seconds(self, window_seconds : float, network=None):
        """The generator's, widened for a wrong delay by the farthest its copy is shifted from its partner's arrival."""
        start, end = super().reference_time_window_seconds(window_seconds, network)
        reach = self._reach_seconds(self.network if network is None else network)
        return max(start - reach, 0.0), min(end + reach, float(window_seconds))

    def generate(
        self,
        num_waveforms : int,
        sample_rate_hertz : float,
        duration_seconds : float,
        streams : Optional[BatchStreams] = None
    ):
        """
        ``num_waveforms`` hard examples' polarisations, (Batch, 2, Time), or
        (Batch, Detectors, 2, Time) for mismatched waveforms, and their
        parameters: the generator's, a reversed waveform's or the detectors'
        common values.
        """
        streams = self.resolve_streams(streams)
        if isinstance(self.violation, MismatchedWaveforms):
            return self._mismatched(num_waveforms, sample_rate_hertz, duration_seconds, streams)
        waveforms, parameters = self.generator.generate(num_waveforms, sample_rate_hertz, duration_seconds, streams)
        before, after = self._extent(self.generator.get_signal_extent_seconds(parameters), num_waveforms)
        if isinstance(self.violation, TimeReversed):
            waveforms = _reflect(waveforms)
            before, after = after, before
        parameters[SIGNAL_EXTENT] = np.stack([before, after], axis=-1)
        return waveforms, parameters

    @staticmethod
    def _extent(extent, num_waveforms : int):
        return tuple(np.broadcast_to(np.asarray(side, dtype=np.float64), (num_waveforms,)) for side in extent)

    def _mismatched(self, num_waveforms, sample_rate_hertz, duration_seconds, streams):
        """Detector j's waveform from the streams that differ under ``comp/{j}`` (see MismatchedWaveforms)."""
        if self.network is None:
            raise ValueError("MismatchedWaveforms gives each detector its own waveform: give the generator a network.")
        violation = self.violation
        waveforms, parameters, before, after = [], [], [], []
        for j in range(self._num_detectors(self.network)):
            routed = _MismatchedStreams(streams, violation.differ, j)
            component_waveforms, component_parameters = self.generator.generate(
                num_waveforms, sample_rate_hertz, duration_seconds, routed)
            unused = [draw for draw in violation.differ if draw not in routed.routed]
            if unused:
                raise ValueError(
                    f"MismatchedWaveforms differs in {unused}, which {type(self.generator).__name__} drew "
                    "nothing from: its waveforms would not differ there."
                )
            waveforms.append(jnp.asarray(component_waveforms))
            parameters.append(component_parameters)
            extent = self._extent(self.generator.get_signal_extent_seconds(component_parameters), num_waveforms)
            before.append(extent[0])
            after.append(extent[1])
        waveforms = jnp.stack(waveforms, axis=1)
        if violation.correlation is not None:
            rho = np.asarray(
                violation.correlation.sample(num_waveforms, streams.stream("violation/correlation")), dtype=np.float32)
            shared, _ = self.generator.generate(num_waveforms, sample_rate_hertz, duration_seconds, streams)
            rho = jnp.asarray(rho).reshape((-1, 1, 1, 1))
            waveforms = jnp.sqrt(rho) * jnp.asarray(shared)[:, None] + jnp.sqrt(1.0 - rho) * waveforms
        common = _common_values(parameters)
        common[SIGNAL_EXTENT] = np.stack([np.max(before, axis=0), np.max(after, axis=0)], axis=-1)
        return waveforms, common

    def draw_projection(
        self, network, time_delays, streams : BatchStreams, label : str, num_examples : int,
        offsets_seconds=None, extent_seconds=None, window_seconds : Optional[float] = None,
    ):
        """
        The violation's change to the projection, from the generator's
        ``inject/{label}/violation/...`` streams: (shifts, kept), each
        (Batch, Detectors) or None. ``shifts`` (seconds) is the time each
        detector's copy is moved by, beyond its own delay (``time_delays``,
        the projection's, (Batch, Detectors)); ``kept`` is 1 in the detectors
        that keep the signal and 0 in the others.

        A wrong delay keeps its copy in the window (``window_seconds`` long)
        by its direction, from each example's placement, ``offsets_seconds``
        (Batch,), its reference time's offset from the window's centre, and
        the signal's ``extent_seconds`` (Batch, 2), before and after it.
        """
        violation = self.violation
        if not isinstance(violation, (WrongDelay, OneDetector)):
            return None, None
        num_detectors = int(ops.shape(time_delays)[-1])
        present = np.asarray(network.present, dtype=np.float64) if isinstance(network, ExampleNetworks) \
            else np.ones((num_examples, num_detectors))
        if np.any(np.sum(present > 0, axis=1) < 2):
            raise ValueError(
                f"A {type(violation).__name__} hard example needs two detectors present, but an example has one."
            )

        def stream(name):
            return streams.stream(f"inject/{label}/violation/{name}")

        rows = np.arange(num_examples)
        detector = _choose(present, stream("detector"))
        if isinstance(violation, OneDetector):
            kept = np.zeros((num_examples, num_detectors), dtype=np.float32)
            kept[rows, detector] = 1.0
            return None, kept

        partners = present.copy()
        partners[rows, detector] = 0.0
        partner = _choose(partners, stream("partner"))
        drawn = np.where(stream("sign").random(num_examples) < 0.5, -1.0, 1.0)
        margin = np.asarray(violation.margin_seconds.sample(num_examples, stream("margin")), dtype=np.float64)
        location = np.broadcast_to(np.asarray(network.location, dtype=np.float64), (num_examples, num_detectors, 3))
        light_seconds = np.linalg.norm(location[rows, detector] - location[rows, partner], axis=-1) / C
        delays = np.asarray(time_delays, dtype=np.float64)
        reach = light_seconds + margin
        if offsets_seconds is None or window_seconds is None:
            raise ValueError(
                "A WrongDelay hard example keeps its shifted copy in the window, so its projection needs each "
                "example's placement (the injection parameter 'shift') and the window's length."
            )
        arrival = np.asarray(offsets_seconds, dtype=np.float64) + delays[rows, partner]
        sign = self._direction(drawn, arrival, reach, extent_seconds, float(window_seconds))
        shifts = np.zeros((num_examples, num_detectors))
        # The copy's delay from its partner becomes sign x (light travel + margin).
        shifts[rows, detector] = sign * reach - (delays[rows, detector] - delays[rows, partner])
        return shifts, None

    def _direction(self, drawn, arrival, reach, extent_seconds, window_seconds : float):
        """
        Each copy's direction: ``drawn``, or the other where only that keeps
        the copy, ``reach`` from its partner's ``arrival`` (seconds from the
        window's centre), in the window: its whole signal, ``extent_seconds``
        before and after it, or its reference time if the generator allows
        partial signals. Where neither does, the partner's own signal is one
        the window cannot hold (placement centres it, with a warning), and
        the drawn direction is kept.
        """
        if self.allow_partially_out_of_window:
            before = after = 0.0
        elif extent_seconds is None:
            raise ValueError(
                "A WrongDelay hard example keeps its shifted copy's whole signal in the window, so its projection "
                "needs each signal's extent (its parameters' signal_extent_seconds)."
            )
        else:
            extent = np.asarray(extent_seconds, dtype=np.float64)
            before, after = extent[..., 0], extent[..., 1]
        half = window_seconds / 2.0

        def fits(sign):
            copy = arrival + sign * reach
            return (copy - before >= -half) & (copy + after <= half)

        return np.where(fits(drawn) | ~fits(-drawn), drawn, -drawn)
