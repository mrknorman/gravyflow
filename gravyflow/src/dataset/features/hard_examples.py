"""
Hard examples: signals that a detector network cannot receive from any
astrophysical source of their generator, each matched to its generator's real signals but for one
violation of what a real signal is.

A ``gf.HardExampleGenerator`` wraps a generator and one violation. It draws its
waveforms as the generator does (its own draws, under its own name, from the
generator's distributions), and the injection layer places, projects, scales,
crops and whitens them exactly as it does the generator's, so a hard example
differs from a real one in its violation alone:

- ``gf.WrongDelay``: one present detector's copy is shifted so that its delay
  from another present detector is that pair's light-travel time plus a margin,
  in the direction that keeps it where the generator's signals arrive where
  only one does (of whole signals, for a generator whose signal cut by the
  window can repeat);
- ``gf.MismatchedWaveforms``: each detector has a waveform of its own, its
  differing draws its own and the rest shared;
- ``gf.TimeReversed``: the waveform runs backwards in time, for a generator
  whose reversal leaves its family (a binary's chirp); every other is refused.

Each example says what it is: ``ReturnVariables.VIOLATION`` gives each
generator's ``gf.ViolationCode`` (NONE for a real signal).

Each violation is one class, a ``Violation``: what it checks, draws and changes
are its own hooks, which the hard example calls in every case. A new kind of
violation also adds its code to ``gf.ViolationCode``.
"""

from copy import copy
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import ClassVar, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C, ExampleNetworks
from gravyflow.src.dataset.features.injection import (
    SIGNAL_EXTENT, IncoherentGenerator, WaveformGenerator, _common_values, _equal_by_specification,
    _signal_extent, get_support_bounds,
)
from gravyflow.src.dataset.tools.snr import _spectrum_for
from gravyflow.src.rng.consumers import check_name
from gravyflow.src.rng.tree import BatchStreams


#: The least a wrong delay's margin may be, in seconds. The projection's delays
#: are float32, within about 0.1 us of the float64 light travel a margin is
#: added to, so a smaller margin may leave a delay a real source makes. It
#: guards that rounding alone: a margin this small is a wrong delay by
#: construction, not one a model can detect (see gf.WrongDelay).
MIN_MARGIN_SECONDS = 1.0e-6


class ViolationCode(IntEnum):
    """
    What each example's signal is, as ``ReturnVariables.VIOLATION`` labels
    it: how it was built, not whether its violation shows above the noise
    (see gf.WrongDelay).
    """
    NONE = 0
    WRONG_DELAY = 1
    MISMATCHED_WAVEFORMS = 2
    TIME_REVERSED = 3


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
    spin). A content draw, a shape draw that is no parameter (a burst's
    ``"noise"``), never is.
    """
    if draw not in (generator.distributed_attributes or ()):
        return False
    value = getattr(generator, draw)
    for component in value if isinstance(value, (tuple, list)) else (value,):
        low, high = get_support_bounds(component, draw)
        if low != high:
            return False
    return True


@dataclass(frozen=True)
class Violation:
    """
    Base of the violations a gf.HardExampleGenerator gives its generator's
    signals. A violation is a frozen dataclass of its parameters with its
    ``code``, and it acts only through the hooks below, each of which changes
    nothing by default: the hard example calls every hook and holds no case of
    its own. So a new violation of an existing kind (a subclass that keeps its
    parent's code) is one class, and a new kind is one class and a member of
    gf.ViolationCode for its code, which only an edit of gf.ViolationCode
    adds. The base itself, a class that names no ``code``, and one whose code
    is NONE, a real signal's, are no violation and are refused by name.
    """
    code : ClassVar[ViolationCode]

    def __new__(cls, *args, **kwargs):
        # Here, not in __post_init__, so that a subclass's own __post_init__ cannot skip it.
        code = getattr(cls, "code", None)
        if not isinstance(code, ViolationCode):
            raise TypeError(
                f"{cls.__name__} names no code (a gf.ViolationCode), so it labels nothing: gf.Violation is the base "
                f"of the violations, not one. Use one of them ({_usable_violations()}), or give a subclass its code."
            )
        if code is ViolationCode.NONE:
            raise TypeError(
                f"{cls.__name__}'s code is gf.ViolationCode.NONE, a real signal's, so every one of its examples "
                "would be labelled a real signal. A violation labels its examples by a code of its own: a new kind "
                "of violation adds its code to gf.ViolationCode."
            )
        return super().__new__(cls)

    def check(self, hard_example : "HardExampleGenerator") -> None:
        """
        Refuse by name a hard example it cannot be given, wherever the hard
        example is used: one whose generator's signals it cannot violate, or
        whose own settings (its scaling, its placement: those the injection
        layer reads) it cannot hold.
        """

    def fewest_detectors(self, generator : WaveformGenerator) -> int:
        """
        The fewest independent detectors an example needs present for its
        signal to be one no source of ``generator`` gives, counted as the
        rank of the span of their response tensors (``_response_rank``), a
        network that can present fewer refused by name: by default, one more
        than the waveforms its polarisations carry
        (``independent_waveforms``), since that many detectors of
        independent responses receive any waveforms in them, so a violation
        of the detectors' coherence (a wrong delay, a mismatch) is a real
        signal there.
        """
        return generator.independent_waveforms + 1

    def reach_seconds(self, network) -> float:
        """
        The farthest it moves a detector's copy in time from a real arrival of
        the example's signal, either way, on ``network``: the window holds that
        much either side of where real signals arrive
        (HardExampleGenerator.check_window). 0.0: it moves no copy.
        """
        return 0.0

    def generate(self, generator : WaveformGenerator, network, num_waveforms : int, sample_rate_hertz : float,
                 duration_seconds : float, streams : BatchStreams):
        """
        The polarisations, (Batch, 2, Time) or (Batch, Detectors, 2, Time),
        and their parameters with the signal's extent (``SIGNAL_EXTENT``),
        drawn from ``generator`` (every example injected) on ``network``:
        the generator's own.
        """
        waveforms, parameters = generator.generate(num_waveforms, sample_rate_hertz, duration_seconds, streams)
        parameters[SIGNAL_EXTENT] = np.stack(_signal_extent(generator, parameters, num_waveforms), axis=-1)
        return waveforms, parameters

    def detector_balance(self, polarisations, background, sample_rate_hertz : float,
                         onsource_duration_seconds : Optional[float]):
        """Each detector's gain on its copy as it is projected, (Batch, Detectors), or None: none."""
        return None

    def projection(self, hard_example, network, time_delays, present : np.ndarray, stream, num_examples : int,
                   offsets_seconds, extent_seconds, window_seconds : Optional[float]):
        """
        Its change to the projection: each detector's shift, (Batch,
        Detectors) seconds, or None (see HardExampleGenerator.draw_projection),
        from the violation's streams (``stream(name)``): None, no shift.
        """
        return None


def _usable_violations() -> str:
    """
    The violations a hard example takes, by name: every subclass of
    Violation, at any depth, whose code (its own or inherited) is a
    gf.ViolationCode other than NONE, the rule ``Violation.__new__`` builds by.
    """
    usable, pending = [], list(Violation.__subclasses__())
    while pending:
        violation = pending.pop(0)
        pending.extend(violation.__subclasses__())
        code = getattr(violation, "code", None)
        if isinstance(code, ViolationCode) and code is not ViolationCode.NONE and violation.__name__ not in usable:
            usable.append(violation.__name__)
    return ", ".join(usable)


@dataclass(frozen=True)
class WrongDelay(Violation):
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

    The margin is exact for the drawn pair only. The shifted copy's delay from
    each other present detector is what it is: it may exceed that pair's
    light travel by more than the margin, or lie within it. So the margin is
    a lower bound on the largest violation an example shows, and a margin
    drawn smaller (a harder example, as a curriculum would draw it) sets that
    bound, not the violation every pair sees.

    The label (gf.ViolationCode.WRONG_DELAY) marks the construction, not its
    detectability. What a wrong delay leaves that no real source gives is
    the part of the detectors' strain that no source's polarisations, at any
    sky, fit, and in noise that part can be smaller than the noise: for a
    generator whose polarisations carry two waveforms (a white-noise burst)
    on three detectors, a wrong delay at a network SNR of 12 can leave less
    than an SNR of 1 of it. A label of what a model can detect needs a margin
    of its own, measured on the examples.

    The example is placed as the generator's signals are, exactly, and the
    copy's reference time is kept where a real signal's reference time
    arrives in a detector (``arrival_bounds_seconds``: the generator's
    placement bounds widened by the detector margin, which leaves its whole
    signal in the window unless the generator allows partial signals), by
    its direction: the drawn one, or the other where only that keeps it
    there. So the copy reaches no part of the window a real arrival does not,
    though its density near the range's edges need not be a real arrival's.
    A range narrower than twice the farthest shift (the network's longest
    light travel plus the largest margin), for the generator's longest
    signal, cannot hold the copy whatever the placement and is refused by
    name when the dataset is built and as the copy is projected.

    A delay shows only where the window holds something of the signal that a
    shift changes: its envelope's ends, a changing frequency, its content.
    What a window holds of a signal it cuts off can repeat itself (the
    generator's ``repeats_when_cut``: a periodic wave, a chirplet whose start
    and end frequencies can be equal, or a sine-Gaussian, longer than the
    window; a ringdown's tail, a binary's included), and a shift of that part
    is a shift by whole periods, or a gain and a phase, so its delay shows only
    modulo its period, and a wrong delay may be a real one. So partial
    signals (the hard example's own allow_partially_out_of_window) of such a
    generator are refused by name. A whole signal shows its shift by as much
    as its envelope changes: a long periodic wave, or a sine-Gaussian of high
    quality factor, little.
    """
    margin_seconds : Union[float, "gf.Distribution"]

    code : ClassVar[ViolationCode] = ViolationCode.WRONG_DELAY

    def __post_init__(self):
        margin = _distribution(self.margin_seconds, "WrongDelay's margin_seconds")
        low, high = get_support_bounds(margin, "WrongDelay's margin_seconds")
        if not (np.isfinite(low) and np.isfinite(high) and low >= MIN_MARGIN_SECONDS):
            raise ValueError(
                f"WrongDelay's margin_seconds must be bounded and at least {MIN_MARGIN_SECONDS:g} s (what the "
                f"projection's float32 delays resolve), so that every wrong delay exceeds its pair's light travel; "
                f"{margin!r} reaches {low:g} to {high:g} s."
            )
        object.__setattr__(self, "margin_seconds", margin)

    @property
    def max_margin_seconds(self) -> float:
        return float(get_support_bounds(self.margin_seconds, "WrongDelay's margin_seconds")[1])

    def check(self, hard_example):
        """Refuse by name partial signals of a generator whose signal cut by the window can repeat (see the class)."""
        generator = hard_example.generator
        if hard_example.allow_partially_out_of_window and generator.repeats_when_cut:
            raise ValueError(
                f"A WrongDelay hard example of a {type(generator).__name__} cannot take partial signals "
                "(allow_partially_out_of_window): what the window holds of a signal it cuts off can repeat itself "
                "(repeats_when_cut: a wave of one frequency or period, its envelope's ends outside the window, or a "
                "ringdown's tail), and a time shift of that is a shift by whole periods, or a gain and a phase, so "
                "its delay shows only modulo its period and a wrong delay may be one a real source gives. Keep its "
                "signals whole in the window."
            )

    def reach_seconds(self, network) -> float:
        """
        ``network``'s longest light travel plus the largest margin, the
        farthest a copy lies from its partner's arrival, a real one: 0.0
        without a network.
        """
        if network is None:
            return 0.0
        return float(network.max_arrival_time_difference_seconds) + self.max_margin_seconds

    def projection(self, hard_example, network, time_delays, present, stream, num_examples,
                   offsets_seconds, extent_seconds, window_seconds):
        """One present detector's copy shifted from its partner's arrival, by its direction (see the class)."""
        num_detectors = present.shape[-1]
        rows = np.arange(num_examples)
        detector = _choose(present, stream("detector"))
        partners = present.copy()
        partners[rows, detector] = 0.0
        partner = _choose(partners, stream("partner"))
        drawn = np.where(stream("sign").random(num_examples) < 0.5, -1.0, 1.0)
        margin = np.asarray(self.margin_seconds.sample(num_examples, stream("margin")), dtype=np.float64)
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
        sign = self._direction(hard_example, drawn, arrival, reach, extent_seconds, float(window_seconds))
        shifts = np.zeros((num_examples, num_detectors))
        # The copy's delay from its partner becomes sign x (light travel + margin).
        shifts[rows, detector] = sign * reach - (delays[rows, detector] - delays[rows, partner])
        return shifts

    @staticmethod
    def _direction(hard_example, drawn, arrival, reach, extent_seconds, window_seconds : float):
        """
        Each copy's direction: ``drawn``, unless the other puts the copy,
        ``reach`` from its partner's ``arrival`` (seconds from the window's
        centre), nearer where ``hard_example``'s real signals arrive in a
        detector (``arrival_bounds_seconds`` for the signal's
        ``extent_seconds``, before and after it). The partner's own arrival
        is a real one, so one direction at least keeps the copy there when the
        range is twice the reach wide (the refusal of ``check_window``),
        and where neither does, the copy lies outside it by no more than its
        partner, which placement's rounding to a sample can put there.
        """
        if extent_seconds is None and not hard_example.allow_partially_out_of_window:
            raise ValueError(
                "A WrongDelay hard example keeps its shifted copy where real signals arrive, its whole signal in "
                "the window, so its projection needs each signal's extent (its parameters' signal_extent_seconds)."
            )
        low, high, _ = hard_example.arrival_bounds_seconds(
            {SIGNAL_EXTENT: extent_seconds}, window_seconds, int(np.size(arrival)))

        def outside(sign):
            shifted = arrival + sign * reach
            return np.maximum(low - shifted, 0.0) + np.maximum(shifted - high, 0.0)

        return np.where(outside(drawn) <= outside(-drawn), drawn, -drawn)


@dataclass(frozen=True)
class MismatchedWaveforms(Violation):
    """
    A waveform of its own in each detector: what differs, and how much, is
    controlled. Each detector's waveform is the generator's, with the draws
    named in ``differ`` its own (from the generator's streams under
    ``comp/{j}``) and every other draw shared, so the detectors' waveforms
    differ in those draws alone: a parameter (one of the generator's
    ``distributed_attributes``, e.g. the masses, for chirps of other lengths)
    or a content draw the generator names among its ``shape_attributes``
    (``"noise"``, a white-noise burst's random content). One of them at least
    changes the waveform's shape (the generator's ``shape_attributes``: a
    binary's masses or spins, a burst's frequencies, length or noise) and can
    take more than one value: waveforms alike but for the amplitude and the
    phase each detector receives them with are what other skies and
    polarisations give, and a draw of one value is the same in every detector,
    so a mismatch in those draws alone is refused. The presence and the
    placement are shared, so the signal's extent, which placement reads, is
    every detector's: a draw that moves it (the generator's
    ``extent_attributes``: a burst's length, and a white-noise burst's band
    edges, drawn again with its length) is refused unless it takes one value,
    since the example placed for its longest burst would keep a detector's
    shorter one from the window's edges where a real burst of that length
    reaches them. So a binary differs in its masses or spins, a ringdown in
    its frequency, a chirplet in its frequencies, a periodic wave in its
    frequency and a white-noise burst in its noise; a sine-Gaussian, whose
    every shape draw sets its envelope's length, takes no mismatch. A
    parameter that differs is labelled NaN, as an
    IncoherentGenerator's is: the label as the generator returns it, so for a
    binary, which labels its heavier body body 1, a mismatch in mass_1_msun
    alone can make MASS_1_MSUN, MASS_2_MSUN or both NaN (where a detector's
    mass 1 falls below the shared mass 2, its bodies swap).

    Every detector carries one source's loudness: as it is projected, each
    detector's waveform is rescaled so that its noise-weighted polarisation
    power (<h+, h+> + <hx, hx> in that detector's noise, over the window) is
    detector 0's waveform's there (``detector_balance``). Differing masses at
    one distance are of different loudness, so without it the detectors'
    SNRs would spread wider than any real signal's; with it, each detector's
    SNR is what a real source with detector 0's waveform gives there, and
    differs from a real signal's only through the antenna pattern (exactly
    for a binary's dominant mode, whose polarisations' ratio is its
    inclination's, shared). Detector 0's waveform is a draw of the same
    source as any other's, so it stands for the source. For that reason a
    mismatch is scaled to a network SNR only (gf.ScalingTypes.SNR): an hrss,
    hpeak or distance of the detectors' different waveforms is refused.

    ``correlation`` (None, or a float or gf.Distribution in [0, 1), drawn per
    example from ``wave/{g}/violation/correlation``) mixes in the shared
    waveform, the one drawn with nothing differing: detector j's is
    sqrt(rho) h_shared + sqrt(1 - rho) h_j, so any two detectors' waveforms
    correlate by rho. It is taken only where the detectors differ in their
    content alone (a shape draw that is no parameter, which the generator
    names in its ``shape_attributes``: a white-noise burst's ``"noise"``): the
    content is independent, so the mix is a burst of the same parameters
    (Gaussian content of the same power) inside the same extent. Where a
    parameter differs, the shared waveform's own draw of it gives the mix
    another shape (two waves' or two binaries' chirps beat, a cue in one
    detector alone), so a correlation is refused by name. rho = 1 would be the coherent signal, a
    real one, so it is refused; None is rho = 0, the waveforms' own.
    """
    differ : Tuple[str, ...]
    correlation : Union[None, float, "gf.Distribution"] = None

    code : ClassVar[ViolationCode] = ViolationCode.MISMATCHED_WAVEFORMS

    def __post_init__(self):
        differ = (self.differ,) if isinstance(self.differ, str) else self.differ
        differ = tuple(differ) if isinstance(differ, (tuple, list)) else ()
        if not differ or not all(isinstance(name, str) for name in differ):
            raise ValueError(
                "MismatchedWaveforms differs in at least one named draw (a parameter, or 'noise'), "
                f"not {self.differ!r}."
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

    def check(self, hard_example):
        """
        Refuse by name a mismatch of draws its generator does not make, or
        that leave the shape alone, take one value or move the extent, a
        correlation of differing parameters, and a scaling other than a
        network SNR (see the class).
        """
        generator = hard_example.generator
        parameters = tuple(generator.distributed_attributes or ())
        # A content draw is a shape draw that is no parameter: a burst's noise.
        content = [draw for draw in generator.shape_attributes if draw not in parameters]
        unknown = [draw for draw in self.differ if draw not in parameters and draw not in content]
        if unknown:
            raise ValueError(
                f"MismatchedWaveforms differs in {unknown}, which {type(generator).__name__} does not draw: "
                f"name its parameters ({', '.join(parameters)}) or its content ({', '.join(content) or 'none'})."
            )
        drawn = [draw for draw in self.differ if draw in parameters]
        if self.correlation is not None and drawn:
            raise ValueError(
                "MismatchedWaveforms' correlation mixes in the shared waveform, drawn with nothing differing, so it "
                f"holds only where the detectors differ in their content alone ({', '.join(content) or 'none'} for "
                f"a {type(generator).__name__}): with {drawn} differing, the shared waveform's own draw gives it "
                "another shape, a beat with each detector's wave or chirp. Differ in its content alone, or give "
                "correlation=None."
            )
        shape = [draw for draw in self.differ if draw in generator.shape_attributes]
        if not shape:
            raise ValueError(
                f"MismatchedWaveforms of a {type(generator).__name__} differs in {list(self.differ)}, none "
                "of them a draw that changes the waveform's shape "
                f"({', '.join(generator.shape_attributes) or 'it names none'}): waveforms alike but for the "
                "amplitude and the phase each detector receives them with are what other skies and "
                "polarisations give. Differ in at least one of those."
            )
        if all(_draws_one_value(generator, draw) for draw in shape):
            raise ValueError(
                f"MismatchedWaveforms of a {type(generator).__name__} differs in {list(self.differ)}, but "
                f"each of its draws that changes the shape ({', '.join(shape)}) has one value only, so every "
                "detector would receive one waveform: a real signal. Give one of them a distribution of more "
                "than one value."
            )
        moving = [draw for draw in self.differ
                  if draw in generator.extent_attributes and not _draws_one_value(generator, draw)]
        if moving:
            free = [draw for draw in generator.shape_attributes if draw not in generator.extent_attributes]
            raise ValueError(
                f"MismatchedWaveforms of a {type(generator).__name__} differs in {moving}, which moves its signal's "
                f"extent (its extent_attributes, {', '.join(generator.extent_attributes)}): the example is placed "
                "once for every detector, as its longest signal allows, so a detector's shorter signal would never "
                "come as near the window's edge as a real signal of its length does. Differ in a shape draw that "
                f"leaves the extent alone ({', '.join(free) or 'it has none'})."
            )
        # A network SNR only: its detectors' waveforms have no one hrss, hpeak or distance.
        type_ = getattr(hard_example.scaling_method, "type_", None)
        if type_ is not None and type_ is not gf.ScalingTypes.SNR:
            raise ValueError(
                f"A MismatchedWaveforms hard example is scaled to a network SNR, not {type_}: each detector's waveform "
                "carries one source's noise-weighted loudness, which no one hrss, hpeak or distance of the "
                "detectors' different waveforms gives. Give its generator a gf.ScalingTypes.SNR scaling."
            )

    def generate(self, generator, network, num_waveforms, sample_rate_hertz, duration_seconds, streams):
        """
        Detector j's waveform from the streams that differ under ``comp/{j}``
        (see the class), and the signal's extent, every detector's, refused by
        name where one detector's differs (a draw the generator makes again
        with a differing one, which its ``extent_attributes`` leave out).
        """
        if network is None:
            raise ValueError("MismatchedWaveforms gives each detector its own waveform: give the generator a network.")
        waveforms, parameters, extents = [], [], []
        for j in range(_num_detectors(network)):
            routed = _MismatchedStreams(streams, self.differ, j)
            component_waveforms, component_parameters = generator.generate(
                num_waveforms, sample_rate_hertz, duration_seconds, routed)
            unused = [draw for draw in self.differ if draw not in routed.routed]
            if unused:
                raise ValueError(
                    f"MismatchedWaveforms differs in {unused}, which {type(generator).__name__} drew "
                    "nothing from: its waveforms would not differ there."
                )
            waveforms.append(jnp.asarray(component_waveforms))
            parameters.append(component_parameters)
            extents.append(np.stack(_signal_extent(generator, component_parameters, num_waveforms), axis=-1))
        if any(np.any(extent != extents[0]) for extent in extents[1:]):
            raise ValueError(
                f"MismatchedWaveforms of a {type(generator).__name__} differing in {list(self.differ)}: its "
                "detectors' signals extend differently, though the example is placed once for all of them, so a "
                "draw that moves the extent came out differently in each (one the generator makes again with a "
                "differing draw): it belongs in the generator's extent_attributes."
            )
        waveforms = jnp.stack(waveforms, axis=1)
        if self.correlation is not None:
            rho = np.asarray(
                self.correlation.sample(num_waveforms, streams.stream("violation/correlation")), dtype=np.float32)
            shared, _ = generator.generate(num_waveforms, sample_rate_hertz, duration_seconds, streams)
            rho = jnp.asarray(rho).reshape((-1, 1, 1, 1))
            waveforms = jnp.sqrt(rho) * jnp.asarray(shared)[:, None] + jnp.sqrt(1.0 - rho) * waveforms
        common = _common_values(parameters)
        common[SIGNAL_EXTENT] = extents[0]
        return waveforms, common

    def detector_balance(self, polarisations, background, sample_rate_hertz, onsource_duration_seconds):
        """
        Detector j's waveform rescaled so that its noise-weighted polarisation
        power, <h+, h+> + <hx, hx> in detector j's noise (``background``, its
        spectrum estimated as SNR scaling's is) over the onsource window, is
        what detector 0's waveform has there: every detector then carries one
        source's intrinsic loudness (see the class). A detector whose waveform
        is empty gets 0.
        """
        polarisations = ops.convert_to_tensor(polarisations, dtype="float32")
        spectrum = _spectrum_for(background, sample_rate_hertz, None, None, None, None, None)

        def rho(strain):
            # Each detector's sqrt(<h, h>) in its own noise, over the onsource window.
            if onsource_duration_seconds is not None:
                strain = gf.crop_samples(strain, onsource_duration_seconds, sample_rate_hertz)
            return gf.snr(strain, spectrum, sample_rate_hertz, per_detector=True)

        reference = ops.broadcast_to(polarisations[:, :1], ops.shape(polarisations))
        wanted = [rho(reference[:, :, p]) for p in (0, 1)]
        own = [rho(polarisations[:, :, p]) for p in (0, 1)]
        # Squared relative to the largest, so a waveform of unit amplitude in
        # physical noise (an SNR near 1e23) cannot overflow float32.
        largest = ops.max(ops.stack(wanted + own), axis=0)
        largest = ops.where(largest > 0, largest, 1.0)
        wanted, own = (sum((value / largest) ** 2 for value in values) for values in (wanted, own))
        return ops.where(own > 0, ops.sqrt(wanted / ops.where(own > 0, own, 1.0)), 0.0)


@dataclass(frozen=True)
class TimeReversed(Violation):
    """
    The waveform reversed in time about its reference time (a binary's
    merger) before it is projected, so the reference time is placed as a real
    signal's is: a chirp's frequency falls from its merger.

    Only a generator that declares ``time_reversal_leaves_family`` takes it:
    gf.CBCGenerator, whose reversed chirp no binary makes. Any other is
    refused by name. The reversal of a sine-Gaussian, a white-noise burst, a
    periodic wave or a chirplet is one its generator makes (a symmetric
    wavelet, another burst, another wave, a chirplet from its end frequency to
    its start). A ringdown's reversal, which rises into its peak and stops,
    is no ringdown, but its signal then lies before its reference time where
    a ringdown's lies after it, so a reversed ringdown placed by its
    reference time is not placed as the generator's signals are.
    """
    code : ClassVar[ViolationCode] = ViolationCode.TIME_REVERSED

    def fewest_detectors(self, generator):
        """One: the reversed waveform is no real signal of its generator in any detector."""
        return 1

    def check(self, hard_example):
        generator = hard_example.generator
        if not generator.time_reversal_leaves_family:
            raise ValueError(
                f"TimeReversed takes a generator whose waveform reversed in time leaves its family "
                f"(time_reversal_leaves_family: a CBCGenerator), not a {type(generator).__name__}: a sine-Gaussian "
                "reversed is the same wavelet, a white-noise burst another such burst, a periodic wave a periodic "
                "wave, a chirplet one from its end to its start, and a ringdown's signal reversed lies before its "
                "reference time, so it would not be placed as a ringdown's signals are."
            )

    def generate(self, generator, network, num_waveforms, sample_rate_hertz, duration_seconds, streams):
        """The generator's waveforms reversed, and their extent with them."""
        waveforms, parameters = super().generate(
            generator, network, num_waveforms, sample_rate_hertz, duration_seconds, streams)
        before, after = np.moveaxis(parameters[SIGNAL_EXTENT], -1, 0)
        parameters[SIGNAL_EXTENT] = np.stack([after, before], axis=-1)
        return _reflect(waveforms), parameters


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


#: The least singular value of the present detectors' stacked response
#: tensors (each of Frobenius norm about 0.7, held in float64 by a Network and
#: ExampleNetworks) that counts as one more independent response. Responses
#: equal, opposite or summing to another's fall below it by far: the Einstein
#: Telescope's triangle, whose tensors sum to zero to 1e-7 in lal's float32
#: arm angles, leaves 1e-10; the least of H1, L1 and V1's is 0.19.
_INDEPENDENT_RESPONSE = 1.0e-6


def _response_rank(response, present) -> np.ndarray:
    """
    Each example's number of independent present detectors: the rank of the
    span of its present detectors' (``present``, (Batch, Detectors)) response
    tensors, (Detectors, 3, 3) or (Batch, Detectors, 3, 3), to
    ``_INDEPENDENT_RESPONSE``: (Batch,).

    Every source gives a detector its response tensor's contraction with the
    wave's, so detectors whose responses are equal, opposite or combinations
    of others' (a detector given twice, at one site or another; a triangle's
    three, which sum to zero) add a detector but no response in which a
    coherence violation shows where the others' would not: detectors of
    equal or opposite responses count once, and one whose response is a
    combination of others' adds none (a triangle counts two).
    The count is the safe one, so it may refuse a network where dependent
    detectors would show a violation plainly (two detectors of one response
    holding different waveforms).
    """
    present = np.asarray(present) > 0
    num_examples, num_detectors = present.shape
    response = np.broadcast_to(np.asarray(response, dtype=np.float64), (num_examples, num_detectors, 3, 3))
    rows = np.where(present[..., None], response.reshape(num_examples, num_detectors, 9), 0.0)
    return np.linalg.matrix_rank(rows, tol=_INDEPENDENT_RESPONSE)


def _fewest_independent(network) -> int:
    """
    The fewest independent detectors (``_response_rank``) an example of
    ``network`` can present: every subset of its fewest present detectors
    (``num_present``'s least, or all) counted, where its geometry is fixed
    (a Network; a gf.Detectors of bundled detectors and slots at fixed
    sites). Where a slot's geometry is drawn, the fewest present detectors,
    the most they can count: each example is counted as it is projected.
    """
    if isinstance(network, gf.Detectors):
        fewest = network.num_detectors if network.num_present is None else \
            int(np.ceil(get_support_bounds(network.num_present, "num_present")[0]))
        response = network.fixed_response()
        if response is None:
            return fewest
    else:
        response = np.asarray(network.response, dtype=np.float64)
        fewest = len(response)
    detectors = np.arange(len(response))
    subsets = np.array([np.isin(detectors, subset) for subset in combinations(detectors, fewest)])
    return int(np.min(_response_rank(response, subsets)))


def _num_detectors(network) -> int:
    return int(np.asarray(network.num_detectors).ravel()[0])


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
    gf.MismatchedWaveforms or gf.TimeReversed): matched to the
    generator's real signals in everything else.

    It draws as the generator does, from its distributions, but under its own
    name (``name``, default its position and class), so its examples are
    independent of the generator's own in a dataset beside it. Its scaling
    method, injection chance, padding, allow_partially_out_of_window, sky and
    reference time are its own, the generator's when it is built: the injection layer reads them from it,
    and they fingerprint it, so one changed afterwards (as the validator
    does) changes the hard example, and the generator's are not read. It
    compares, as it fingerprints, by those, its generator and its violation
    (``specification``). Its
    network is the generator's, or the dataset's. Placement,
    the SNR target over the present detectors, the window and whitening are
    the injection layer's, as for the generator, so a hard example sits in the
    window as a real signal does (a wrong delay keeps its shifted copy where
    its generator's signals arrive by the copy's direction, gf.WrongDelay).

    The violation acts through its own hooks (``Violation``), which this calls
    in every case. Every example needs the independent detectors present
    that the violation asks of the generator (``Violation.fewest_detectors``:
    one more than the independent waveforms the generator's polarisations
    carry for a wrong delay or a mismatch, one for a reversal), counted as the
    rank of the span of their response tensors (``_response_rank``:
    detectors of equal or opposite responses count once, and one whose
    response is a combination of others' adds none). A network whose
    geometry is fixed (a Network, or gf.Detectors of bundled detectors and
    fixed slots) that can give fewer is refused by name when it is set,
    every subset of its fewest present detectors counted; where a
    slot's geometry is drawn, its fewest present detectors are checked when it
    is set and each example as it is projected. TimeReversed takes only a
    binary, whose reversal leaves its family (gf.TimeReversed), and WrongDelay
    no partial signals of a generator whose signal cut by the window can
    repeat itself (``repeats_when_cut``, gf.WrongDelay).

    Its generator and violation may be reassigned after it is built, and its
    generator changed, as the validator changes a dataset's generators: each
    is checked to be what it can be (a coherent generator, not a
    gf.IncoherentGenerator or another hard example; a gf.Violation) as it is
    assigned, and every other check runs wherever the hard example is used,
    as its network is set (when it is built, and as a dataset binds it) and
    as each batch is projected (``_check``).

    ``ReturnVariables.VIOLATION`` gives each example's ``violation.code``.
    """

    __eq__ = _equal_by_specification

    def __init__(self, generator : WaveformGenerator, violation : Violation, name : Optional[str] = None):
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
        # Its network is set last, so the checks it runs (``_check``) see the rest.
        self.network = generator.network

    def __repr__(self) -> str:
        return f"HardExampleGenerator(generator={self.generator!r}, violation={self.violation!r}, name={self.name!r})"

    @property
    def generator(self) -> WaveformGenerator:
        return self._generator

    @generator.setter
    def generator(self, generator : WaveformGenerator):
        # Checked on every assignment, not only when it is built: an incoherent generator's detectors would hold
        # waveforms of their own labelled with this violation, and another hard example's violation would be dropped
        # with its projection.
        if not isinstance(generator, WaveformGenerator) or isinstance(
                generator, (IncoherentGenerator, HardExampleGenerator)):
            raise TypeError(
                f"A HardExampleGenerator wraps a coherent waveform generator (a CBCGenerator, a WNBGenerator...), "
                f"not {generator!r}."
            )
        self._generator = generator

    @property
    def violation(self) -> Violation:
        return self._violation

    @violation.setter
    def violation(self, violation : Violation):
        # Checked on every assignment, as its generator is.
        if not isinstance(violation, Violation):
            raise TypeError(
                f"A HardExampleGenerator's violation is a gf.Violation ({_usable_violations()}), not {violation!r}."
            )
        self._violation = violation

    @property
    def independent_waveforms(self) -> int:
        """Its generator's: the violation changes what the detectors receive, not what the polarisations carry."""
        return self.generator.independent_waveforms

    @property
    def shape_attributes(self) -> Tuple[str, ...]:
        """Its generator's, whose draws it makes."""
        return self.generator.shape_attributes

    @property
    def extent_attributes(self) -> Tuple[str, ...]:
        """Its generator's, whose draws it makes."""
        return self.generator.extent_attributes

    @property
    def repeats_when_cut(self) -> bool:
        """Its generator's, whose signals it makes."""
        return self.generator.repeats_when_cut

    def specification(self) -> dict:
        """
        The hard example by content (``utils.fingerprint.canonical`` reads it):
        its own settings (``_settings``, its name and network among them), its
        generator and its violation.
        """
        return {**self._settings(), "generator": self.generator, "violation": self.violation}

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        # Checked on every assignment: when it is built, and a dataset's binding.
        network = self.init_network(network, self.geometry_stream())
        self._check(network)
        self._network = network

    def _check(self, network, present : Optional[np.ndarray] = None) -> None:
        """
        Refuse by name what cannot hold its violation, wherever it is used:
        as its network is set (when it is built, and as a dataset binds it)
        and as each batch is projected (``draw_projection``). Its generator's
        draws and its own settings (``Violation.check``), since the generator
        is the caller's own object and either may change after the hard
        example is built (the generator's and the violation's types are
        checked as they are assigned); and the independent detectors an example can
        present (``Violation.fewest_detectors``): ``network``'s fewest, or
        where ``present`` (Batch, Detectors) is given, each example's own.
        """
        self.violation.check(self)
        if network is None:
            return
        if present is None:
            fewest, where = _fewest_independent(network), "its network can have"
        else:
            fewest, where = int(np.min(_response_rank(network.response, present))), "an example of this batch has"
        needed = self.violation.fewest_detectors(self.generator)
        if fewest < needed:
            raise ValueError(
                f"A {type(self.violation).__name__} hard example of a {type(self.generator).__name__} needs "
                f"{needed} detectors present in every example, but {where} {fewest} (counted as the rank of their "
                f"response tensors: detectors of equal or opposite responses count once, and one whose response is "
                f"a combination of others' adds none): there its signal may be one a real source gives (the "
                f"generator's polarisations carry {self.generator.independent_waveforms} independent waveforms; see "
                f"gf.{type(self.violation).__name__}). Give it more independent detectors, or a num_present of at "
                f"least {needed}."
            )

    def get_max_generated_duration(self):
        return self.generator.get_max_generated_duration()

    def get_signal_extent_seconds(self, parameters):
        """The extent ``generate`` records: the generator's, reversed with the waveform, or every detector's one."""
        extent = np.asarray(parameters[SIGNAL_EXTENT])
        return extent[..., 0], extent[..., 1]

    def in_band_duration_seconds(self, parameters, lower_frequency_hertz : float):
        """The generator's: a reversed chirp keeps its length, run backwards."""
        return self.generator.in_band_duration_seconds(parameters, lower_frequency_hertz)

    def arrival_bounds_seconds(self, parameters, window_seconds : float, num_examples : int):
        """
        Where each example's real reference time can arrive in a detector, as
        an offset from the window's centre, (low, high, too_long), each
        (Batch,): its placement bounds (``placement_bounds_seconds``, whose
        ``too_long`` it passes on) widened by the detector margin, the
        farthest a detector's arrival lies from the placed one. Unless the
        generator allows partial signals, the whole signal arriving there
        lies in the window, as the placement narrowed its bounds for exactly
        that.
        """
        low, high, too_long = self.placement_bounds_seconds(parameters, window_seconds, num_examples)
        margin = self.detector_margin_seconds()
        return low - margin, high + margin, too_long

    def check_window(self, window_seconds : float) -> None:
        """
        Refuse by name a window that cannot hold a copy its violation moves
        (``Violation.reach_seconds``) whatever the placement: the copy is kept
        where real signals arrive (``arrival_bounds_seconds``) and lies up to
        that reach from a real arrival, which can lie anywhere there, so that
        range must be twice the reach wide for any signal of the generator,
        its longest put either side of its reference time. Checked when a
        dataset is built, and at every projection (``draw_projection``), where
        the window meets the violation, so a gf.InjectionGenerator used alone
        refuses it too.
        """
        reach = self.violation.reach_seconds(self.network)
        if reach > 0.0:
            signal = float(self.generator.get_max_generated_duration())
            extents = {SIGNAL_EXTENT: np.array([[signal, 0.0], [0.0, signal]])}
            earliest, latest, too_long = self.arrival_bounds_seconds(extents, window_seconds, 2)
            width = 0.0 if np.any(too_long) else float(np.min(latest - earliest))
            if 2.0 * reach > width:
                held = "its reference time" if self.allow_partially_out_of_window else \
                    f"its signal (up to {signal:g} s long)"
                name = type(self.violation).__name__
                raise ValueError(
                    f"A {name} hard example moves a detector's copy up to {reach:g} s from a real arrival, either "
                    f"way (its reach_seconds), and keeps it where real signals arrive, which leaves {held} in the "
                    f"window: that needs a range {2.0 * reach:g} s wide, but the {window_seconds:g} s window and the "
                    f"padding, widened by the detector margin, give {width:g} s. Give the violation less reach (see "
                    f"gf.{name}), a longer window or more padding."
                )

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
        # Every example drawn, then its presence its own (the hard example's
        # injection chance, from its own chance stream), as a generator's is.
        generator = copy(self.generator)
        generator.injection_chance = 1.0
        waveforms, parameters = self.violation.generate(
            generator, self.network, num_waveforms, sample_rate_hertz, duration_seconds, streams)
        return self.apply_injection_chance(waveforms, streams), parameters

    def detector_balance(self, polarisations, background, sample_rate_hertz : float,
                         onsource_duration_seconds : Optional[float] = None):
        """
        The violation's gain for each detector's copy, (Batch, Detectors), or
        None (``Violation.detector_balance``: a mismatched waveform's
        detectors balanced to one source's loudness), its scaling checked as
        the batch was projected (``draw_projection``). ``polarisations`` is
        (Batch, Detectors, 2, Time) for one waveform per detector, the window
        the injection layer crops centred in it.
        """
        return self.violation.detector_balance(polarisations, background, sample_rate_hertz,
                                               onsource_duration_seconds)

    def draw_projection(
        self, network, time_delays, streams : BatchStreams, label : str, num_examples : int,
        offsets_seconds=None, extent_seconds=None, window_seconds : Optional[float] = None,
    ):
        """
        The violation's change to the projection, from the generator's
        ``inject/{label}/violation/...`` streams: each detector's shift,
        (Batch, Detectors) seconds, or None, the time its copy is moved by
        beyond its own delay (``time_delays``, the projection's, (Batch,
        Detectors)).

        A wrong delay keeps its copy in the window (``window_seconds`` long)
        by its direction, from each example's placement, ``offsets_seconds``
        (Batch,), its reference time's offset from the window's centre, and
        the signal's ``extent_seconds`` (Batch, 2), before and after it; a
        window that cannot hold its copy is refused (``check_window``).
        """
        if window_seconds is not None:
            self.check_window(float(window_seconds))
        num_detectors = int(ops.shape(time_delays)[-1])
        present = np.asarray(network.present, dtype=np.float64) if isinstance(network, ExampleNetworks) \
            else np.ones((num_examples, num_detectors))
        # Every check again, each example's network counted with its geometry known.
        self._check(network, present)

        def stream(name):
            return streams.stream(f"inject/{label}/violation/{name}")

        return self.violation.projection(
            self, network, time_delays, present, stream, num_examples,
            offsets_seconds, extent_seconds, window_seconds)
