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
  only one does;
- ``gf.MismatchedWaveforms``: each detector has a waveform of its own, its
  differing draws its own and the rest shared;
- ``gf.TimeReversed``: the waveform runs backwards in time, for a generator
  whose reversal leaves its family (a binary's chirp); every other is refused.

Each example says what it is: ``ReturnVariables.VIOLATION`` gives each
generator's ``gf.ViolationCode`` (NONE for a real signal).

Each violation is one class, a ``Violation``: what it checks, draws and changes
are its own hooks, which the hard example calls in every case.
"""

from copy import copy
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
from gravyflow.src.dataset.tools.snr import _spectrum_for
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
    its own, so a new violation is one class.
    """
    code : ClassVar[ViolationCode]

    def check(self, generator : WaveformGenerator) -> None:
        """Refuse by name a generator whose signals it cannot violate, when the hard example is built."""

    def fewest_detectors(self, generator : WaveformGenerator) -> int:
        """
        The fewest detectors an example needs present for its signal to be
        one no source of ``generator`` gives, a network that can present
        fewer refused by name: by default, one more than the waveforms its
        polarisations carry (``independent_waveforms``), since that many
        detectors receive any waveforms in them, so a violation of the
        detectors' coherence (a wrong delay, a mismatch) is a real signal
        there.
        """
        return generator.independent_waveforms + 1

    def distinct_detectors(self, response, location, present : np.ndarray) -> np.ndarray:
        """
        How many of each example's present detectors (``present``, (Batch,
        Detectors)) count towards ``fewest_detectors``, from their response
        tensors (Detectors, 3, 3) or (Batch, Detectors, 3, 3) and locations
        ((Batch,) Detectors, 3): (Batch,). By default co-located detectors of
        one response, equal or opposite, count once, since every source gives
        them one signal (up to its sign).
        """
        return _distinct(present, response, location)

    def check_scaling(self, scaling_method) -> None:
        """Refuse by name a scaling it cannot be given, when the hard example is built and as it is scaled."""

    def reach_seconds(self, network) -> float:
        """
        The farthest it moves a detector's copy in time from its partner's
        arrival, either way, on ``network``: 0.0, none.
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
        parameters[SIGNAL_EXTENT] = np.stack(_extent(generator, parameters, num_waveforms), axis=-1)
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
    name when the dataset is built.
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

    def reach_seconds(self, network) -> float:
        """``network``'s longest light travel plus the largest margin: 0.0 without a network."""
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
        low, high = hard_example.arrival_bounds_seconds(
            {SIGNAL_EXTENT: extent_seconds}, window_seconds, int(np.size(arrival)))

        def outside(sign):
            copy = arrival + sign * reach
            return np.maximum(low - copy, 0.0) + np.maximum(copy - high, 0.0)

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

    def check(self, generator):
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

    def check_scaling(self, scaling_method):
        """A network SNR only: its detectors' waveforms have no one hrss, hpeak or distance."""
        type_ = getattr(scaling_method, "type_", None)
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
            extents.append(np.stack(_extent(generator, component_parameters, num_waveforms), axis=-1))
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

    def distinct_detectors(self, response, location, present):
        """Every present detector: each holds a waveform of its own, which detectors alike only make plainer."""
        return np.sum(np.asarray(present) > 0, axis=-1)

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

    def check(self, generator):
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


def _extent(generator : WaveformGenerator, parameters, num_waveforms : int):
    """``generator``'s signal extent for its ``parameters``, (before, after), each (Batch,) float64."""
    return tuple(np.broadcast_to(np.asarray(side, dtype=np.float64), (num_waveforms,))
                 for side in generator.get_signal_extent_seconds(parameters))


#: How far apart two detectors' response tensors (entries of order 1/2, held
#: in float32 by a fixed network) and locations (metres) may be and still be
#: one: a detector given twice, or slots at one site, are equal to the bit.
_SAME_RESPONSE, _SAME_LOCATION_METERS = 1.0e-6, 1.0


def _distinct(present, response, location=None) -> np.ndarray:
    """
    Each example's number of present detectors (``present``, (Batch,
    Detectors)) that differ from every other in their response tensor, up to
    its sign, or, given ``location``, in their response or their location:
    (Batch,).
    """
    present = np.asarray(present) > 0
    num_examples, num_detectors = present.shape
    response = np.broadcast_to(np.asarray(response, dtype=np.float64), (num_examples, num_detectors, 3, 3))
    pairs = response[:, :, None] - response[:, None], response[:, :, None] + response[:, None]
    same = np.min([np.max(np.abs(pair), axis=(-2, -1)) for pair in pairs], axis=0) <= _SAME_RESPONSE
    if location is not None:
        location = np.broadcast_to(np.asarray(location, dtype=np.float64), (num_examples, num_detectors, 3))
        same &= np.max(np.abs(location[:, :, None] - location[:, None]), axis=-1) <= _SAME_LOCATION_METERS
    # A detector counts unless one present before it is the same.
    earlier = np.tril(np.ones((num_detectors, num_detectors), dtype=bool), -1)
    repeated = np.any(same & earlier & present[:, None, :], axis=-1)
    return np.sum(present & ~repeated, axis=-1)


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
    method, injection chance, padding, sky and reference time are its own,
    the generator's when it is built: the injection layer reads them from it,
    and they fingerprint it, so one changed afterwards (as the validator
    does) changes the hard example, and the generator's are not read. Its
    network is the generator's, or the dataset's. Placement,
    the SNR target over the present detectors, the window and whitening are
    the injection layer's, as for the generator, so a hard example sits in the
    window as a real signal does (a wrong delay keeps its shifted copy where
    its generator's signals arrive by the copy's direction, gf.WrongDelay).

    The violation acts through its own hooks (``Violation``), which this calls
    in every case. Every example needs the detectors present that the
    violation asks of the generator (``Violation.fewest_detectors``: one
    more than the independent waveforms the generator's polarisations carry
    for a wrong delay or a mismatch, one for a reversal), counted as the violation counts them
    (``Violation.distinct_detectors``: detectors that receive one signal
    alike may count once): a network or a num_present that can
    give fewer is refused by name when it is set, and an example with fewer
    as it is projected. TimeReversed takes only a binary, whose reversal
    leaves its family (gf.TimeReversed).

    ``ReturnVariables.VIOLATION`` gives each example's ``violation.code``.
    """

    def __init__(self, generator : WaveformGenerator, violation : Violation, name : Optional[str] = None):
        if not isinstance(generator, WaveformGenerator) or isinstance(
                generator, (IncoherentGenerator, HardExampleGenerator)):
            raise TypeError(
                f"A HardExampleGenerator wraps a coherent waveform generator (a CBCGenerator, a WNBGenerator...), "
                f"not {generator!r}."
            )
        if not isinstance(violation, Violation):
            raise TypeError(
                f"A HardExampleGenerator's violation is a gf.Violation "
                f"({', '.join(v.__name__ for v in Violation.__subclasses__())}), not {violation!r}."
            )
        violation.check(generator)
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
        violation.check_scaling(self.scaling_method)

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
        # Checked on every assignment, a dataset's binding included.
        network = self.init_network(network, self.geometry_stream())
        if network is not None:
            self._check_detectors(self._fewest_present(network), "its network can have")
        self._network = network

    def _check_detectors(self, fewest : int, where : str) -> None:
        """Refuse by name ``fewest`` detectors present, counted by the violation, where it needs more."""
        needed = self.violation.fewest_detectors(self.generator)
        if fewest < needed:
            raise ValueError(
                f"A {type(self.violation).__name__} hard example of a {type(self.generator).__name__} needs "
                f"{needed} detectors present in every example, but {where} {fewest} (counted as "
                f"gf.{type(self.violation).__name__} counts them: detectors that receive one signal alike may count "
                f"once): there its signal is one a real source gives (the generator's "
                f"polarisations carry {self.generator.independent_waveforms} independent waveforms; see "
                f"gf.{type(self.violation).__name__}). Give it more detectors, or a num_present of at least {needed}."
            )

    def _fewest_present(self, network) -> int:
        """The fewest detectors an example of ``network`` can present, counted by the violation where its geometry is fixed."""
        if isinstance(network, gf.Detectors):
            if network.num_present is None:
                return network.num_detectors
            return int(np.ceil(get_support_bounds(network.num_present, "num_present")[0]))
        present = np.ones((1, _num_detectors(network)))
        return int(self.violation.distinct_detectors(network.response, network.location, present)[0])

    def get_max_generated_duration(self):
        return self.generator.get_max_generated_duration()

    def get_signal_extent_seconds(self, parameters):
        """The extent ``generate`` records: the generator's, reversed with the waveform, or every detector's one."""
        extent = np.asarray(parameters[SIGNAL_EXTENT])
        return extent[..., 0], extent[..., 1]

    def in_band_duration_seconds(self, parameters, lower_frequency_hertz : float):
        """The generator's: a reversed chirp keeps its length, run backwards."""
        return self.generator.in_band_duration_seconds(parameters, lower_frequency_hertz)

    def arrival_bounds_seconds(self, parameters, window_seconds : float, num_examples : int, range_seconds=None):
        """
        Where each example's real reference time can arrive in a detector, as
        an offset from the window's centre, (low, high), each (Batch,): its
        placement bounds (``placement_bounds_seconds``) widened by the
        detector margin, the farthest a detector's arrival lies from the
        placed one. Unless the generator allows partial signals, the whole
        signal arriving there lies in the window, as the placement narrowed
        its bounds for exactly that.
        """
        low, high, _ = self.placement_bounds_seconds(parameters, window_seconds, num_examples, range_seconds)
        margin = self.detector_margin_seconds()
        return low - margin, high + margin

    def check_window(self, window_seconds : float) -> None:
        """
        Refuse by name a window that cannot hold a shifted copy
        (``reach_seconds``, a wrong delay's) whatever the placement: its copy
        is kept where real signals arrive (``arrival_bounds_seconds``,
        gf.WrongDelay), and its partner's arrival can lie anywhere there, so
        that range must be twice the farthest shift wide for any signal of the
        generator, its longest put either side of its reference time. Checked
        when a dataset is built, and at every projection (``draw_projection``),
        where the window meets the violation, so a gf.InjectionGenerator used
        alone refuses it too.
        """
        low, high = self.placement_range_seconds(window_seconds)
        reach = self.violation.reach_seconds(self.network)
        if reach > 0.0:
            signal = float(self.generator.get_max_generated_duration())
            extents = {SIGNAL_EXTENT: np.array([[signal, 0.0], [0.0, signal]])}
            _, _, too_long = self.placement_bounds_seconds(extents, window_seconds, 2, range_seconds=(low, high))
            earliest, latest = self.arrival_bounds_seconds(extents, window_seconds, 2, range_seconds=(low, high))
            width = 0.0 if np.any(too_long) else float(np.min(latest - earliest))
            if 2.0 * reach > width:
                held = "its reference time" if self.allow_partially_out_of_window else \
                    f"its signal (up to {signal:g} s long)"
                raise ValueError(
                    f"A {type(self.violation).__name__} hard example shifts its copy up to {reach:g} s from its "
                    f"partner's arrival (the network's longest light travel plus the largest margin), either way, "
                    f"and keeps it where real signals arrive, which leaves {held} in the window: that needs a "
                    f"range {2.0 * reach:g} s wide, but the {window_seconds:g} s window and the padding, widened "
                    f"by the detector margin, give {width:g} s. Give a smaller margin, a longer window or more "
                    "padding."
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
        detectors balanced to one source's loudness), its scaling checked
        again, as it may have changed since the hard example was built.
        ``polarisations`` is (Batch, Detectors, 2, Time) for one waveform per
        detector, the window the injection layer crops centred in it.
        """
        self.violation.check_scaling(self.scaling_method)
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
        # A network drawn per example is counted here, its geometry known.
        self._check_detectors(
            int(np.min(self.violation.distinct_detectors(network.response, network.location, present))),
            "an example of this batch has")

        def stream(name):
            return streams.stream(f"inject/{label}/violation/{name}")

        return self.violation.projection(
            self, network, time_delays, present, stream, num_examples,
            offsets_seconds, extent_seconds, window_seconds)
