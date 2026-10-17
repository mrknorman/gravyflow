from dataclasses import dataclass, fields
from enum import Enum, auto
from itertools import count
from typing import Callable, Dict, Iterator, List, Optional, Union
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
import jax

import gravyflow as gf

from gravyflow.src.dataset.conditioning.detector import DEFAULT_NOISE_PROFILE, read_noise_curve
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from gravyflow.src.dataset.config import WindowSpec, ignore_scale_factor
from gravyflow.src.rng import BatchStreams, SeedTree
from gravyflow.src.rng.legacy import warn_deprecated
from gravyflow.src.utils.shapes import ShapeEnforcer

class NoiseType(Enum):
    WHITE = auto()
    COLORED = auto()
    PSEUDO_REAL = auto()
    REAL = auto()

from gravyflow.src.utils.numerics import ensure_even

#: A noise batcher: the batch at an address, from its streams.
NoiseBatcher = Callable[[BatchStreams], Dict]


def unit_normal_noise(
    streams: BatchStreams,
    ifos: List["gf.IFO"],
    window: str,
    num_examples_per_batch: int,
    num_samples: int,
):
    """
    Unit Gaussian noise, (Batch, IFO, Samples), float32.

    Each detector's noise comes from its own stream, ``noise/{IFO}/{window}``,
    so adding a detector changes no other detector's noise.
    """
    return jnp.stack([
        jax.random.normal(
            streams.key(f"noise/{ifo.name}/{window}"),
            shape=(num_examples_per_batch, num_samples),
            dtype=jnp.float32
        )
        for ifo in ifos
    ], axis=1)


def color_noise(white_noise, interpolated_asd, *, sample_rate_hertz: float):
    """
    Colour unit Gaussian noise so that its one-sided PSD is ``interpolated_asd ** 2``.

    ``interpolated_asd`` is the ASD on the rfft grid of the noise's length,
    broadcastable to (Batch, IFOs, Freqs). White noise w ~ N(0, 1) is coloured
    as x = irfft(sqrt(fs / 2) * ASD * rfft(w)): the factor sqrt(fs / 2) turns
    unit-variance samples into a one-sided density of 1 / Hz.
    """
    num_samples = white_noise.shape[-1]
    colored_noise_fft = (
        np.float32(np.sqrt(sample_rate_hertz / 2.0)) * interpolated_asd * jnp.fft.rfft(white_noise)
    )
    return jnp.fft.irfft(colored_noise_fft, n=num_samples)


def interpolate_onsource_offsource_psd(
    num_samples_list: List[int],
    sample_rate_hertz: float,
    psd_freq: jnp.ndarray,
    psd_val: jnp.ndarray
) -> List[jnp.ndarray]:
    """
    Interpolates a spectrum onto the rfft grid of each length in ``num_samples_list``.

    Each bin k gets the spectrum at its own frequency, k * fs / N. DC and
    frequencies below the spectrum's lowest are set to zero; frequencies above
    its highest hold its last value. Returns the onsource and offsource spectra.
    """

    psd_freq = jnp.asarray(psd_freq, dtype=jnp.float32)
    psd_val = jnp.asarray(psd_val, dtype=jnp.float32)

    interpolated_psds = []
    
    for num in num_samples_list:
        new_freqs = jnp.asarray(
            np.fft.rfftfreq(int(num), 1.0 / sample_rate_hertz), dtype=jnp.float32
        )
        
        interp_val = jnp.interp(new_freqs, psd_freq, psd_val)
        interp_val = jnp.where(
            (new_freqs > 0.0) & (new_freqs >= psd_freq[0]), interp_val, 0.0
        )
        
        # Complex, to multiply the complex rfft.
        interp_val = ops.cast(interp_val, "complex64")
        
        interpolated_psds.append(interp_val)
    
    return interpolated_psds[0], interpolated_psds[1]



def white_noise_batcher(
    num_examples_per_batch: int,
    ifos: List["gf.IFO"],
    onsource_duration_seconds: float,
    crop_duration_seconds: float,
    offsource_duration_seconds: float,
    sample_rate_hertz: float,
) -> NoiseBatcher:
    """
    White Gaussian noise, one batch per address: a dict of ONSOURCE and
    OFFSOURCE, each (Batch, IFO, Samples). Synthetic noise has no
    START_GPS_TIME or DATA_LABEL.
    """
    window_spec = WindowSpec.from_params(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds
    )

    def batch(streams: BatchStreams) -> Dict:
        return {
            RV.ONSOURCE: unit_normal_noise(
                streams, ifos, "on", num_examples_per_batch, window_spec.num_onsource_samples
            ),
            RV.OFFSOURCE: unit_normal_noise(
                streams, ifos, "off", num_examples_per_batch, window_spec.num_offsource_samples
            ),
        }

    return batch


def colored_noise_batcher(
    num_examples_per_batch: int,
    onsource_duration_seconds: float,
    crop_duration_seconds : float,
    offsource_duration_seconds: float,
    ifos : List["gf.IFO"],
    sample_rate_hertz: float,
    noise_profile : "gf.NoiseProfile" = None
) -> NoiseBatcher:
    """
    Colored Gaussian noise, one batch per address, in strain.

    Each detector's noise has the one-sided PSD of its curve in
    ``noise_profile`` (default ``DEFAULT_NOISE_PROFILE``, O3_ACTUAL), and no
    power below the curve's band. A sample rate whose Nyquist frequency lies
    beyond a curve is refused (``read_noise_curve``): no curve is extrapolated.
    """
    window_spec = WindowSpec.from_params(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds
    )
    num_samples_list = [
        ensure_even(int(duration * sample_rate_hertz)) for duration in (
            window_spec.total_onsource_duration_seconds, offsource_duration_seconds
        )
    ]

    profile = DEFAULT_NOISE_PROFILE if noise_profile is None else noise_profile
    onsource_asds, offsource_asds = [], []
    for ifo in ifos:
        frequencies, asd = read_noise_curve(profile.path(ifo), up_to_hertz=sample_rate_hertz / 2.0)
        onsource_asd, offsource_asd = interpolate_onsource_offsource_psd(
            num_samples_list,
            sample_rate_hertz,
            jnp.array(frequencies, dtype=jnp.float32),
            jnp.array(asd, dtype=jnp.float32)
        )
        onsource_asds.append(onsource_asd)
        offsource_asds.append(offsource_asd)

    # (1, IFOs, Freqs), to broadcast over the batch.
    onsource_asds = ops.expand_dims(ops.stack(onsource_asds, axis=0), 0)
    offsource_asds = ops.expand_dims(ops.stack(offsource_asds, axis=0), 0)

    def batch(streams: BatchStreams) -> Dict:
        return {
            RV.ONSOURCE: color_noise(
                unit_normal_noise(streams, ifos, "on", num_examples_per_batch, num_samples_list[0]),
                onsource_asds,
                sample_rate_hertz=sample_rate_hertz
            ),
            RV.OFFSOURCE: color_noise(
                unit_normal_noise(streams, ifos, "off", num_examples_per_batch, num_samples_list[1]),
                offsource_asds,
                sample_rate_hertz=sample_rate_hertz
            ),
        }

    return batch


def white_noise_generator(
    num_examples_per_batch: int,
    ifos: List["gf.IFO"],
    onsource_duration_seconds: float,
    crop_duration_seconds: float,
    offsource_duration_seconds: float,
    sample_rate_hertz: float,
    seed: int
) -> Iterator:
    """Deprecated: white_noise_batcher, iterated over the direct streams of ``seed``."""
    warn_deprecated("white_noise_generator", "call white_noise_batcher(...) with each batch's streams")
    batch = white_noise_batcher(
        num_examples_per_batch, ifos, onsource_duration_seconds, crop_duration_seconds,
        offsource_duration_seconds, sample_rate_hertz,
    )
    tree = SeedTree(seed)
    return (batch(tree.direct(k)) for k in count())


def colored_noise_generator(
    num_examples_per_batch: int,
    onsource_duration_seconds: float,
    crop_duration_seconds: float,
    offsource_duration_seconds: float,
    ifos: List["gf.IFO"],
    sample_rate_hertz: float,
    scale_factor: None,
    seed: int,
    noise_profile: "gf.NoiseProfile" = None
) -> Iterator:
    """
    Deprecated: colored_noise_batcher, iterated over the direct streams of
    ``seed``. ``scale_factor`` is ignored: the noise is strain.
    """
    warn_deprecated("colored_noise_generator", "call colored_noise_batcher(...) with each batch's streams")
    ignore_scale_factor(scale_factor)
    batch = colored_noise_batcher(
        num_examples_per_batch, onsource_duration_seconds, crop_duration_seconds,
        offsource_duration_seconds, ifos, sample_rate_hertz, noise_profile=noise_profile,
    )
    tree = SeedTree(seed)
    return (batch(tree.direct(k)) for k in count())


class ReaderExhausted(Exception):
    """A sequential reader has no batch left: a GRID pass has ended, or no segment loaded."""


def sequential_batcher(iterator: Iterator) -> NoiseBatcher:
    """
    A batcher over a sequential reader: each call takes the reader's next
    batch, whatever the address. PSEUDO_REAL and GRID noise (until it is
    addressed by index) and a reader of the caller's own are read this way. A reader that ends raises
    ``ReaderExhausted``, by name: a bare StopIteration would end, or break,
    whatever generator called the batcher.
    """
    def batch(streams: BatchStreams) -> Dict:
        try:
            return next(iterator)
        except StopIteration:
            raise ReaderExhausted("The noise reader has no batch left: a GRID pass has ended, "
                                  "or no segment could be loaded.") from None

    return batch


def _until_exhausted(batches: Iterator) -> Iterator:
    """``batches`` until its reader is exhausted (``ReaderExhausted``), then the end."""
    try:
        yield from batches
    except ReaderExhausted:
        return


def enforce_shapes(batch: Dict, num_ifos: int) -> Dict:
    """One batch, checked against the noise shape contracts."""
    return next(ShapeEnforcer.wrap_generator(iter([batch]), num_ifos=num_ifos))


def sequential_seed(tree: SeedTree, split: str, consumer: str) -> int:
    """An integer seed for a sequential reader, from its own stream of the tree."""
    return int(tree.stream(split, consumer).integers(2**63))


@dataclass
class Obtainer:
    """Base class for data obtainers (noise and transient)."""
    ifo_data_obtainer: Union[None, gf.IFODataObtainer] = None
    ifos: List[gf.IFO] = None

    def specification(self) -> dict:
        """
        What determines the noise served (``validate.provenance.canonical`` reads it):
        every field.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        if self.ifos is None:
            self.ifos = [gf.IFO.L1]
        elif not isinstance(self.ifos, list) and not isinstance(self.ifos, tuple):
            self.ifos = [self.ifos]


@dataclass
class NoiseObtainer(Obtainer):
    """
    Obtainer for noise data (white, colored, pseudo-real, or real).

    Real noise with no ``ifo_data_obtainer`` uses public GWOSC data: O1, O2 and O3,
    DATA with the BURST_CAT1 and BURST_CAT2 vetoes, with known events and Gravity Spy
    glitches removed.

    Noise is returned in strain (WHITE noise, which has no detector, in unit
    variance). COLORED noise follows ``noise_profile``: by default
    ``gf.NoiseProfile.O3_ACTUAL``, each detector's measured O3 curve, which the
    obtainer then names, so that a fingerprint of it names the curve.

    An obtainer is a specification: building batches never changes it, so one
    obtainer can serve any number of datasets and splits.
    """
    noise_type: NoiseType = NoiseType.REAL
    noise_profile: "gf.NoiseProfile" = None

    def __post_init__(self):
        super().__post_init__()
        if self.noise_type == NoiseType.REAL and self.ifo_data_obtainer is None:
            self.ifo_data_obtainer = gf.IFODataObtainer(
                data_quality=gf.DataQuality.BEST,
                data_labels=[gf.DataLabel.NOISE],
            )
        if self.noise_type == NoiseType.COLORED and self.noise_profile is None:
            self.noise_profile = DEFAULT_NOISE_PROFILE

    def batcher(
            self,
            tree : SeedTree,
            group : str,
            sample_rate_hertz : float  = None,
            onsource_duration_seconds : float = None,
            crop_duration_seconds : float = None,
            offsource_duration_seconds : float = None,
            num_examples_per_batch : int = None,
            ifos: List[gf.IFO] = None,
            scale_factor : None = None,
            sampling_mode : "gf.SamplingMode" = None,
            steps_per_epoch : int = None,
        ) -> NoiseBatcher:
        """
        The noise of split ``group``, one batch per address.

        WHITE and COLORED noise, and REAL noise drawn at random, is a pure
        function of the batch's streams: REAL batch k of epoch e is batch
        e * ``steps_per_epoch`` + k of the split's passes (``random_batcher``).
        REAL noise read by a transient obtainer is that obtainer's own batcher,
        addressed by index too. PSEUDO_REAL and GRID noise is read sequentially,
        from readers seeded from ``tree`` and ``group``, until it is addressed by
        index.
        ``scale_factor`` is deprecated and ignored.
        """
        ignore_scale_factor(scale_factor)
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if offsource_duration_seconds is None:
            offsource_duration_seconds = gf.Defaults.offsource_duration_seconds
        if crop_duration_seconds is None:
            crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if sampling_mode is None:
            sampling_mode = gf.SamplingMode.RANDOM
        # The detectors for this call only: the obtainer's own `ifos` is never
        # overwritten, so one obtainer can serve datasets with different detectors.
        ifos = gf.src.utils.numerics.ensure_list(ifos) if ifos is not None else list(self.ifos)

        match self.noise_type:
            case NoiseType.WHITE:
                batch = white_noise_batcher(
                    num_examples_per_batch,
                    ifos,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    sample_rate_hertz,
                )

            case NoiseType.COLORED:
                batch = colored_noise_batcher(
                    num_examples_per_batch,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    ifos,
                    sample_rate_hertz,
                    noise_profile=self.noise_profile
                )

            case NoiseType.PSEUDO_REAL:
                batch = sequential_batcher(self.pseudo_real_noise_generator(
                    num_examples_per_batch,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    sample_rate_hertz,
                    group,
                    tree=tree,
                    ifos=ifos
                ))

            case NoiseType.REAL if isinstance(self.ifo_data_obtainer, gf.TransientDataObtainer):
                batch = self.ifo_data_obtainer.batcher(
                    tree,
                    group,
                    sample_rate_hertz=sample_rate_hertz,
                    onsource_duration_seconds=onsource_duration_seconds,
                    crop_duration_seconds=crop_duration_seconds,
                    offsource_duration_seconds=offsource_duration_seconds,
                    num_examples_per_batch=num_examples_per_batch,
                    ifos=ifos,
                    steps_per_epoch=steps_per_epoch,
                )

            case NoiseType.REAL if addressed_by_index(self, sampling_mode):
                batch = self._reader(ifos, tree, group, sample_rate_hertz).random_batcher(
                    tree,
                    WindowSpec.from_params(
                        sample_rate_hertz=sample_rate_hertz,
                        onsource_duration_seconds=onsource_duration_seconds,
                        offsource_duration_seconds=offsource_duration_seconds,
                        crop_duration_seconds=crop_duration_seconds,
                    ),
                    num_examples_per_batch,
                    ifos,
                    steps_per_epoch=steps_per_epoch,
                )

            case NoiseType.REAL:
                ifo_data_obtainer = self._reader(ifos, tree, group, sample_rate_hertz)
                batch = sequential_batcher(ifo_data_obtainer.get_onsource_offsource_chunks(
                    sample_rate_hertz,
                    onsource_duration_seconds,
                    crop_duration_seconds,
                    offsource_duration_seconds,
                    num_examples_per_batch,
                    ifos,
                    seed=sequential_seed(tree, group, "noise/real"),
                    sampling_mode=sampling_mode,
                    group=group
                ))

            case _:
                raise ValueError(
                    f"NoiseType {self.noise_type} not recognised, please choose"
                    "from NoiseType.WHITE, NoiseType.COLORED, "
                    "NoiseType.PSEUDO_REAL, or NoiseType.REAL. "
                )

        def checked(streams: BatchStreams) -> Dict:
            return enforce_shapes(batch(streams), num_ifos=len(ifos))

        # Noise addressed by index can be prepared ahead of a run (``Dataset.prepare``);
        # an evaluation set of transients has a length of its own (``batch.batches``).
        checked.prepare = getattr(batch, "prepare", None)
        checked.batches = getattr(batch, "batches", None)
        return checked

    def _reader(self, ifos, tree : SeedTree, group : str, sample_rate_hertz : float):
        """
        A fresh copy of the IFO data obtainer with the split's segments: a
        copy carries none of the state the original built while serving (see
        BaseDataObtainer._served_state), so no other split's segments reach
        this one, and the caller's obtainer is never changed.
        """
        if not self.ifo_data_obtainer:
            raise ValueError("""
                No IFO obtainer object present. In order to acquire real
                noise please parse a IFOObtainer object to NoiseObtainer
                either during initlisation or through setting
                NoiseObtainer.ifo_data_obtainer
            """)
        reader = deepcopy(self.ifo_data_obtainer)
        reader.get_valid_segments(
            ifos,
            sequential_seed(tree, group, "noise/segment_order"),
            group,
        )
        return reader

    def __call__(
            self,
            sample_rate_hertz : float  = None,
            onsource_duration_seconds : float = None,
            crop_duration_seconds : float = None,
            offsource_duration_seconds : float = None,
            num_examples_per_batch : float = None,
            ifos: List[gf.IFO] = None,
            scale_factor : None = None,
            group : str = None,
            seed : int = None,
            sampling_mode : "gf.SamplingMode" = None
        ) -> Iterator:
        """
        An iterator over the batches k = 0, 1, ... of split ``group`` (epoch
        0; required) of the seed tree rooted at ``seed`` (default
        ``gf.Defaults.seed``).
        ``scale_factor`` is deprecated and ignored.
        """
        ignore_scale_factor(scale_factor)
        if group is None:
            raise ValueError("Name the split: group='train', 'validate' or 'test'.")
        tree = SeedTree.of(seed)
        batch = self.batcher(
            tree,
            group,
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            num_examples_per_batch=num_examples_per_batch,
            ifos=ifos,
            sampling_mode=sampling_mode,
        )
        return _until_exhausted(batch(tree.batch(group, 0, k)) for k in count())

    def pseudo_real_noise_generator(
        self,
        num_examples_per_batch: int,
        onsource_duration_seconds: float,
        crop_duration_seconds : float,
        offsource_duration_seconds: float,
        sample_rate_hertz: float,
        group : str,
        tree : SeedTree,
        ifos : List[gf.IFO] = None
    ):
        """
        Gaussian noise coloured by the PSDs of real segments, read in order,
        in strain. Batch b of the reader draws from the streams of (group,
        epoch 0, b).
        """
        ifos = list(self.ifos) if ifos is None else ifos

        window_spec = WindowSpec.from_params(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds
        )
        num_samples_list = [
            int(duration * sample_rate_hertz) for duration in (
                window_spec.total_onsource_duration_seconds, offsource_duration_seconds
            )
        ]

        reader = self._reader(ifos, tree, group, sample_rate_hertz)
        num_batches_in_segment : int = int(
              reader.max_segment_duration_seconds
            * reader.saturation
            / (num_examples_per_batch * onsource_duration_seconds)
        )

        batch_index = count()
        for segment in reader.acquire(
            sample_rate_hertz,
            reader.valid_segments,
            ifos
        ):
            onsource_asds, offsource_asds = [], []
            for data in segment.data:
                # The PSD of strain (~1e-46) underflows float32: the spectrum
                # is interpolated in its normalised units (the PSD of the
                # segment over its scale), and the scale returns on the ASD.
                spectrum = gf.psd(data, nperseg=1024, sample_rate_hertz=sample_rate_hertz)
                onsource_psd, offsource_psd = interpolate_onsource_offsource_psd(
                    num_samples_list,
                    sample_rate_hertz,
                    spectrum.frequencies,
                    ops.reshape(spectrum.normalised, (-1,)) / sample_rate_hertz
                )
                scale = ops.cast(ops.reshape(spectrum.scale, ()), "complex64")
                onsource_asds.append(ops.sqrt(onsource_psd) * scale)
                offsource_asds.append(ops.sqrt(offsource_psd) * scale)

            onsource_asds = ops.expand_dims(ops.stack(onsource_asds, axis=0), 0)
            offsource_asds = ops.expand_dims(ops.stack(offsource_asds, axis=0), 0)

            for _ in range(num_batches_in_segment):
                streams = tree.batch(group, 0, next(batch_index))
                yield {
                    RV.ONSOURCE: color_noise(
                        unit_normal_noise(streams, ifos, "on", num_examples_per_batch, num_samples_list[0]),
                        onsource_asds,
                        sample_rate_hertz=sample_rate_hertz
                    ),
                    RV.OFFSOURCE: color_noise(
                        unit_normal_noise(streams, ifos, "off", num_examples_per_batch, num_samples_list[1]),
                        offsource_asds,
                        sample_rate_hertz=sample_rate_hertz
                    ),
                }


def addressed_by_index(obtainer, sampling_mode=None) -> bool:
    """
    Whether ``obtainer``'s batches are a pure function of their address: simulated
    noise (WHITE, COLORED), transients, bare or read as REAL noise (their own
    ``batcher``), and real noise drawn at random (``SamplingMode.RANDOM``) through a
    noise obtainer (``random_batcher``). The rest is read in order.
    """
    if not reads_real_data(obtainer):
        return True
    real = isinstance(obtainer, NoiseObtainer) and obtainer.noise_type == NoiseType.REAL
    reader = obtainer.ifo_data_obtainer if real else obtainer
    if isinstance(reader, gf.TransientDataObtainer):
        return True
    return (real and sampling_mode in (None, gf.SamplingMode.RANDOM)
            and isinstance(reader, gf.NoiseDataObtainer))


def reads_real_data(obtainer) -> bool:
    """
    Whether ``obtainer`` reads real data. Every obtainer does but simulated noise
    (WHITE, COLORED), which is made from its batch's streams alone; None, the
    default obtainer, is real noise, and a reader of the caller's own is taken to
    read real data. Data that reads none is split-neutral (``split_spec_of``)
    and a pure function of its address; real data is read in order, under a split.
    """
    return not (isinstance(obtainer, NoiseObtainer) and obtainer.noise_type in (NoiseType.WHITE, NoiseType.COLORED))


def obtainer_batcher(obtainer, tree: SeedTree, group: str, **kwargs) -> NoiseBatcher:
    """
    The batcher of any obtainer, its own: a NoiseObtainer's, or a transient
    obtainer's, addressed by index. The tree orders and balances the split; the
    split's members never depend on it. A reader of the caller's own (no
    ``batcher``) is read in order from a fresh copy, seeded from its own stream.
    """
    if hasattr(obtainer, "batcher"):
        return obtainer.batcher(tree, group, **kwargs)
    kwargs.pop("sampling_mode", None)
    kwargs.pop("steps_per_epoch", None)
    reader = deepcopy(obtainer)
    return sequential_batcher(reader(**kwargs, group=group, seed=sequential_seed(tree, group, "noise/reader")))
