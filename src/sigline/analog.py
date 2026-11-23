"""Mixing and analog modulation: AM, FM, PM and SSB, as subsystems.

Each function wires library blocks and returns them as a :class:`Subsystem`
with input ``x`` and output ``y``. Inner blocks have names, so parameters
stay attributes (``mod.index.g = 0.8``) and blocks can be replaced.

Modulators turn a real message into complex baseband; for a real passband
signal follow with ``mixer(fc, "exp")`` and take the real part.
Demodulators take complex baseband and can low-pass it first to
``channel_width``. Functions that need the sample rate to set up a block (a
filter, a scale) take ``fs``; any of them takes it to be called alone.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig

from ._format import si
from .blocks import Delay, FuncChunk, Gain, Product, Sum, Unwrap
from .filters import FIR, IIR
from .sources import Const, Sine
from .subsystem import Subsystem

__all__ = [
    "am_demod",
    "am_mod",
    "fm_demod",
    "fm_mod",
    "hilbert_taps",
    "mixer",
    "pm_demod",
    "pm_mod",
    "ssb_demod",
    "ssb_mod",
]

# lo kind -> (frequency sign, phase shift / rad, complex)
_LO = {
    "cos": (1, 0.0, False),
    "sin": (1, -np.pi / 2, False),
    "exp": (1, 0.0, True),
    "exp-": (-1, 0.0, True),
}


def mixer(f, lo="cos", phase=0.0, *, fs=None, name=None):
    """Multiply by a phase-continuous local oscillator: ``Sine`` and ``Product``.

    Parameters
    ----------
    f : float
        LO frequency / Hz. Retune with ``mix.lo.f``; the phase stays
        continuous.
    lo : {"cos", "sin", "exp", "exp-"}, default "cos"
        ``exp`` is ``e^{+j phi}`` (up-conversion); ``exp-`` is an LO at
        ``-f``, ``e^{-j phi}`` (down-conversion to complex baseband), so its
        ``lo.f`` is negative.
    phase : float, default 0.0
        Start phase / rad.
    fs : float, optional
        Sample rate / Hz, needed only to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``lo`` (``Sine``), ``mix``
        (``Product``).

    Raises
    ------
    ValueError
        Unknown ``lo``.

    Examples
    --------
    >>> import numpy as np
    >>> import sigline as sl
    >>> mix = sl.mixer(0.25, "exp", fs=1.0)
    >>> np.round(mix(np.ones(4)), 12)
    array([ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j])
    """
    if lo not in _LO:
        raise ValueError(f"lo must be one of {sorted(_LO)}, not {lo!r}")
    sign, shift, cplx = _LO[lo]
    osc = Sine(sign * f, phase=sign * phase + shift, complex=cplx, fs=fs, name="lo")
    prod = Product(fs=fs, name="mix")
    osc >> prod["x1"]
    return Subsystem(
        inputs={"x": prod["x0"]},
        outputs={"y": prod},
        name=name,
        info=lambda: f"{si(sign * osc.f, 'Hz')} {lo}",
    )


def expj(p):
    """Return ``exp(j p)``."""
    return np.exp(1j * p)


def phase_step(z, prev):
    """Return the phase from ``prev`` to ``z``: ``angle(z conj(prev))``."""
    return np.angle(z * np.conj(prev))


def _width(channel_width):
    return f", {si(channel_width, 'Hz')}" if channel_width else ""


def _channel(channel_width, taps, fs):
    # low-pass to a two-sided width / Hz, or None
    if channel_width is None:
        return None
    if fs is None:
        raise ValueError("channel_width needs fs")
    if not 0 < channel_width / 2 < fs / 2:
        raise ValueError(f"channel_width must be between 0 and fs = {fs:g} Hz")
    return FIR(sig.firwin(int(taps), channel_width / 2, fs=fs), fs=fs, name="channel")


def am_mod(index=0.5, carrier=True, *, fs=None, name=None):
    """Amplitude modulation: complex baseband ``1 + index x`` (real valued).

    Parameters
    ----------
    index : float, default 0.5
        Modulation index for a message within ``[-1, 1]``.
    carrier : bool, default True
        ``False`` gives double sideband with suppressed carrier,
        ``index * x``.
    fs : float, optional
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``index`` (``Gain``); with carrier also ``carrier``
        (``Const``) and ``sum``.
    """
    gain = Gain(index, fs=fs, name="index")
    if not carrier:
        return Subsystem(
            inputs={"x": gain},
            outputs={"y": gain},
            name=name,
            info=lambda: f"m={gain.g:g} DSB-SC",
        )
    add = Sum(fs=fs, name="sum")
    Const(1.0, fs=fs, name="carrier") >> add["x0"]
    gain >> add["x1"]
    return Subsystem(
        inputs={"x": gain}, outputs={"y": add}, name=name, info=lambda: f"m={gain.g:g}"
    )


def am_demod(
    index=0.5,
    carrier=True,
    method="envelope",
    channel_width=None,
    taps=101,
    *,
    fs=None,
    name=None,
):
    """Amplitude demodulation of complex baseband.

    ``envelope`` takes ``|z|`` (needs the carrier; phase does not matter);
    ``coherent`` takes ``Re(z)`` (needs carrier phase 0, works without
    carrier). With a carrier, it is removed and the index divided out.

    Parameters
    ----------
    index : float, default 0.5
    carrier : bool, default True
    method : {"envelope", "coherent"}, default "envelope"
    channel_width : float, optional
        Low-pass to this two-sided width / Hz first; needs ``fs``.
    taps : int, default 101
        Of the channel filter.
    fs : float, optional
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``channel`` (``FIR``, if any), ``detector`` (``FuncChunk``),
        ``carrier`` and ``sum`` (with carrier), ``scale`` (``Gain``).

    Raises
    ------
    ValueError
        Unknown method, envelope detection without carrier, or a channel
        width without ``fs`` or outside ``0 .. fs``.
    """
    if method not in ("envelope", "coherent"):
        raise ValueError("method must be 'envelope' or 'coherent'")
    if method == "envelope" and not carrier:
        raise ValueError("envelope detection needs the carrier; use 'coherent'")
    chan = _channel(channel_width, taps, fs)
    det = FuncChunk(np.abs if method == "envelope" else np.real, fs=fs, name="detector")
    scale = Gain(1 / index, fs=fs, name="scale")
    if carrier:
        remove = Sum("+-", fs=fs, name="sum")
        det >> remove["x0"]
        Const(1.0, fs=fs, name="carrier") >> remove["x1"]
        remove >> scale
    else:
        det >> scale
    if chan is not None:
        chan >> det
    return Subsystem(
        inputs={"x": det if chan is None else chan},
        outputs={"y": scale},
        name=name,
        info=lambda: method + _width(channel_width),
    )


def fm_mod(deviation, *, fs, name=None):
    """Frequency modulation: ``exp(j phi)``, ``phi`` the integral of the message.

    ``phi[n] = phi[n-1] + 2 pi deviation x[n-1] / fs``: a message of 1
    shifts the frequency by ``deviation``. The phase at a sample integrates
    the message held over the samples before, exactly, as in the continuous
    blocks; the output follows the message by one sample.

    Parameters
    ----------
    deviation : float
        Peak frequency deviation / Hz for ``|x| = 1``.
    fs : float
        Sample rate / Hz, for the scale.
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``scale`` (``Gain``, ``2 pi deviation / fs``), ``phase``
        (``IIR`` accumulator), ``exp`` (``FuncChunk``).
    """
    scale = Gain(2 * np.pi * deviation / fs, fs=fs, name="scale")
    phase = IIR(b=[0.0, 1.0], a=[1.0, -1.0], fs=fs, name="phase")
    exp = FuncChunk(expj, fs=fs, name="exp")
    scale >> phase >> exp
    return Subsystem(
        inputs={"x": scale},
        outputs={"y": exp},
        name=name,
        info=lambda: f"Δf={si(scale.g * fs / (2 * np.pi), 'Hz')}",
    )


def fm_demod(deviation, channel_width=None, taps=101, *, fs, name=None):
    """Frequency demodulation: quadrature discriminator.

    ``y[n] = angle(z[n] conj(z[n-1])) fs / (2 pi deviation)``, so
    ``fm_demod`` after ``fm_mod`` returns the message, one sample later,
    while the phase step stays below pi.

    Parameters
    ----------
    deviation : float
        Frequency / Hz that maps to an output of 1.
    channel_width : float, optional
        Low-pass to this two-sided width / Hz first. Carson's rule,
        ``2 (deviation + highest message frequency)``, holds 98 % of the
        power; leave a margin above it, or clipped sidebands distort.
    taps : int, default 101
        Of the channel filter.
    fs : float
        Sample rate / Hz, for the scale.
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``channel`` (``FIR``, if any), ``previous`` (``Delay``),
        ``discriminator`` (``FuncChunk``), ``scale`` (``Gain``).
    """
    chan = _channel(channel_width, taps, fs)
    prev = Delay(1, init=1.0, fs=fs, name="previous")
    disc = FuncChunk(phase_step, fs=fs, name="discriminator")
    scale = Gain(fs / (2 * np.pi * deviation), fs=fs, name="scale")
    prev >> disc["prev"]
    disc >> scale
    if chan is not None:
        chan >> disc["z"]
        chan >> prev
    return Subsystem(
        inputs={"x": [disc["z"], prev] if chan is None else chan},
        outputs={"y": scale},
        name=name,
        info=lambda: (
            f"Δf={si(fs / (2 * np.pi * scale.g), 'Hz')}" + _width(channel_width)
        ),
    )


def pm_mod(index, *, fs=None, name=None):
    """Phase modulation: ``exp(j index x)``.

    Parameters
    ----------
    index : float
        Phase / rad for ``x = 1``.
    fs : float, optional
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``index`` (``Gain``), ``exp`` (``FuncChunk``).
    """
    gain = Gain(index, fs=fs, name="index")
    exp = FuncChunk(expj, fs=fs, name="exp")
    gain >> exp
    return Subsystem(
        inputs={"x": gain},
        outputs={"y": exp},
        name=name,
        info=lambda: f"{gain.g:g} rad",
    )


def pm_demod(index, channel_width=None, taps=101, *, fs=None, name=None):
    """Phase demodulation, unwrapped across chunks.

    The phase may wrap many times; each step must stay below pi. With a
    channel filter the first ``taps - 1`` samples are its start-up
    transient: they stay wrapped, so the transient cannot leave a 2 pi
    offset.

    Parameters
    ----------
    index : float
        Phase / rad that maps to an output of 1.
    channel_width : float, optional
        Low-pass to this two-sided width / Hz first; needs ``fs``.
    taps : int, default 101
    fs : float, optional
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``channel`` (``FIR``, if any), ``angle`` (``FuncChunk``),
        ``unwrap`` (``Unwrap``), ``scale`` (``Gain``).
    """
    chan = _channel(channel_width, taps, fs)
    angle = FuncChunk(np.angle, fs=fs, name="angle")
    skip = int(taps) - 1 if chan is not None else 0
    scale = Gain(1 / index, fs=fs, name="scale")
    angle >> Unwrap(skip=skip, fs=fs, name="unwrap") >> scale
    if chan is not None:
        chan >> angle
    return Subsystem(
        inputs={"x": angle if chan is None else chan},
        outputs={"y": scale},
        name=name,
        info=lambda: f"{1 / scale.g:g} rad" + _width(channel_width),
    )


def hilbert_taps(n=101):
    """Return a Hamming-windowed FIR Hilbert transformer with ``n`` taps.

    Parameters
    ----------
    n : int, default 101
        Odd; the group delay is ``(n - 1) / 2`` samples.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        Even ``n``.
    """
    if n % 2 == 0:
        raise ValueError("a Hilbert transformer needs an odd number of taps")
    k = np.arange(n) - (n - 1) // 2
    h = np.zeros(n)
    odd = k % 2 != 0
    h[odd] = 2 / (np.pi * k[odd])
    return h * np.hamming(n)


def _check_side(side):
    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'")


def ssb_mod(side="upper", taps=101, *, fs=None, name=None):
    """Single sideband: ``x + j H{x}`` (upper) or ``x - j H{x}`` (lower).

    ``H`` is an FIR Hilbert transformer; the real part is delayed by the same
    ``(taps - 1) / 2`` samples.

    Parameters
    ----------
    side : {"upper", "lower"}, default "upper"
    taps : int, default 101
        Odd; more taps suppress the other sideband better near DC.
    fs : float, optional
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``delay`` (``Delay``), ``hilbert`` (``FIR``), ``j``
        (``Gain``, ``+-1j``), ``sum``.

    Raises
    ------
    ValueError
        Unknown side, or even ``taps``.
    """
    _check_side(side)
    taps = int(taps)
    delay = Delay((taps - 1) // 2, fs=fs, name="delay")
    hilbert = FIR(hilbert_taps(taps), fs=fs, name="hilbert")
    j = Gain(1j if side == "upper" else -1j, fs=fs, name="j")
    add = Sum(fs=fs, name="sum")
    delay >> add["x0"]
    hilbert >> j >> add["x1"]
    return Subsystem(
        inputs={"x": [delay, hilbert]},
        outputs={"y": add},
        name=name,
        info=lambda: "USB" if np.imag(j.g) > 0 else "LSB",
    )


def ssb_demod(side="upper", channel_width=None, taps=101, *, fs=None, name=None):
    """Single sideband demodulation: one-sided channel filter, then ``Re(z)``.

    Parameters
    ----------
    side : {"upper", "lower"}, default "upper"
    channel_width : float, optional
        Pass ``0 .. channel_width`` / Hz (upper) or ``-channel_width .. 0``
        (lower), which rejects the other sideband; needs ``fs``.
    taps : int, default 101
    fs : float, optional
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``channel`` (complex band-pass ``FIR``, if any), ``real``
        (``FuncChunk``).

    Raises
    ------
    ValueError
        Unknown side, or a channel width without ``fs``.
    """
    _check_side(side)
    real = FuncChunk(np.real, fs=fs, name="real")
    chan = None
    if channel_width is not None:
        if fs is None:
            raise ValueError("channel_width needs fs")
        half = channel_width / 2
        k = np.arange(taps) - (taps - 1) / 2
        sign = 1.0 if side == "upper" else -1.0
        lp = sig.firwin(int(taps), half, fs=fs)
        bp = lp * np.exp(2j * np.pi * sign * half * k / fs)
        chan = FIR(bp, fs=fs, name="channel")
        chan >> real
    band = "USB" if side == "upper" else "LSB"
    return Subsystem(
        inputs={"x": real if chan is None else chan},
        outputs={"y": real},
        name=name,
        info=lambda: band + _width(channel_width),
    )
