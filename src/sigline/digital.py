"""Digital modulation: bits, constellations, mapping, pulse shaping, FSK.

Rate-changing steps are separate blocks, so every chunk maps cleanly. The
mapper, demapper, pulse shaper, matched filter and FSK modulator are
functions returning subsystems of library blocks::

    bits >> mapper(qam(16)) >> pulse_shaper(8)                  # transmitter
    rx >> matched_filter(8) >> Downsample(8) >> demapper(qam(16))  # receiver
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .analog import fm_mod
from .blocks import Func
from .core import Block
from .filters import FIR
from .multirate import Interpolate, Repeat
from .subsystem import Subsystem

__all__ = [
    "BitsToWords",
    "FSKDemod",
    "WordsToBits",
    "demapper",
    "fsk_mod",
    "mapper",
    "matched_filter",
    "nearest",
    "pulse_shaper",
    "pam",
    "psk",
    "qam",
    "rrc_taps",
]


def _to_bits(words, k):
    # integer words to k bits each, most significant first
    sh = np.arange(k - 1, -1, -1)
    return ((np.asarray(words, np.int64)[:, None] >> sh) & 1).reshape(-1).astype(float)


def _to_words(bits, k):
    # bits (thresholded at 0.5) to words of k bits, most significant first
    b = (np.real(np.asarray(bits)) > 0.5).astype(np.int64).reshape(-1, k)
    return (b << np.arange(k - 1, -1, -1)).sum(axis=1)


class WordsToBits(Block):
    """Integer words to bits, ``bits`` per word, most significant first.

    Parameters
    ----------
    bits : int

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import WordsToBits
    >>> WordsToBits(3)(np.array([5, 2]))
    array([1., 0., 1., 0., 1., 0.])
    """

    def __init__(self, bits, **kw):
        self.bits = int(bits)
        self.resampling_ratio = Fraction(self.bits)
        super().__init__(**kw)

    def step(self, x):
        """Return the bits of each word."""
        return _to_bits(np.rint(np.real(np.asarray(x))), self.bits)

    def info(self):
        """Return bits per word."""
        return f"{self.bits} bits/word"


class BitsToWords(Block):
    """Bits (thresholded at 0.5) to integer words, most significant first.

    Parameters
    ----------
    bits : int
    """

    def __init__(self, bits, **kw):
        self.bits = int(bits)
        self.resampling_ratio = Fraction(1, self.bits)
        super().__init__(**kw)

    def step(self, x):
        """Return one word per ``bits`` bits."""
        return _to_words(x, self.bits).astype(float)

    def info(self):
        """Return bits per word."""
        return f"{self.bits} bits/word"


def _bits_per_symbol(M):
    k = int(round(np.log2(M)))
    if M < 2 or 2**k != M:
        raise ValueError(f"M must be a power of 2, not {M}")
    return k


def _gray_order(values):
    # word p ^ (p >> 1) gets values[p]: neighbours in `values` differ by one bit
    out = np.empty(len(values), dtype=np.asarray(values).dtype)
    p = np.arange(len(values))
    out[p ^ (p >> 1)] = values
    return out


def _unit_energy(points):
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def nearest(y, points):
    """Return the index of the nearest point for each sample.

    Parameters
    ----------
    y : array_like
        Samples.
    points : array_like
        Constellation.

    Returns
    -------
    numpy.ndarray
        Integer indices into ``points``.

    Examples
    --------
    >>> from sigline.digital import nearest
    >>> nearest([0.9, -1.2, 0.1], [-1.0, 1.0])
    array([1, 0, 1])
    """
    y, points = np.asarray(y), np.asarray(points)
    out = np.empty(len(y), dtype=np.int64)
    for i in range(0, len(y), 8192):  # bounded memory
        out[i : i + 8192] = np.argmin(
            np.abs(y[i : i + 8192, None] - points[None, :]), axis=1
        )
    return out


def pam(M):
    """Return Gray-coded M-PAM levels with unit average energy.

    ``pam(M)[w]`` is the level of word ``w``.

    Parameters
    ----------
    M : int
        Power of 2.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> from sigline import pam
    >>> pam(4) * np.sqrt(5)
    array([-3., -1.,  3.,  1.])
    """
    _bits_per_symbol(M)
    return _unit_energy(_gray_order(2.0 * np.arange(M) - M + 1))


def psk(M):
    """Return Gray-coded M-PSK points on the unit circle.

    BPSK is ``[1, -1]``; QPSK sits at odd multiples of 45 degrees.

    Parameters
    ----------
    M : int
        Power of 2.

    Returns
    -------
    numpy.ndarray
    """
    _bits_per_symbol(M)
    offset = np.pi / 4 if M == 4 else 0.0
    points = np.exp(1j * (2 * np.pi * np.arange(M) / M + offset))
    points[np.abs(points.imag) < 1e-15] = points[np.abs(points.imag) < 1e-15].real
    return _gray_order(points)


def qam(M):
    """Return Gray-coded square M-QAM points with unit average energy.

    The upper half of each word's bits selects the in-phase level, the lower
    half the quadrature level, each Gray coded.

    Parameters
    ----------
    M : int
        4, 16, 64, 256, ...

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        ``M`` is not an even power of 2.
    """
    k = _bits_per_symbol(M)
    if k % 2:
        raise ValueError(f"square QAM needs an even number of bits, not M = {M}")
    side = _gray_order(2.0 * np.arange(2 ** (k // 2)) - 2 ** (k // 2) + 1)
    w = np.arange(M)
    return _unit_energy(side[w >> (k // 2)] + 1j * side[w & (2 ** (k // 2) - 1)])


def table(w, points=()):
    """Return ``points[w]``: the point of each word."""
    return np.asarray(points)[np.rint(np.real(w)).astype(np.int64)]


def decide(y, points=()):
    """Return the word of the nearest point to each sample."""
    return nearest(y, points).astype(float)


def mapper(points, *, fs=None, name=None):
    """Bits to symbols: each ``log2(M)`` bits, MSB first, pick a point.

    Parameters
    ----------
    points : array_like
        Constellation; ``points[w]`` is the symbol of word ``w``, e.g.
        ``qam(16)``.
    fs : float, optional
        Bit rate / Hz.
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``words`` (``BitsToWords``) and ``table`` (``Func``, its
        ``params["points"]`` the constellation).
    """
    points = np.asarray(points)
    k = _bits_per_symbol(len(points))
    words = BitsToWords(k, fs=fs, name="words")
    lookup = Func(
        table,
        params={"points": points},
        fs=None if fs is None else fs / k,
        name="table",
    )
    words >> lookup
    return Subsystem(
        inputs={"x": words},
        outputs={"y": lookup},
        name=name,
        info=lambda: f"{len(lookup.params['points'])} points",
    )


def demapper(points, *, fs=None, name=None):
    """Symbols to bits: nearest point (hard decision), its word's bits.

    Parameters
    ----------
    points : array_like
        The constellation of the :func:`mapper`.
    fs : float, optional
        Symbol rate / Hz.
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``decide`` (``Func``, its ``params["points"]`` the
        constellation) and ``bits`` (``WordsToBits``).
    """
    points = np.asarray(points)
    k = _bits_per_symbol(len(points))
    choose = Func(decide, params={"points": points}, fs=fs, name="decide")
    bits = WordsToBits(k, fs=fs, name="bits")
    choose >> bits
    return Subsystem(
        inputs={"x": choose},
        outputs={"y": bits},
        name=name,
        info=lambda: f"{len(choose.params['points'])} points",
    )


def rrc_taps(beta=0.35, sps=8, span=10):
    """Return root-raised-cosine taps with unit energy.

    Two of them in a row (transmit and matched filter) form a raised cosine,
    a Nyquist pulse: no intersymbol interference at the symbol instants.

    Parameters
    ----------
    beta : float, default 0.35
        Roll-off, ``0 < beta <= 1``; bandwidth ``(1 + beta) Rs / 2``.
    sps : int, default 8
        Samples per symbol.
    span : int, default 10
        Length in symbols; ``span * sps + 1`` taps, delay ``span / 2`` symbols.

    Returns
    -------
    numpy.ndarray
    """
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (
            np.sin(np.pi * t * (1 - beta))
            + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
        ) / (np.pi * t * (1 - (4 * beta * t) ** 2))
    h[t == 0] = 1 - beta + 4 * beta / np.pi
    edge = np.isclose(np.abs(t), 1 / (4 * beta))
    a = np.pi / (4 * beta)
    h[edge] = (
        beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(a) + (1 - 2 / np.pi) * np.cos(a))
    )
    return h / np.sqrt(np.sum(h**2))


def pulse_shaper(sps, beta=0.35, span=10, *, fs=None, name=None):
    """Symbols to samples: upsample by ``sps`` and filter with an RRC.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    beta : float, default 0.35
    span : int, default 10
    fs : float, optional
        Symbol rate / Hz.
    name : str, optional

    Returns
    -------
    Subsystem
        Member ``interpolate`` (``Interpolate`` with :func:`rrc_taps`).
    """
    up = Interpolate(sps, taps=rrc_taps(beta, sps, span), fs=fs, name="interpolate")
    return Subsystem(
        inputs={"x": up},
        outputs={"y": up},
        name=name,
        info=lambda: f"{up.L} sps, RRC β={beta:g}",
    )


def matched_filter(sps, beta=0.35, span=10, *, fs=None, name=None):
    """RRC matched filter; follow with ``Downsample(sps)`` for the symbols.

    After :func:`pulse_shaper`, the symbols come back ``span`` symbols
    later, at sample phase 0.

    Parameters
    ----------
    sps : int
    beta : float, default 0.35
    span : int, default 10
    fs : float, optional
        Sample rate / Hz.
    name : str, optional

    Returns
    -------
    Subsystem
        Member ``fir`` (``FIR`` with :func:`rrc_taps`).
    """
    fir = FIR(rrc_taps(beta, sps, span), fs=fs, name="fir")
    return Subsystem(
        inputs={"x": fir},
        outputs={"y": fir},
        name=name,
        info=lambda: f"{sps} sps, RRC β={beta:g}",
    )


def _fsk_tones(M, h, Rs):
    # tone frequencies / Hz, spaced h Rs, centred on 0
    return (np.arange(M) - (M - 1) / 2) * h * Rs


def level(m, M=2):
    """Return the tone level ``m - (M - 1) / 2`` of FSK symbol ``m``."""
    return np.clip(np.rint(np.real(m)), 0, M - 1) - (M - 1) / 2


def fsk_mod(M=2, sps=8, h=1.0, *, fs, name=None):
    """Continuous-phase FSK: FM of the symbols held for ``sps`` samples.

    Symbol ``m`` sends a tone at ``(m - (M-1)/2) h Rs``. Input: symbol
    indices ``0 .. M-1`` at the symbol rate ``Rs`` (bits for ``M = 2``; else
    ``BitsToWords``). Output: complex baseband, ``sps`` samples per symbol.
    ``h = 0.5`` is MSK.

    Parameters
    ----------
    M : int, default 2
    sps : int, default 8
    h : float, default 1.0
        Modulation index; tones ``h Rs`` apart (1: orthogonal for
        noncoherent detection).
    fs : float
        Symbol rate / Hz, the input rate.
    name : str, optional

    Returns
    -------
    Subsystem
        Members ``level`` (``Func``), ``hold`` (``Repeat``) and an
        :func:`fm_mod` with deviation ``h Rs``.
    """
    lev = Func(level, params={"M": int(M)}, fs=fs, name="level")
    hold = Repeat(sps, fs=fs, name="hold")
    fm = fm_mod(h * fs, fs=fs * sps, name="fm")
    lev >> hold >> fm
    return Subsystem(
        inputs={"x": lev},
        outputs={"y": fm},
        name=name,
        info=lambda: f"{M}-FSK, h={h:g}",
    )


class FSKDemod(Block):
    """Noncoherent FSK detection: per symbol, the tone with the most energy.

    Symbols start at sample ``offset`` and every ``sps`` samples after.

    Parameters
    ----------
    M, sps, h :
        As for :func:`fsk_mod`.
    offset : int, default 0
    """

    def __init__(self, M=2, sps=8, h=1.0, offset=0, **kw):
        self.M, self.sps, self.h, self.offset = int(M), int(sps), h, int(offset)
        self.resampling_ratio = Fraction(1, self.sps)
        super().__init__(**kw)

    def reset(self):
        """Clear the partial symbol."""
        self._buf = None

    def step(self, x):
        """Return one symbol index per ``sps`` samples."""
        n_out = len(x) // self.sps
        buf = x if self._buf is None else np.concatenate([self._buf, x])
        if self._buf is None:
            buf = np.concatenate([np.zeros(self.offset, complex), buf])  # align
        k = n_out * self.sps
        sym, self._buf = buf[:k].reshape(n_out, self.sps), buf[k:]
        n = np.arange(self.sps)
        f = _fsk_tones(self.M, self.h, self.fs / self.sps)
        ref = np.exp(-2j * np.pi * f[:, None] * n[None, :] / self.fs)
        return np.argmax(np.abs(sym @ ref.T), axis=1).astype(float)

    def info(self):
        """Return order and index."""
        return f"{self.M}-FSK, h={self.h:g}"
