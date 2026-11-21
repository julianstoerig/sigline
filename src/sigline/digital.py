"""Digital modulation: bits, constellations, mapping, pulse shaping, FSK.

Rate-changing steps are separate blocks, so every chunk maps cleanly:

    bits >> Mapper(qam(16)) >> PulseShaper(8)                 # transmitter
    rx >> MatchedFilter(8) >> Downsample(8) >> Demapper(qam(16))  # receiver
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .core import Block
from .filters import FIR
from .multirate import Interpolate

__all__ = [
    "BitsToWords",
    "Demapper",
    "Mapper",
    "MatchedFilter",
    "PulseShaper",
    "WordsToBits",
    "pam",
    "psk",
    "qam",
    "rrc_taps",
]


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
        w = np.rint(np.real(np.asarray(x))).astype(np.int64)
        sh = np.arange(self.bits - 1, -1, -1)
        return ((w[:, None] >> sh[None, :]) & 1).reshape(-1).astype(float)

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
        b = (np.real(np.asarray(x)) > 0.5).astype(np.int64).reshape(-1, self.bits)
        return (b << np.arange(self.bits - 1, -1, -1)).sum(axis=1).astype(float)

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


class Mapper(Block):
    """Bits to symbols: each ``log2(M)`` bits, MSB first, pick a point.

    Parameters
    ----------
    points : array_like
        Constellation; ``points[w]`` is the symbol of word ``w``, e.g.
        ``qam(16)``.
    """

    def __init__(self, points, **kw):
        self.points = np.asarray(points)
        self.k = _bits_per_symbol(len(self.points))
        self.resampling_ratio = Fraction(1, self.k)
        super().__init__(**kw)

    def step(self, x):
        """Return one symbol per ``k`` bits."""
        b = (np.real(np.asarray(x)) > 0.5).astype(np.int64).reshape(-1, self.k)
        return self.points[(b << np.arange(self.k - 1, -1, -1)).sum(axis=1)]

    def info(self):
        """Return the number of points."""
        return f"{len(self.points)} points"


class Demapper(Block):
    """Symbols to bits: nearest point (hard decision), its word's bits.

    Parameters
    ----------
    points : array_like
        The constellation of the :class:`Mapper`.
    """

    def __init__(self, points, **kw):
        self.points = np.asarray(points)
        self.k = _bits_per_symbol(len(self.points))
        self.resampling_ratio = Fraction(self.k)
        super().__init__(**kw)

    def step(self, x):
        """Return ``k`` bits per symbol."""
        y = np.asarray(x)
        w = np.empty(len(y), dtype=np.int64)
        for i in range(0, len(y), 8192):  # bounded memory
            yy = y[i : i + 8192]
            w[i : i + 8192] = np.argmin(
                np.abs(yy[:, None] - self.points[None, :]), axis=1
            )
        sh = np.arange(self.k - 1, -1, -1)
        return ((w[:, None] >> sh[None, :]) & 1).reshape(-1).astype(float)

    def info(self):
        """Return the number of points."""
        return f"{len(self.points)} points"


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


class PulseShaper(Interpolate):
    """Symbols to samples: upsample by ``sps`` and filter with an RRC.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    beta : float, default 0.35
    span : int, default 10
    """

    def __init__(self, sps, beta=0.35, span=10, **kw):
        self.beta, self.span = beta, int(span)
        super().__init__(sps, taps=rrc_taps(beta, sps, span), **kw)

    def info(self):
        """Return samples per symbol and roll-off."""
        return f"{self.L} sps, RRC β={self.beta:g}"


class MatchedFilter(FIR):
    """RRC matched filter; follow with ``Downsample(sps)`` for the symbols.

    With :class:`PulseShaper` before it, the symbols come back ``span``
    symbols later, at sample phase 0.

    Parameters
    ----------
    sps : int
    beta : float, default 0.35
    span : int, default 10
    """

    def __init__(self, sps, beta=0.35, span=10, **kw):
        self.sps, self.beta, self.span = int(sps), beta, int(span)
        super().__init__(rrc_taps(beta, sps, span), **kw)

    def info(self):
        """Return samples per symbol and roll-off."""
        return f"{self.sps} sps, RRC β={self.beta:g}"
