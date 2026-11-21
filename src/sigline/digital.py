"""Digital modulation: bits, constellations, mapping, pulse shaping, FSK.

Rate-changing steps are separate blocks, so every chunk maps cleanly:

    bits >> Mapper(qam(16)) >> PulseShaper(8)                 # transmitter
    rx >> MatchedFilter(8) >> Downsample(8) >> Demapper(qam(16))  # receiver
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .core import Block

__all__ = ["BitsToWords", "WordsToBits", "pam", "psk", "qam"]


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
