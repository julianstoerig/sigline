"""Digital modulation: bits, constellations, mapping, pulse shaping, FSK.

Rate-changing steps are separate blocks, so every chunk maps cleanly:

    bits >> Mapper(qam(16)) >> PulseShaper(8)                 # transmitter
    rx >> MatchedFilter(8) >> Downsample(8) >> Demapper(qam(16))  # receiver
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .core import Block

__all__ = ["BitsToWords", "WordsToBits"]


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
