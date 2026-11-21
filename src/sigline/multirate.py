"""Rate-changing blocks.

Each sets ``resampling_ratio`` (output rate / input rate); the diagram gives
every block its own ``fs`` and chunk length. All keep their state and sample
phase across chunks, so chunking never changes results.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .core import Block

__all__ = ["Repeat", "Upsample"]


def _zeros(x, n):
    return np.zeros((n,) + x.shape[1:], dtype=x.dtype)


class Upsample(Block):
    """Insert ``L - 1`` zeros after every sample (no filtering).

    Parameters
    ----------
    L : int
        Factor.
    phase : int, default 0
        Output position of each input sample within its group of ``L``.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Upsample
    >>> Upsample(3)(np.array([1.0, 2.0]))
    array([1., 0., 0., 2., 0., 0.])
    """

    def __init__(self, L, phase=0, **kw):
        self.L, self.phase = int(L), int(phase)
        self.resampling_ratio = Fraction(self.L)
        super().__init__(**kw)

    def step(self, x):
        """Return the zero-stuffed chunk."""
        y = _zeros(x, len(x) * self.L)
        y[self.phase :: self.L] = x
        return y

    def info(self):
        """Return the factor."""
        return f"↑{self.L}"


class Repeat(Block):
    """Hold every sample for ``L`` output samples (zero-order hold).

    Parameters
    ----------
    L : int
    """

    def __init__(self, L, **kw):
        self.L = int(L)
        self.resampling_ratio = Fraction(self.L)
        super().__init__(**kw)

    def step(self, x):
        """Return each sample ``L`` times."""
        return np.repeat(x, self.L, axis=0)

    def info(self):
        """Return the factor."""
        return f"hold ×{self.L}"
