"""Rate-changing blocks.

Each sets ``resampling_ratio`` (output rate / input rate); the diagram gives
every block its own ``fs`` and chunk length. All keep their state and sample
phase across chunks, so chunking never changes results.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.signal as sig

from . import _numba
from .core import Block

__all__ = ["Downsample", "Interpolate", "Repeat", "Upsample"]


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


class Downsample(Block):
    """Keep every ``M``-th sample (no filtering); phase kept across chunks.

    Parameters
    ----------
    M : int
    phase : int, default 0
        Which sample of each group of ``M`` is kept.
    """

    def __init__(self, M, phase=0, **kw):
        self.M, self.phase = int(M), int(phase)
        self.resampling_ratio = Fraction(1, self.M)
        super().__init__(**kw)

    def reset(self):
        """Restart the sample count."""
        self.k = 0

    def step(self, x):
        """Return the kept samples."""
        first = (self.phase - self.k) % self.M
        self.k += len(x)
        return x[first :: self.M]

    def info(self):
        """Return the factor."""
        return f"↓{self.M}"


@_numba.njit
def _poly_kernel(ext, h, up, down, m0, n_start, y):
    # output m is sample j = m * down of the zero-stuffed, filtered input:
    # y[m] = sum_q h[p + q up] x[j // up - q], p = j % up; ext holds input
    # samples with global indices n_start, n_start + 1, ... (history first)
    for i in range(y.size):
        j = (m0 + i) * down
        n_hi = j // up
        t = j - n_hi * up
        e = n_hi - n_start
        acc = y[i] * 0
        while t < h.size:
            acc += h[t] * ext[e]
            t += up
            e -= 1
        y[i] = acc


class _Polyphase(Block):
    """Streaming polyphase FIR resampling by ``up / down``.

    Equals zero-stuffing by ``up``, FIR filtering with ``taps`` and keeping
    every ``down``-th sample, but computes only the kept samples (numba).
    Without numba, or for vector signals, it runs that reference directly.
    """

    up = 1
    down = 1

    def reset(self):
        """Clear history and counters."""
        self.hist = None
        self.n_in = self.m_out = 0
        self.zi, self.k = None, 0  # reference path

    @property
    def _filter_taps(self):
        return np.asarray(self.taps)

    def step(self, x):
        """Resample one chunk."""
        if x.ndim > 1 or not _numba.HAVE_NUMBA:
            return self._step_reference(x)
        h = np.ascontiguousarray(self._filter_taps)
        K = -(-len(h) // self.up)
        dt = np.result_type(x, h)
        if self.hist is None or self.hist.dtype != dt or len(self.hist) != K - 1:
            old, self.hist = self.hist, np.zeros(K - 1, dt)
            if old is not None and len(old) and K > 1:
                m = min(len(old), K - 1)
                self.hist[-m:] = old[-m:]
        ext = np.concatenate([self.hist, x.astype(dt, copy=False)])
        n_end = self.n_in + len(x)
        n_out = -(-(n_end * self.up) // self.down) - self.m_out
        y = np.zeros(n_out, dt)
        _poly_kernel(ext, h, self.up, self.down, self.m_out, self.n_in - (K - 1), y)
        self.hist = ext[len(ext) - (K - 1) :] if K > 1 else ext[:0]
        self.n_in, self.m_out = n_end, self.m_out + n_out
        return y

    def _step_reference(self, x):
        u = _zeros(x, len(x) * self.up)
        u[:: self.up] = x
        h = self._filter_taps
        if self.zi is None:
            self.zi = np.zeros((len(h) - 1,) + x.shape[1:], np.result_type(u, h))
        y, self.zi = sig.lfilter(h, 1.0, u, axis=0, zi=self.zi)
        first = (-self.k) % self.down
        self.k += len(u)
        y = y[first :: self.down]
        self.m_out += len(y)
        return y


class Interpolate(_Polyphase):
    """Upsample by ``L`` and low-pass filter.

    Parameters
    ----------
    L : int
    taps : array_like, optional
        Default: ``16 L + 1`` taps, cut-off at the input Nyquist, gain ``L``.
    """

    def __init__(self, L, taps=None, **kw):
        self.L = self.up = int(L)
        self.resampling_ratio = Fraction(self.L)
        if taps is None:
            taps = self.L * sig.firwin(16 * self.L + 1, 1.0 / self.L)
        self.taps = np.asarray(taps)
        super().__init__(**kw)

    def info(self):
        """Return factor and taps."""
        return f"↑{self.L}, {len(self.taps)} taps"
