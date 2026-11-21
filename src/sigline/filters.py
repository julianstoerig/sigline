"""Linear filters with persistent state."""

from __future__ import annotations

import numpy as np
import scipy.signal as sig

from .core import Block

__all__ = ["FIR"]


class FIR(Block):
    """FIR filter for real, complex and vector signals.

    The state is the last ``len(taps) - 1`` inputs, so ``taps`` can change
    between runs without a transient. From 128 taps, long chunks use FFT
    convolution.

    Parameters
    ----------
    taps : array_like
        Impulse response.
    """

    def __init__(self, taps, **kw):
        self.taps = np.asarray(taps)
        super().__init__(**kw)

    def reset(self):
        """Clear the input history."""
        self.hist = None

    def step(self, x):
        """Filter one chunk."""
        K = len(self.taps) - 1
        if self.hist is None:
            self.hist = np.zeros((K,) + x.shape[1:], dtype=x.dtype)
        elif len(self.hist) != K:  # taps changed length: keep the newest inputs
            h = np.zeros((K,) + x.shape[1:], dtype=self.hist.dtype)
            m = min(K, len(self.hist))
            if m:
                h[K - m :] = self.hist[len(self.hist) - m :]
            self.hist = h
        ext = np.concatenate([self.hist, x])
        taps = self.taps.reshape((-1,) + (1,) * (x.ndim - 1))
        if K >= 128 and len(x) >= 2 * K:
            y = sig.oaconvolve(ext, taps, mode="valid", axes=0)
        else:
            y = sig.convolve(ext, taps, mode="valid", method="direct")
        self.hist = ext[len(ext) - K :] if K else ext[:0]
        return y

    def freqz(self, n=2048, whole=False):
        """Return the frequency response.

        Parameters
        ----------
        n : int, default 2048
            Number of frequencies.
        whole : bool, default False
            Up to ``fs`` instead of ``fs / 2``.

        Returns
        -------
        f : numpy.ndarray
            Frequency / Hz (needs ``fs``).
        H : numpy.ndarray
            Complex response.
        """
        return sig.freqz(self.taps, 1, worN=n, whole=whole, fs=self.fs)
