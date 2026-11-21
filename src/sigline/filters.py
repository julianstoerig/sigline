"""Linear filters with persistent state."""

from __future__ import annotations

import numpy as np
import scipy.signal as sig

from .core import Block

__all__ = ["FIR", "IIR"]


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

    def info(self):
        """Return the number of taps."""
        return f"{len(self.taps)} taps"


class IIR(Block):
    """IIR filter from ``sos`` (recommended) or ``b, a``.

    ``IIR(b=[kp + ki / fs, -kp], a=[1, -1])`` is a discrete PI controller.

    Parameters
    ----------
    sos : array_like, optional
        Second-order sections, e.g. from ``scipy.signal.butter(..., output="sos")``.
    b, a : array_like, optional
        Numerator and denominator; ``a`` defaults to ``[1]``.

    Raises
    ------
    ValueError
        Neither ``sos`` nor ``b``.
    """

    def __init__(self, sos=None, b=None, a=None, **kw):
        if sos is None:
            if b is None:
                raise ValueError("give sos= or b=, a=")
            a = [1.0] if a is None else a
        self.sos = sos
        self.b = None if b is None else np.asarray(b, float)
        self.a = None if a is None else np.asarray(a, float)
        super().__init__(**kw)

    def reset(self):
        """Clear the filter state."""
        self.zi = None

    def step(self, x):
        """Filter one chunk."""
        tail = x.shape[1:]
        if self.sos is not None:
            if self.zi is None:
                self.zi = np.zeros((len(self.sos), 2) + tail, np.result_type(x, float))
            y, self.zi = sig.sosfilt(self.sos, x, axis=0, zi=self.zi)
        else:
            if self.zi is None:
                n = max(len(self.a), len(self.b)) - 1
                self.zi = np.zeros((n,) + tail, np.result_type(x, float))
            y, self.zi = sig.lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return y

    def freqz(self, n=2048, whole=False):
        """Return ``(f / Hz, H)``; see :meth:`FIR.freqz`."""
        if self.sos is not None:
            return sig.freqz_sos(self.sos, worN=n, whole=whole, fs=self.fs)
        return sig.freqz(self.b, self.a, worN=n, whole=whole, fs=self.fs)

    def info(self):
        """Return sections or order."""
        if self.sos is not None:
            return f"{len(self.sos)} SOS"
        return f"order {max(len(self.a), len(self.b)) - 1}"
