"""Scope: records the signal on its input line, plots it and its spectrum."""

from __future__ import annotations

import threading

import numpy as np
import scipy.signal as sig

from .core import Block

__all__ = ["Scope"]


class Scope(Block):
    """Record the input.

    Parameters
    ----------
    label : str, optional
        Legend text; also the default name.
    keep : int, optional
        Keep only the newest ``keep`` recorded samples (ring buffer).
    every : int, default 1
        Record every ``every``-th sample.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Scope
    >>> sc = Scope(every=2)
    >>> sc(np.arange(5.0)), sc(np.arange(5.0, 8.0))
    ((), ())
    >>> sc.data
    array([0., 2., 4., 6.])
    """

    outputs = ()

    def __init__(self, label=None, keep=None, every=1, **kw):
        self.label, self.keep, self.every = label, keep, int(every)
        self._lock = threading.Lock()  # data may be read while a run appends
        if label is not None:
            kw.setdefault("name", label)
        super().__init__(**kw)
        self.label = self.label or self.name

    def reset(self):
        """Forget the recording."""
        with self._lock:
            self._chunks, self._len, self._dropped, self._seen = [], 0, 0, 0

    def step(self, x):
        """Record one chunk."""
        with self._lock:
            if self.every > 1:
                first = (-self._seen) % self.every
                self._seen += len(x)
                x = x[first :: self.every]
            self._chunks.append(np.array(x, copy=True))
            self._len += len(x)
            if self.keep is not None and self._len > 2 * self.keep:
                d = np.concatenate(self._chunks)
                self._dropped += len(d) - self.keep
                self._chunks, self._len = [d[-self.keep :]], self.keep
        return {}

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, data)`` consistent with each other, also during a run."""
        with self._lock:
            return self._t_data()

    def _t_data(self):
        if not self._chunks:
            return np.empty(0), np.empty(0)
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        d = self._chunks[0]
        if self.keep is not None:
            d = d[-self.keep :]
        start = self._dropped + (self._len - len(d))
        return (start + np.arange(len(d))) * self.every / self.fs, d

    @property
    def data(self) -> np.ndarray:
        """Recorded samples; the newest ``keep`` if set."""
        with self._lock:
            if not self._chunks:
                return np.empty(0)
            if len(self._chunks) > 1:
                self._chunks = [np.concatenate(self._chunks)]
            d = self._chunks[0]
            return d[-self.keep :] if self.keep is not None else d

    @property
    def t(self) -> np.ndarray:
        """Time / s of each recorded sample, absolute since the start."""
        return self.snapshot()[0]

    def plot(self, ax=None, **kw):
        """Plot the recording over time; complex signals as I and Q.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Default: the current axes.
        **kw
            Passed to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        t, d = self.snapshot()
        if np.iscomplexobj(d):
            ax.plot(t, d.real, label=f"{self.label} (I)", **kw)
            ax.plot(t, d.imag, label=f"{self.label} (Q)", **kw)
        else:
            ax.plot(t, d, label=self.label, **kw)
        ax.set_xlabel("t / s")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        return ax

    def welch(self, **kw):
        """Return the power spectral density by Welch's method.

        No detrending, unlike SciPy's default: DC can be a carrier. Complex
        signals give a two-sided, centred spectrum.

        Parameters
        ----------
        **kw
            Passed to :func:`scipy.signal.welch`; ``nperseg`` defaults to
            ``min(4096, len(data))``.

        Returns
        -------
        f : numpy.ndarray
            Frequency / Hz.
        P : numpy.ndarray
            PSD / (unit^2/Hz).
        """
        d = self.data
        kw.setdefault("nperseg", min(4096, len(d)))
        kw.setdefault("detrend", False)
        if np.iscomplexobj(d):
            f, P = sig.welch(d, fs=self.fs, return_onesided=False, **kw)
            return np.fft.fftshift(f), np.fft.fftshift(P)
        return sig.welch(d, fs=self.fs, **kw)

    def psd(self, ax=None, **kw):
        """Plot :meth:`welch` in dB.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        **kw
            Passed to :meth:`welch`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        f, P = self.welch(**kw)
        ax.plot(f, 10 * np.log10(P + 1e-30), label=self.label)
        ax.set_xlabel("f / Hz")
        ax.set_ylabel("PSD / (dB/Hz)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        return ax
