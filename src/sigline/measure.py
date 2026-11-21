"""Measurement sinks: spectrum, power, constellation, eye, SNR, bit errors.

All keep bounded state (running sums, ring buffers, histograms), so they
suit long and live runs, and every decision depends on absolute sample
indices, so chunking never changes a result. Use them as blocks or as
probes: ``d.probe("rx", sink=SpectrumAnalyzer())``.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.signal as sig

from ._format import si
from .core import Block

__all__ = ["ConstellationScope", "PowerMeter", "SpectrumAnalyzer"]


def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-300))


class SpectrumAnalyzer(Block):
    """Running Welch spectrum with max-hold and an optional waterfall.

    Real input gives a one-sided, complex input a two-sided centred
    spectrum, as a density / (unit^2/Hz).

    Parameters
    ----------
    nfft : int, default 4096
        Frame length.
    overlap : float, default 0.5
        Frame overlap; hop ``nfft - int(nfft * overlap)``.
    window : str, default "hann"
        SciPy window name.
    average : {"mean", "exp"}, default "mean"
        ``mean`` equals ``scipy.signal.welch(..., detrend=False)``; ``exp``
        weights each new frame by ``alpha``, for live displays.
    alpha : float, default 0.1
    waterfall : int, default 0
        Keep the newest ``waterfall`` frames in dB.
    skip : int, default 0
        Ignore the first samples (start-up transients).
    label : str, optional
        Legend text; also the default name.
    """

    outputs = ()

    def __init__(
        self,
        nfft=4096,
        overlap=0.5,
        window="hann",
        average="mean",
        alpha=0.1,
        waterfall=0,
        skip=0,
        label=None,
        **kw,
    ):
        if average not in ("mean", "exp"):
            raise ValueError("average must be 'mean' or 'exp'")
        self.nfft, self.overlap, self.window = int(nfft), overlap, window
        self.average, self.alpha, self.waterfall_rows = average, alpha, int(waterfall)
        self.skip, self.label = int(skip), label
        if label is not None:
            kw.setdefault("name", label)
        super().__init__(**kw)
        self.label = self.label or self.name

    def reset(self):
        """Forget all frames."""
        self._buf = self._acc = self._max = self._avg = None
        self.frames = self._seen = 0
        self._rows = deque(maxlen=self.waterfall_rows or 1)
        self.complex = None

    @property
    def hop(self) -> int:
        """Samples between frame starts."""
        return self.nfft - int(self.nfft * self.overlap)

    def step(self, x):
        """Add the complete frames of this chunk."""
        if x.ndim > 1:
            raise ValueError(
                f"{self.name}: analyse one component, e.g. Func(lambda v: v[:, 0])"
            )
        if self._seen < self.skip:
            drop = min(len(x), self.skip - self._seen)
            self._seen += drop
            x = x[drop:]
            if not len(x):
                return {}
        if self.complex is None:
            self.complex = np.iscomplexobj(x)
            self._win = sig.get_window(self.window, self.nfft)
            self._scale = 1.0 / (self.fs * np.sum(self._win**2))
        buf = x if self._buf is None else np.concatenate([self._buf, x])
        if len(buf) < self.nfft:
            self._buf = buf
            return {}
        F = 1 + (len(buf) - self.nfft) // self.hop
        frames = np.lib.stride_tricks.sliding_window_view(buf, self.nfft)[:: self.hop][
            :F
        ]
        self._buf = buf[F * self.hop :]
        if self.complex:
            P = np.abs(np.fft.fft(frames * self._win, axis=1)) ** 2 * self._scale
        else:
            P = np.abs(np.fft.rfft(frames * self._win, axis=1)) ** 2 * self._scale
            P[:, 1 : -1 if self.nfft % 2 == 0 else None] *= 2
        mx = P.max(axis=0)
        self._max = mx if self._max is None else np.maximum(self._max, mx)
        if self.average == "exp":
            avg = self._avg
            for row in P:
                avg = row if avg is None else (1 - self.alpha) * avg + self.alpha * row
            self._avg = avg
        if self.waterfall_rows:
            for row in P:
                self._rows.append(_db(self._shift(row)))
        # sum and count as one tuple, so a reader in another thread sees a pair
        acc = P.sum(axis=0) if self._acc is None else self._acc[0] + P.sum(axis=0)
        self._acc = (acc, self.frames + F)
        self.frames += F
        return {}

    def _shift(self, v):
        return np.fft.fftshift(v) if self.complex else v

    @property
    def f(self) -> np.ndarray:
        """Frequency / Hz of each bin."""
        if self.complex:
            return np.fft.fftshift(np.fft.fftfreq(self.nfft, 1 / self.fs))
        return np.fft.rfftfreq(self.nfft, 1 / self.fs)

    @property
    def psd(self) -> np.ndarray:
        """Averaged PSD / (unit^2/Hz).

        Raises
        ------
        RuntimeError
            Before the first complete frame.
        """
        if self._acc is None:
            raise RuntimeError(
                f"{self.name}: no complete frame yet ({self.nfft} samples)"
            )
        if self.average == "exp":
            return self._shift(self._avg)
        acc, frames = self._acc
        return self._shift(acc / frames)

    @property
    def psd_db(self) -> np.ndarray:
        """PSD / (dB/Hz)."""
        return _db(self.psd)

    @property
    def max_hold(self) -> np.ndarray:
        """Highest PSD of any frame / (unit^2/Hz)."""
        return self._shift(self._max)

    @property
    def waterfall(self) -> np.ndarray:
        """Newest frames in dB, shape ``(frames, bins)``, newest last."""
        return np.array(self._rows)

    def peak(self) -> tuple[float, float]:
        """Return frequency / Hz and PSD of the strongest bin."""
        p = self.psd
        i = int(np.argmax(p))
        return float(self.f[i]), float(p[i])

    def plot(self, ax=None, max_hold=True, **kw):
        """Plot the PSD in dB, with max-hold.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.plot(self.f, self.psd_db, label=self.label, **kw)
        if max_hold:
            ax.plot(
                self.f, _db(self.max_hold), lw=0.6, alpha=0.6, label=f"{self.label} max"
            )
        ax.set_xlabel("f / Hz")
        ax.set_ylabel("PSD / (dB/Hz)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        return ax

    def plot_waterfall(self, ax=None, **kw):
        """Plot the waterfall: frequency across, time up, newest on top.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        w = self.waterfall
        kw.setdefault("cmap", "viridis")
        ax.imshow(
            w,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[self.f[0], self.f[-1], 0, len(w) * self.hop / self.fs],
            **kw,
        )
        ax.set_xlabel("f / Hz")
        ax.set_ylabel("t / s")
        return ax

    def info(self):
        """Return frame length and averaging."""
        return f"{self.nfft}-pt, {self.average}"


class PowerMeter(Block):
    """Mean power, overall and per ``interval``.

    Parameters
    ----------
    interval : float, optional
        Seconds per reading in :attr:`readings`.
    unit : {"dB", "dBm", "dBFS"}, default "dB"
        ``dB``: 10 log10 of mean |x|^2. ``dBm``: |x|^2 in W. ``dBFS``: 0 dBFS
        is a full-scale sine (real) or a full-scale |x| (complex).
    full_scale : float, default 1.0
        Amplitude for ``dBFS``.

    Raises
    ------
    ValueError
        Unknown unit.
    """

    outputs = ()

    def __init__(self, interval=None, unit="dB", full_scale=1.0, **kw):
        if unit not in ("dB", "dBm", "dBFS"):
            raise ValueError("unit must be 'dB', 'dBm' or 'dBFS'")
        self.interval, self.unit, self.full_scale = interval, unit, full_scale
        super().__init__(**kw)

    def reset(self):
        """Forget all power."""
        self.total, self.count = 0.0, 0
        self._cur, self._cur_n = 0.0, 0
        self._readings = []
        self.complex = None

    def _to_unit(self, p):
        if self.unit == "dB":
            return _db(p)
        if self.unit == "dBm":
            return _db(p / 1e-3)
        ref = self.full_scale**2 if self.complex else self.full_scale**2 / 2
        return _db(p / ref)

    def _n_int(self):
        return max(1, round(self.interval * self.fs))

    def step(self, x):
        """Accumulate this chunk's power."""
        if self.complex is None:
            self.complex = np.iscomplexobj(x)
        p = np.abs(np.asarray(x)) ** 2
        if p.ndim > 1:
            p = p.sum(axis=tuple(range(1, p.ndim)))
        self.total += float(p.sum())
        self.count += len(p)
        if self.interval:
            n_int, i = self._n_int(), 0
            while i < len(p):
                take = min(n_int - self._cur_n, len(p) - i)
                self._cur += float(p[i : i + take].sum())
                self._cur_n += take
                i += take
                if self._cur_n == n_int:
                    self._readings.append(self._cur / n_int)
                    self._cur, self._cur_n = 0.0, 0
        return {}

    @property
    def power(self) -> float:
        """Mean |x|^2 of everything seen."""
        return self.total / self.count if self.count else float("nan")

    @property
    def level(self) -> float:
        """:attr:`power` in ``unit``."""
        return float(self._to_unit(self.power))

    @property
    def readings(self) -> np.ndarray:
        """Power per completed interval, in ``unit``."""
        return self._to_unit(np.array(self._readings))

    @property
    def readings_t(self) -> np.ndarray:
        """End time / s of each reading."""
        return (np.arange(len(self._readings)) + 1) * self._n_int() / self.fs

    def info(self):
        """Return unit and interval."""
        return self.unit + (f" every {si(self.interval, 's')}" if self.interval else "")


def _nearest(y, points):
    # nearest constellation point per sample, in blocks to bound memory
    out = np.empty(len(y), dtype=points.dtype)
    for i in range(0, len(y), 8192):
        yy = y[i : i + 8192]
        out[i : i + 8192] = points[
            np.argmin(np.abs(yy[:, None] - points[None, :]), axis=1)
        ]
    return out


class ConstellationScope(Block):
    """Newest symbols, and MER / EVM against a reference constellation.

    Decision-directed: each symbol (one sample per symbol) is compared with
    its nearest reference point, from symbol ``skip`` on.

    Parameters
    ----------
    reference : array_like, optional
        Ideal points; without it, only points are kept.
    keep : int, default 4000
        Symbols kept for plotting.
    skip : int, default 0
        Symbols ignored for MER / EVM (receiver convergence).
    label : str, optional
    """

    outputs = ()

    def __init__(self, reference=None, keep=4000, skip=0, label=None, **kw):
        self.reference = None if reference is None else np.asarray(reference)
        self.keep, self.skip, self.label = int(keep), int(skip), label
        if label is not None:
            kw.setdefault("name", label)
        super().__init__(**kw)
        self.label = self.label or self.name

    def reset(self):
        """Forget symbols and errors."""
        self._pts = np.zeros(0, complex)
        self.n = self.count = 0
        self._err = self._ref = 0.0

    def step(self, x):
        """Keep the symbols and add their errors."""
        x = np.asarray(x)
        self._pts = np.concatenate([self._pts, x])[-self.keep :]
        j = self.n + np.arange(len(x))
        self.n += len(x)
        if self.reference is not None:
            y = x[j >= self.skip]
            d = _nearest(y, self.reference)
            self._err += float(np.sum(np.abs(y - d) ** 2))
            self._ref += float(np.sum(np.abs(d) ** 2))
            self.count += len(y)
        return {}

    @property
    def points(self) -> np.ndarray:
        """The newest ``keep`` symbols."""
        return self._pts

    @property
    def mer_db(self) -> float:
        """Modulation error ratio / dB: reference power over error power."""
        return (
            float(10 * np.log10(self._ref / self._err)) if self._err else float("inf")
        )

    @property
    def evm_percent(self) -> float:
        """RMS error vector over RMS reference / %."""
        return (
            float(100 * np.sqrt(self._err / self._ref)) if self._ref else float("nan")
        )

    def plot(self, ax=None, **kw):
        """Plot the points, the reference and MER / EVM.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        kw.setdefault("ms", 1.5)
        kw.setdefault("alpha", 0.5)
        pts = self._pts
        ax.plot(pts.real, pts.imag, ".", **kw)
        if self.reference is not None:
            ax.plot(self.reference.real, self.reference.imag, "r+", ms=6)
            ax.set_title(
                f"{self.label}: MER {self.mer_db:.1f} dB, EVM {self.evm_percent:.2f} %",
                fontsize=9,
            )
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        return ax

    def info(self):
        """Return what it measures."""
        return "MER/EVM" if self.reference is not None else f"last {self.keep}"
