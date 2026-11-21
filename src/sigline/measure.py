"""Measurement sinks: spectrum, power, constellation, eye, SNR, bit errors.

All keep bounded state (running sums, ring buffers, histograms), so they
suit long and live runs, and every decision depends on absolute sample
indices, so chunking never changes a result. Use them as blocks or as
probes: ``d.probe("rx", sink=SpectrumAnalyzer())``.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import scipy.signal as sig

from ._format import si
from .core import Block

__all__ = [
    "BERCounter",
    "ConstellationScope",
    "EyeDiagram",
    "PowerMeter",
    "SNREstimator",
    "Spectrogram",
    "SpectrumAnalyzer",
    "qam_ambiguities",
]


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
        """Plot the PSD in dB, with max-hold; empty before the first frame.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.set_xlabel("f / Hz")
        ax.set_ylabel("PSD / (dB/Hz)")
        ax.grid(alpha=0.3)
        if self._acc is None:  # live plots may ask before the first frame
            ax.set_title(f"{self.label}: waiting for {self.nfft} samples", fontsize=9)
            return ax
        ax.plot(self.f, self.psd_db, label=self.label, **kw)
        if max_hold:
            ax.plot(
                self.f, _db(self.max_hold), lw=0.6, alpha=0.6, label=f"{self.label} max"
            )
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


class Spectrogram(Block):
    """Short-time spectra over time: the newest ``keep`` frames.

    Equals ``scipy.signal.spectrogram(..., detrend=False)`` (density scaling;
    one-sided for real, two-sided centred for complex input). Safe to read
    while a diagram runs.

    Parameters
    ----------
    nfft : int, default 256
        Frame length.
    overlap : float, default 0.5
        Frame overlap; hop ``nfft - int(nfft * overlap)``.
    window : str, default "hann"
    keep : int, default 400
        Frames kept (ring buffer).
    label : str, optional
        Title; also the default name.
    """

    outputs = ()

    def __init__(
        self, nfft=256, overlap=0.5, window="hann", keep=400, label=None, **kw
    ):
        self.nfft, self.overlap, self.window = int(nfft), overlap, window
        self.keep, self.label = int(keep), label
        if label is not None:
            kw.setdefault("name", label)
        self._lock = threading.Lock()
        super().__init__(**kw)
        self.label = self.label or self.name

    def reset(self):
        """Forget all frames."""
        with self._lock:
            self._buf, self._buf0 = None, 0  # _buf0: absolute index of _buf[0]
            self._rows = deque(maxlen=self.keep)
            self._starts = deque(maxlen=self.keep)
            self.frames = 0
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
        if self.complex:
            P = np.abs(np.fft.fft(frames * self._win, axis=1)) ** 2 * self._scale
            P = np.fft.fftshift(P, axes=1)
        else:
            P = np.abs(np.fft.rfft(frames * self._win, axis=1)) ** 2 * self._scale
            P[:, 1 : -1 if self.nfft % 2 == 0 else None] *= 2
        starts = self._buf0 + self.hop * np.arange(F)
        with self._lock:
            self._rows.extend(P)
            self._starts.extend(starts)
            self.frames += F
        self._buf = buf[F * self.hop :]
        self._buf0 += F * self.hop
        return {}

    @property
    def f(self) -> np.ndarray:
        """Frequency / Hz of each bin."""
        if self.complex:
            return np.fft.fftshift(np.fft.fftfreq(self.nfft, 1 / self.fs))
        return np.fft.rfftfreq(self.nfft, 1 / self.fs)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t, P)``: frame centre times / s, PSD / (unit^2/Hz).

        ``P`` has shape ``(frames, bins)``, newest last.
        """
        with self._lock:
            starts, rows = np.array(self._starts), np.array(self._rows)
        return (starts + self.nfft / 2) / self.fs, rows

    @property
    def t(self) -> np.ndarray:
        """Centre time / s of each kept frame, absolute since the start."""
        return self.snapshot()[0]

    @property
    def psd(self) -> np.ndarray:
        """PSD / (unit^2/Hz) of each kept frame, shape ``(frames, bins)``."""
        return self.snapshot()[1]

    def plot(self, ax=None, colorbar=True, **kw):
        """Plot time across, frequency up, PSD in dB as colour.

        Empty until the first frame, so live plots can call it any time.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.set_xlabel("t / s")
        ax.set_ylabel("f / Hz")
        t, P = self.snapshot()
        if not len(t):
            ax.set_title(f"{self.label}: waiting for {self.nfft} samples", fontsize=9)
            return ax
        kw.setdefault("cmap", "viridis")
        kw.setdefault("shading", "nearest")
        mesh = ax.pcolormesh(t, self.f, _db(P).T, **kw)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label="PSD / (dB/Hz)")
        ax.set_title(self.label, fontsize=9)
        return ax

    def info(self):
        """Return frame length and frames kept."""
        return f"{self.nfft}-pt, {self.keep} frames"


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


class EyeDiagram(Block):
    """Eye diagram: the signal folded every ``sps`` samples, ``span`` symbols wide.

    Keeps a 2-D histogram (position x amplitude) of all traces and the
    newest ``keep`` traces as lines.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    span : int, default 2
        Symbols per trace.
    offset : int, default 0
        Sample where the first trace starts.
    component : {"real", "imag", "abs"}, default "real"
    bins : int, default 120
        Amplitude bins.
    range : tuple, optional
        Amplitude range; default: from the first 64 traces, plus 15 %.
    keep : int, default 200
        Traces kept as lines.
    interp : int, default 8
        Linear interpolation steps between samples in the histogram.
    label : str, optional
    """

    outputs = ()

    def __init__(
        self,
        sps,
        span=2,
        offset=0,
        component="real",
        bins=120,
        range=None,
        keep=200,
        interp=8,
        label=None,
        **kw,
    ):
        if component not in ("real", "imag", "abs"):
            raise ValueError("component must be 'real', 'imag' or 'abs'")
        self.sps, self.span, self.offset = int(sps), int(span), int(offset)
        self.component, self.bins, self.range0 = component, int(bins), range
        self.keep, self.interp, self.label = int(keep), int(interp), label
        if label is not None:
            kw.setdefault("name", label)
        super().__init__(**kw)
        self.label = self.label or self.name

    def reset(self):
        """Forget all traces."""
        self._buf, self._buf0 = np.zeros(0), 0  # _buf0: absolute index of _buf[0]
        self.n = self.traces_seen = 0
        self.range = self.range0
        self.hist = None
        self._pending = []  # traces before the range is known
        self._lines = deque(maxlen=self.keep)

    @property
    def length(self) -> int:
        """Samples per trace."""
        return self.span * self.sps + 1

    def step(self, x):
        """Cut the complete traces of this chunk."""
        v = getattr(np, self.component)(np.asarray(x)).astype(float)
        self._buf = np.concatenate([self._buf, v])
        self.n += len(v)
        k0 = max(0, -(-(self._buf0 - self.offset) // self.sps))
        k1 = (self.n - self.length - self.offset) // self.sps + 1
        starts = self.offset + self.sps * np.arange(k0, k1)
        if len(starts):
            idx = (starts - self._buf0)[:, None] + np.arange(self.length)[None, :]
            self._add(self._buf[idx])
            nxt = starts[-1] + self.sps
            self._buf, self._buf0 = self._buf[nxt - self._buf0 :], nxt
        return {}

    def _add(self, tr):
        self.traces_seen += len(tr)
        self._lines.extend(tr)
        if self.range is None:
            # the range comes from the first 64 traces, wherever chunks end
            self._pending.append(tr)
            if sum(len(p) for p in self._pending) < 64:
                return
            first = np.concatenate(self._pending)[:64]
            lo, hi = float(first.min()), float(first.max())
            pad = 0.15 * (hi - lo or 1.0)
            self.range = (lo - pad, hi + pad)
            tr = np.concatenate(self._pending)
            self._pending = []
        if self.interp > 1:
            nf = (self.length - 1) * self.interp + 1
            xf = np.arange(nf) / self.interp
            i0 = np.minimum(np.floor(xf).astype(int), self.length - 2)
            fr = xf - i0
            tr = tr[:, i0] * (1 - fr) + tr[:, i0 + 1] * fr
        nf = tr.shape[1]
        t = np.broadcast_to(np.arange(nf), tr.shape)
        h, _, _ = np.histogram2d(
            t.ravel(),
            tr.ravel(),
            bins=[nf, self.bins],
            range=[[-0.5, nf - 0.5], list(self.range)],
        )
        self.hist = h if self.hist is None else self.hist + h

    @property
    def traces(self) -> np.ndarray:
        """The newest ``keep`` traces, shape ``(traces, length)``."""
        return np.array(self._lines)

    def plot(self, ax=None, mode="hist", **kw):
        """Plot the histogram (``mode="hist"``) or the newest traces (``"lines"``).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        tt = np.arange(self.length) / self.sps
        hist = self.hist
        if mode == "hist" and hist is not None:
            kw.setdefault("cmap", "magma")
            ax.imshow(
                np.log1p(hist.T),
                origin="lower",
                aspect="auto",
                extent=[tt[0], tt[-1], self.range[0], self.range[1]],
                **kw,
            )
        else:
            for row in self.traces:
                ax.plot(tt, row, "b-", lw=0.4, alpha=0.3)
        ax.set_xlabel("t / T")
        ax.set_title(f"{self.label}: {self.traces_seen} traces", fontsize=9)
        return ax

    def info(self):
        """Return samples per symbol and span."""
        return f"{self.sps} sps, {self.span} UI"


class SNREstimator(Block):
    """SNR of a symbol stream (one sample per symbol).

    ``m2m4`` is blind (no decisions, works before the receiver locks) and
    needs the constellation's kurtosis ``E|a|^4 / (E|a|^2)^2``; it assumes
    complex Gaussian noise. At high SNR it divides nearly equal numbers, and
    distortion that changes the kurtosis (compression, residual ISI) biases
    it; it then returns inf. ``dd`` compares with the nearest constellation
    point: accurate at high SNR, biased upward once decisions fail.

    Parameters
    ----------
    method : {"m2m4", "dd"}, default "m2m4"
    constellation : array_like, optional
        Needed for ``dd``; gives the kurtosis for ``m2m4``.
    kurtosis : float, optional
        For ``m2m4`` without a constellation.
    skip : int, default 0
        Symbols ignored (receiver convergence).

    Raises
    ------
    ValueError
        Unknown method, or what the method needs is missing.
    """

    outputs = ()

    def __init__(self, method="m2m4", constellation=None, kurtosis=None, skip=0, **kw):
        if method not in ("m2m4", "dd"):
            raise ValueError("method must be 'm2m4' or 'dd'")
        self.method, self.skip = method, int(skip)
        self.constellation = (
            None if constellation is None else np.asarray(constellation)
        )
        if kurtosis is None and self.constellation is not None:
            c = self.constellation
            kurtosis = np.mean(np.abs(c) ** 4) / np.mean(np.abs(c) ** 2) ** 2
        if method == "m2m4" and kurtosis is None:
            raise ValueError("m2m4 needs constellation= or kurtosis=")
        if method == "dd" and self.constellation is None:
            raise ValueError("dd needs constellation=")
        self.kurtosis = kurtosis
        super().__init__(**kw)

    def reset(self):
        """Forget all symbols."""
        self.n = self.count = 0
        self.m2 = self.m4 = 0.0
        self._sig = self._noise = 0.0

    def step(self, x):
        """Accumulate moments or decision errors."""
        j = self.n + np.arange(len(x))
        self.n += len(x)
        y = np.asarray(x)[j >= self.skip]
        if not len(y):
            return {}
        p = np.abs(y) ** 2
        self.m2 += float(p.sum())
        self.m4 += float((p**2).sum())
        self.count += len(y)
        if self.method == "dd":
            d = _nearest(y, self.constellation)
            self._sig += float(np.sum(np.abs(d) ** 2))
            self._noise += float(np.sum(np.abs(y - d) ** 2))
        return {}

    @property
    def snr(self) -> float:
        """Signal-to-noise power ratio (linear)."""
        if not self.count:
            return float("nan")
        if self.method == "dd":
            return self._sig / self._noise if self._noise else float("inf")
        M2, M4 = self.m2 / self.count, self.m4 / self.count
        s = np.sqrt(max(2 * M2**2 - M4, 0.0) / (2 - self.kurtosis))
        return float(s / (M2 - s)) if M2 > s else float("inf")

    @property
    def snr_db(self) -> float:
        """SNR / dB."""
        return float(10 * np.log10(self.snr))

    def info(self):
        """Return the method."""
        return self.method


def qam_ambiguities(constellation):
    """Look-up tables for the 4-fold phase ambiguity of square QAM.

    Parameters
    ----------
    constellation : array_like
        ``constellation[w]`` is the point of word ``w``.

    Returns
    -------
    list of numpy.ndarray
        Four tables; table ``k`` maps a word to the word it becomes after
        rotating by ``k`` x 90 degrees.
    """
    c = np.asarray(constellation)
    return [
        np.argmin(np.abs((c * 1j**k)[:, None] - c[None, :]), axis=1) for k in range(4)
    ]


class BERCounter(Block):
    """Bit error counter with automatic alignment, like a lab BERT.

    Inputs ``desired`` and ``received`` are bit streams (bool, or numbers
    thresholded at 0.5) at one rate. After ``skip`` bits it finds the delay
    of ``received`` (0 .. ``max_delay``) on a window of ``sync`` bits; if the
    best BER there exceeds ``lock_ber`` it tries the next window. Once locked
    it counts every later bit.

    Parameters
    ----------
    skip : int, default 0
        Bits ignored first (receiver convergence).
    sync : int, default 1000
        Bits per alignment window.
    max_delay : int, default 100
    lock_ber : float, default 0.25
    count_sync : bool, default False
        Also count the window used for locking.
    stop_after_errors : int, optional
        Request a stop once this many errors (and ``min_bits`` bits) are counted.
    min_bits : int, default 0
    """

    inputs = ("desired", "received")
    outputs = ()

    def __init__(
        self,
        skip=0,
        sync=1000,
        max_delay=100,
        lock_ber=0.25,
        count_sync=False,
        stop_after_errors=None,
        min_bits=0,
        **kw,
    ):
        self.skip0, self.sync, self.max_delay = int(skip), int(sync), int(max_delay)
        self.lock_ber, self.count_sync = lock_ber, count_sync
        self.stop_after_errors, self.min_bits = stop_after_errors, min_bits
        super().__init__(**kw)

    def reset(self):
        """Unlock and forget all counts."""
        self._d, self._d0 = np.zeros(0, bool), 0  # desired bits from index _d0
        self._r, self._r0 = np.zeros(0, bool), 0  # received bits awaiting the lock
        self.n = 0
        self.skip = self.skip0
        self.locked, self.delay, self.lock_attempts = False, None, 0
        self.errors = self.bits = 0
        self._count_from = None

    @staticmethod
    def _bits(v):
        v = np.asarray(v)
        return v if v.dtype == bool else np.real(v) > 0.5

    def step(self, desired, received):
        """Align, then count errors."""
        d, r = self._bits(desired), self._bits(received)
        self._d = np.concatenate([self._d, d])
        start = self.n
        self.n += len(r)
        if not self.locked:
            self._r = np.concatenate([self._r, r])
            while not self.locked and self.n >= self.skip + self.sync:
                self._try_lock()
            if self.locked:
                j0 = max(self._count_from, self._r0)
                self._count(np.arange(j0, self.n), self._r[j0 - self._r0 :])
                self._r = np.zeros(0, bool)
        else:
            j = np.arange(start, self.n)
            m = j >= self._count_from
            self._count(j[m], r[m])
        # drop history that can no longer be needed
        if self.locked:
            keep = self.n - self.delay - 1
        else:
            keep = min(self._r0, self.skip) - self.max_delay
        drop = min(max(0, keep - self._d0), len(self._d))
        if drop:
            self._d, self._d0 = self._d[drop:], self._d0 + drop
        if not self.locked and self._r0 < self.skip:
            k = min(self.skip - self._r0, len(self._r))
            self._r, self._r0 = self._r[k:], self._r0 + k
        stop = self.stop_after_errors
        if stop and self.errors >= stop and self.bits >= self.min_bits:
            self.request_stop(f"{self.errors} bit errors in {self.bits} bits")
        return {}

    def _try_lock(self):
        j = np.arange(self.skip, self.skip + self.sync)
        r = self._r[j - self._r0]
        best = None
        for D in range(self.max_delay + 1):
            if j[0] - D < self._d0:
                break
            ber = np.mean(r != self._d[j - D - self._d0])
            if best is None or ber < best[0]:
                best = (ber, D)
        self.lock_attempts += 1
        if best is not None and best[0] <= self.lock_ber:
            self.locked, self.delay = True, best[1]
            self._count_from = self.skip if self.count_sync else self.skip + self.sync
        else:
            self.skip += self.sync

    def _count(self, j, r):
        if len(j):
            self.errors += int(
                np.count_nonzero(r != self._d[j - self.delay - self._d0])
            )
            self.bits += len(j)

    @property
    def ber(self) -> float:
        """Errors / bits counted."""
        return self.errors / self.bits if self.bits else float("nan")

    def confidence(self, level=0.95) -> tuple[float, float]:
        """Return the Clopper-Pearson interval of the BER.

        Parameters
        ----------
        level : float, default 0.95

        Returns
        -------
        tuple of float
        """
        from scipy.stats import beta

        k, n, a = self.errors, self.bits, 1 - level
        lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
        hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
        return float(lo), float(hi)

    def summary(self) -> str:
        """Return BER, counts, interval and delay as one line."""
        if not self.locked:
            return f"{self.name}: not locked ({self.lock_attempts} attempts)"
        lo, hi = self.confidence()
        return (
            f"{self.name}: BER {self.ber:.3e} ({self.errors}/{self.bits} bits), "
            f"95% CI [{lo:.2e}, {hi:.2e}], delay {self.delay} bits"
        )

    def info(self):
        """Return the skip."""
        return f"skip {self.skip0}" if self.skip0 else ""
