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
from .digital import nearest

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


class _Sink(Block):
    """Measurement sink: no outputs, and a lock.

    ``step`` updates the state under the lock and the reading properties
    take it too, so a reader in another thread (a live plot) never sees half
    an update.
    """

    outputs = ()

    def __init__(self, **kw):
        self._lock = threading.Lock()
        super().__init__(**kw)


class _Frames:
    """Windowed frames of a stream and their power spectral densities.

    Density scaling as ``scipy.signal.welch(..., detrend=False)``: one-sided
    for real input, two-sided and centred for complex input.
    """

    def __init__(self, nfft, overlap, window):
        self.nfft, self.overlap, self.window = int(nfft), overlap, window
        self._buf, self.start, self.complex = None, 0, None  # start: of _buf[0]

    @property
    def hop(self) -> int:
        return self.nfft - int(self.nfft * self.overlap)

    def freqs(self, fs):
        if self.complex:
            return np.fft.fftshift(np.fft.fftfreq(self.nfft, 1 / fs))
        return np.fft.rfftfreq(self.nfft, 1 / fs)

    def add(self, x, fs, name):
        """Return the start index and PSD of each frame completed by ``x``."""
        if x.ndim > 1:
            raise ValueError(
                f"{name}: analyse one component, e.g. FuncChunk(lambda v: v[:, 0])"
            )
        if self.complex is None:
            self.complex = np.iscomplexobj(x)
            self._win = sig.get_window(self.window, self.nfft)
            self._scale = 1.0 / (fs * np.sum(self._win**2))
        buf = x if self._buf is None else np.concatenate([self._buf, x])
        F = max(0, 1 + (len(buf) - self.nfft) // self.hop)
        if not F:
            self._buf = buf
            return np.zeros(0, int), None
        frames = np.lib.stride_tricks.sliding_window_view(buf, self.nfft)
        frames = frames[:: self.hop][:F] * self._win
        if self.complex:
            P = np.fft.fftshift(np.abs(np.fft.fft(frames, axis=1)) ** 2, axes=1)
        else:
            P = np.abs(np.fft.rfft(frames, axis=1)) ** 2
            P[:, 1 : -1 if self.nfft % 2 == 0 else None] *= 2
        starts = self.start + self.hop * np.arange(F)
        self._buf, self.start = buf[F * self.hop :], self.start + F * self.hop
        return starts, P * self._scale


class SpectrumAnalyzer(_Sink):
    """Running Welch spectrum with max-hold.

    Real input gives a one-sided, complex input a two-sided centred
    spectrum, as a density / (unit^2/Hz). For spectra over time, use
    :class:`Spectrogram`.

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
    skip : int, default 0
        Ignore the first samples (start-up transients).
    """

    def __init__(
        self, nfft=4096, overlap=0.5, window="hann", average="mean", alpha=0.1,
        skip=0, **kw,
    ):  # fmt: skip
        if average not in ("mean", "exp"):
            raise ValueError("average must be 'mean' or 'exp'")
        self.nfft, self.overlap, self.window = int(nfft), overlap, window
        self.average, self.alpha, self.skip = average, alpha, int(skip)
        super().__init__(**kw)

    def reset(self):
        """Forget all frames."""
        with self._lock:
            self._frames = _Frames(self.nfft, self.overlap, self.window)
            self._acc = self._max = self._avg = None
            self.frames = self._seen = 0

    @property
    def hop(self) -> int:
        """Samples between frame starts."""
        return self._frames.hop

    @property
    def complex(self) -> bool | None:
        """Whether the input is complex; None before the first chunk."""
        return self._frames.complex

    def step(self, x):
        """Add the complete frames of this chunk."""
        if self._seen < self.skip:
            drop = min(len(x), self.skip - self._seen)
            self._seen += drop
            x = x[drop:]
            if not len(x):
                return {}
        _, P = self._frames.add(x, self.fs, self.name)
        if P is None:
            return {}
        with self._lock:
            mx = P.max(axis=0)
            self._max = mx if self._max is None else np.maximum(self._max, mx)
            if self.average == "exp":
                avg = self._avg
                for row in P:
                    avg = (
                        row
                        if avg is None
                        else (1 - self.alpha) * avg + self.alpha * row
                    )
                self._avg = avg
            sum_ = P.sum(axis=0)
            self._acc = sum_ if self._acc is None else self._acc + sum_
            self.frames += len(P)
        return {}

    @property
    def f(self) -> np.ndarray:
        """Frequency / Hz of each bin."""
        return self._frames.freqs(self.fs)

    @property
    def psd(self) -> np.ndarray:
        """Averaged PSD / (unit^2/Hz).

        Raises
        ------
        RuntimeError
            Before the first complete frame.
        """
        with self._lock:
            if self._acc is None:
                raise RuntimeError(
                    f"{self.name}: no complete frame yet ({self.nfft} samples)"
                )
            return self._avg if self.average == "exp" else self._acc / self.frames

    @property
    def psd_db(self) -> np.ndarray:
        """PSD / (dB/Hz)."""
        return _db(self.psd)

    @property
    def max_hold(self) -> np.ndarray:
        """Highest PSD of any frame / (unit^2/Hz)."""
        with self._lock:
            return self._max

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
            ax.set_title(f"{self.name}: waiting for {self.nfft} samples", fontsize=9)
            return ax
        ax.plot(self.f, self.psd_db, label=self.name, **kw)
        if max_hold:
            ax.plot(
                self.f, _db(self.max_hold), lw=0.6, alpha=0.6, label=f"{self.name} max"
            )
        ax.legend(loc="upper right", fontsize=8)
        return ax

    def info(self):
        """Return frame length and averaging."""
        return f"{self.nfft}-pt, {self.average}"


class Spectrogram(_Sink):
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
    """

    def __init__(self, nfft=256, overlap=0.5, window="hann", keep=400, **kw):
        self.nfft, self.overlap, self.window = int(nfft), overlap, window
        self.keep = int(keep)
        super().__init__(**kw)

    def reset(self):
        """Forget all frames."""
        with self._lock:
            self._frames = _Frames(self.nfft, self.overlap, self.window)
            self._rows = deque(maxlen=self.keep)
            self._starts = deque(maxlen=self.keep)
            self.frames = 0

    @property
    def hop(self) -> int:
        """Samples between frame starts."""
        return self._frames.hop

    @property
    def complex(self) -> bool | None:
        """Whether the input is complex; None before the first chunk."""
        return self._frames.complex

    def step(self, x):
        """Add the complete frames of this chunk."""
        starts, P = self._frames.add(x, self.fs, self.name)
        if P is not None:
            with self._lock:
                self._rows.extend(P)
                self._starts.extend(starts)
                self.frames += len(P)
        return {}

    @property
    def f(self) -> np.ndarray:
        """Frequency / Hz of each bin."""
        return self._frames.freqs(self.fs)

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
            ax.set_title(f"{self.name}: waiting for {self.nfft} samples", fontsize=9)
            return ax
        kw.setdefault("cmap", "viridis")
        kw.setdefault("shading", "nearest")
        mesh = ax.pcolormesh(t, self.f, _db(P).T, **kw)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label="PSD / (dB/Hz)")
        ax.set_title(self.name, fontsize=9)
        return ax

    def info(self):
        """Return frame length and frames kept."""
        return f"{self.nfft}-pt, {self.keep} frames"


class PowerMeter(_Sink):
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

    def __init__(self, interval=None, unit="dB", full_scale=1.0, **kw):
        if unit not in ("dB", "dBm", "dBFS"):
            raise ValueError("unit must be 'dB', 'dBm' or 'dBFS'")
        self.interval, self.unit, self.full_scale = interval, unit, full_scale
        super().__init__(**kw)

    def reset(self):
        """Forget all power."""
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            return self.total / self.count if self.count else float("nan")

    @property
    def level(self) -> float:
        """:attr:`power` in ``unit``."""
        return float(self._to_unit(self.power))

    @property
    def readings(self) -> np.ndarray:
        """Power per completed interval, in ``unit``."""
        with self._lock:
            return self._to_unit(np.array(self._readings))

    @property
    def readings_t(self) -> np.ndarray:
        """End time / s of each reading."""
        with self._lock:
            n = len(self._readings)
        return (np.arange(n) + 1) * self._n_int() / self.fs

    def info(self):
        """Return unit and interval."""
        return self.unit + (f" every {si(self.interval, 's')}" if self.interval else "")


def _dd_sums(y, points):
    # decision-directed: power of the nearest points, power of the errors
    d = points[nearest(y, points)]
    return float(np.sum(np.abs(d) ** 2)), float(np.sum(np.abs(y - d) ** 2))


class ConstellationScope(_Sink):
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
    """

    def __init__(self, reference=None, keep=4000, skip=0, **kw):
        self.reference = None if reference is None else np.asarray(reference)
        self.keep, self.skip = int(keep), int(skip)
        super().__init__(**kw)

    def reset(self):
        """Forget symbols and errors."""
        with self._lock:
            self._pts = np.zeros(0, complex)
            self.n = self.count = 0
            self._err = self._ref = 0.0

    def step(self, x):
        """Keep the symbols and add their errors."""
        x = np.asarray(x)
        with self._lock:
            self._pts = np.concatenate([self._pts, x])[-self.keep :]
            j = self.n + np.arange(len(x))
            self.n += len(x)
            if self.reference is not None:
                y = x[j >= self.skip]
                ref, err = _dd_sums(y, self.reference)
                self._ref += ref
                self._err += err
                self.count += len(y)
        return {}

    @property
    def points(self) -> np.ndarray:
        """The newest ``keep`` symbols."""
        with self._lock:
            return self._pts

    @property
    def mer_db(self) -> float:
        """Modulation error ratio / dB: reference power over error power."""
        with self._lock:
            ref, err = self._ref, self._err
        return float(10 * np.log10(ref / err)) if err else float("inf")

    @property
    def evm_percent(self) -> float:
        """RMS error vector over RMS reference / %."""
        with self._lock:
            ref, err = self._ref, self._err
        return float(100 * np.sqrt(err / ref)) if ref else float("nan")

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
        pts = self.points
        ax.plot(pts.real, pts.imag, ".", **kw)
        if self.reference is not None:
            ax.plot(self.reference.real, self.reference.imag, "r+", ms=6)
            ax.set_title(
                f"{self.name}: MER {self.mer_db:.1f} dB, EVM {self.evm_percent:.2f} %",
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


class EyeDiagram(_Sink):
    """Eye diagram: the signal folded every ``sps`` samples, ``span`` symbols wide.

    Keeps the traces (all by default) and plots them as a density map: how
    often the traces pass each point.

    Parameters
    ----------
    sps : int
        Samples per symbol.
    span : int, default 2
        Symbols per trace.
    offset : int, default 0
        Sample where the first trace starts.
    component : {"real", "imag", "abs"}, default "real"
    keep : int, optional
        Keep only the newest ``keep`` traces; default all, as a ``Scope``.
    """

    def __init__(self, sps, span=2, offset=0, component="real", keep=None, **kw):
        if component not in ("real", "imag", "abs"):
            raise ValueError("component must be 'real', 'imag' or 'abs'")
        self.sps, self.span, self.offset = int(sps), int(span), int(offset)
        self.component = component
        self.keep = None if keep is None else int(keep)
        super().__init__(**kw)

    def reset(self):
        """Forget all traces."""
        with self._lock:
            self._buf, self._buf0 = np.zeros(0), 0  # _buf0: absolute index
            self.n = self.traces_seen = 0
            self._chunks = []  # arrays of traces, oldest first

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
            with self._lock:
                self.traces_seen += len(starts)
                self._chunks.append(self._buf[idx])
                if self.keep is not None:
                    self._chunks = [self._traces()[-self.keep :]]
            nxt = starts[-1] + self.sps
            self._buf, self._buf0 = self._buf[nxt - self._buf0 :], nxt
        return {}

    def _traces(self):
        # all kept traces as one array; under the lock
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0] if self._chunks else np.zeros((0, self.length))

    @property
    def traces(self) -> np.ndarray:
        """The kept traces, oldest first, shape ``(traces, length)``."""
        with self._lock:
            return self._traces().copy()

    def density(self, bins=150, interp=8):
        """Return how often the traces pass each point: a 2-D histogram.

        Traces are interpolated linearly between samples, so each one counts
        as a continuous line, not as separate dots.

        Parameters
        ----------
        bins : int, default 150
            Amplitude bins.
        interp : int, default 8
            Points per sample interval.

        Returns
        -------
        t : numpy.ndarray
            Position in the trace / symbols, ``(length - 1) * interp + 1``.
        edges : numpy.ndarray
            Amplitude bin edges, ``bins + 1``.
        counts : numpy.ndarray
            Shape ``(len(t), bins)``.
        """
        with self._lock:
            traces = self._traces()
        nt = (self.length - 1) * interp + 1
        t = np.arange(nt) / interp
        if not len(traces):
            return t / self.sps, np.linspace(-1.0, 1.0, bins + 1), np.zeros((nt, bins))
        lo, hi = float(traces.min()), float(traces.max())
        pad = 0.05 * (hi - lo or 1.0)
        edges = np.linspace(lo - pad, hi + pad, bins + 1)
        i0 = np.minimum(t.astype(int), self.length - 2)
        frac = t - i0
        counts = np.zeros(nt * bins)
        cols = np.arange(nt) * bins
        for k in range(0, len(traces), 10_000):  # blocks: bounded memory
            tr = traces[k : k + 10_000]
            y = tr[:, i0] * (1 - frac) + tr[:, i0 + 1] * frac
            b = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, bins - 1)
            counts += np.bincount((cols + b).ravel(), minlength=nt * bins)
        return t / self.sps, edges, counts.reshape(nt, bins)

    def plot(self, ax=None, bins=150, interp=8, color="C0"):
        """Plot all traces as a density map in one colour.

        Opacity grows with the number of traces through a point (log scale);
        where none pass the axes stay transparent, so the map suits light and
        dark backgrounds alike.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        bins, interp : int
            As for :meth:`density`.
        color : color, default "C0"

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgb

        ax = ax or plt.gca()
        t, edges, counts = self.density(bins, interp)
        level = np.log1p(counts) / max(np.log1p(counts.max()), 1e-300)
        rgba = np.zeros((bins, len(t), 4))
        rgba[..., :3] = to_rgb(color)
        rgba[..., 3] = level.T
        ax.imshow(
            rgba,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(t[0], t[-1], edges[0], edges[-1]),
        )
        ax.set(xlabel="t / T", ylabel=self.component)
        ax.grid(alpha=0.3)
        ax.set_title(f"{self.name}: {self.traces_seen} traces", fontsize=9)
        return ax

    def info(self):
        """Return samples per symbol and span."""
        return f"{self.sps} sps, {self.span} UI"


class SNREstimator(_Sink):
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
        with self._lock:
            self.n = self.count = 0
            self.m2 = self.m4 = 0.0
            self._sig = self._noise = 0.0

    def step(self, x):
        """Accumulate moments or decision errors."""
        with self._lock:
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
                sig_, noise = _dd_sums(y, self.constellation)
                self._sig += sig_
                self._noise += noise
        return {}

    @property
    def snr(self) -> float:
        """Signal-to-noise power ratio (linear)."""
        with self._lock:
            count, m2, m4 = self.count, self.m2, self.m4
            sig_, noise = self._sig, self._noise
        if not count:
            return float("nan")
        if self.method == "dd":
            return sig_ / noise if noise else float("inf")
        M2, M4 = m2 / count, m4 / count
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


class BERCounter(_Sink):
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
        with self._lock:
            self._reset()

    def _reset(self):
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
        with self._lock:
            self._step(desired, received)
        return {}

    def _step(self, desired, received):
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
        with self._lock:
            errors, bits = self.errors, self.bits
        return errors / bits if bits else float("nan")

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

        with self._lock:
            k, n = self.errors, self.bits
        a = 1 - level
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
