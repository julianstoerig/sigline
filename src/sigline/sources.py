"""Sources: blocks without inputs."""

from __future__ import annotations

import numpy as np

from ._format import si
from .core import StateBlock

TWO_PI = 2 * np.pi

__all__ = [
    "Const",
    "FromArray",
    "Impulse",
    "Noise",
    "RandomPulses",
    "Signal",
    "Sine",
    "Source",
    "Step",
]


class Source(StateBlock):
    """Base for sources with a sample counter ``k``; implement :meth:`generate`.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Source
    >>> class Ramp(Source):
    ...     def generate(self, n):
    ...         return n.astype(float)
    >>> r = Ramp()
    >>> r(3), r(2)
    (array([0., 1., 2.]), array([3., 4.]))
    """

    inputs = ()

    def reset(self):
        """Restart at sample 0."""
        self.k = 0

    def output(self, L):
        """Return ``generate`` for the next ``L`` sample indices."""
        return self.generate(np.arange(self.k, self.k + L))

    def update(self, L):
        """Advance the counter."""
        self.k += L

    def generate(self, n):
        """Return the samples at indices ``n`` (an int array).

        Parameters
        ----------
        n : numpy.ndarray
            Absolute sample indices; time is ``n / self.fs``.

        Returns
        -------
        numpy.ndarray
            One sample per index.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate()"
        )


class Const(Source):
    """Constant ``value``.

    Parameters
    ----------
    value : float or complex, default 1.0
    """

    def __init__(self, value=1.0, **kw):
        self.value = value
        super().__init__(**kw)

    def generate(self, n):
        """Return ``value`` for every index."""
        return np.full(len(n), self.value)

    def info(self):
        """Return the value."""
        return f"{self.value:g}"


class Step(Source):
    """``before`` until time ``T``, then ``after``.

    Parameters
    ----------
    T : float, default 0.0
        Switching time / s; the first sample with ``n / fs >= T`` is ``after``.
    after : float, default 1.0
    before : float, default 0.0
    """

    def __init__(self, T=0.0, after=1.0, before=0.0, **kw):
        self.T, self.after, self.before = T, after, before
        super().__init__(**kw)

    def generate(self, n):
        """Return ``before`` or ``after`` per index."""
        return np.where(n / self.fs >= self.T, self.after, self.before).astype(float)

    def info(self):
        """Return ``before→after @ T``."""
        return f"{self.before:g}→{self.after:g} @ {si(self.T, 's')}"


class Impulse(Source):
    """Unit sample: ``amp`` at time ``T``, zero elsewhere.

    Parameters
    ----------
    T : float, default 0.0
        Time / s; the impulse is at the first sample with ``n / fs >= T``.
    amp : float, default 1.0
    """

    def __init__(self, T=0.0, amp=1.0, **kw):
        self.T, self.amp = T, amp
        super().__init__(**kw)

    def generate(self, n):
        """Return ``amp`` at the impulse's sample, zero elsewhere."""
        k = int(np.ceil(self.T * self.fs - 1e-9))
        return np.where(n == k, self.amp, 0.0)

    def info(self):
        """Return time and amplitude."""
        amp = f"{self.amp:g} " if self.amp != 1 else ""
        return f"{amp}@ {si(self.T, 's')}"


class Signal(Source):
    """Any vectorised function of time.

    Parameters
    ----------
    fn : callable
        ``fn(t)`` with ``t`` an array of times / s.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Signal
    >>> Signal(lambda t: t**2, fs=2.0)(4)
    array([0.  , 0.25, 1.  , 2.25])
    """

    def __init__(self, fn, **kw):
        self.fn = fn
        super().__init__(**kw)

    def generate(self, n):
        """Return ``fn(n / fs)``."""
        return self.fn(n / self.fs)

    def info(self):
        """Return the function's name."""
        name = getattr(self.fn, "__name__", "")
        return "" if name == "<lambda>" else name


class FromArray(Source):
    """Play back an array, e.g. recorded IQ; zeros or a loop after the end.

    Parameters
    ----------
    data : array_like
        Samples; axis 0 is time, extra axes are kept.
    loop : bool, default False
        Repeat ``data`` instead of padding with zeros.

    Examples
    --------
    >>> from sigline import FromArray
    >>> FromArray([1.0, 2.0])(4), FromArray([1.0, 2.0], loop=True)(3)
    (array([1., 2., 0., 0.]), array([1., 2., 1.]))
    """

    def __init__(self, data, loop=False, **kw):
        self.data, self.loop = np.asarray(data), loop
        super().__init__(**kw)

    def generate(self, n):
        """Return ``data[n]``, zero (or wrapped) past the end."""
        if self.loop:
            return self.data[n % len(self.data)]
        out = np.zeros((len(n),) + self.data.shape[1:], dtype=self.data.dtype)
        ok = n < len(self.data)
        out[ok] = self.data[n[ok]]
        return out

    def info(self):
        """Return the length."""
        return f"{len(self.data)} samples" + (" loop" if self.loop else "")


class Sine(Source):
    """Phase-continuous oscillator ``amp cos(phi)``, or ``amp exp(j phi)``.

    Changing ``f`` between runs keeps the phase continuous.

    Parameters
    ----------
    f : float
        Frequency / Hz.
    amp : float, default 1.0
    phase : float, default 0.0
        Start phase / rad.
    complex : bool, default False
        Complex exponential instead of cosine.
    """

    def __init__(self, f, amp=1.0, phase=0.0, complex=False, **kw):
        self.f, self.amp, self.phase0, self.complex = f, amp, phase, complex
        super().__init__(**kw)

    def reset(self):
        """Restart at the start phase."""
        super().reset()
        self.phi = self.phase0

    def output(self, L):
        """Return the next ``L`` samples."""
        ph = self.phi + TWO_PI * self.f / self.fs * np.arange(L)
        return self.amp * (np.exp(1j * ph) if self.complex else np.cos(ph))

    def update(self, L):
        """Advance the phase by ``L`` samples."""
        super().update(L)
        self.phi = (self.phi + TWO_PI * self.f / self.fs * L) % TWO_PI

    def info(self):
        """Return frequency, kind and amplitude."""
        amp = f" ×{self.amp:g}" if self.amp != 1 else ""
        return si(self.f, "Hz") + (" cplx" if self.complex else "") + amp


class Noise(Source):
    """White Gaussian noise, reproducible after ``reset()``.

    Complex noise is circular, drawn as (I, Q) pairs from one stream, so
    chunking cannot change it.

    Parameters
    ----------
    std : float, default 1.0
        Standard deviation; for complex noise ``E|z|^2 = std^2``.
    seed : int, default 0
    complex : bool, default False
    width : int, optional
        Channels of a vector signal ``(L, width)``, independent of each
        other; default one value per sample.
    """

    def __init__(self, std=1.0, seed=0, complex=False, width=None, **kw):
        self.std, self.seed, self.complex, self.width = std, seed, complex, width
        super().__init__(**kw)

    def reset(self):
        """Reseed."""
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self._next = None

    def output(self, L):
        """Return the next ``L`` samples (drawn once per chunk)."""
        if self._next is None or len(self._next) != L:
            shape = (L,) if self.width is None else (L, int(self.width))
            if self.complex:
                iq = self.rng.standard_normal((*shape, 2))
                self._next = (iq[..., 0] + 1j * iq[..., 1]) / np.sqrt(2)
            else:
                self._next = self.rng.standard_normal(shape)
        return self.std * self._next

    def update(self, L):
        """Advance to the next chunk."""
        super().update(L)
        self._next = None

    def info(self):
        """Return the standard deviation."""
        width = f" ×{self.width}" if self.width is not None else ""
        return f"σ={self.std:g}" + (" cplx" if self.complex else "") + width


class RandomPulses(Source):
    """Pulses of 1 that start at random, 0 between them.

    Each sample starts a pulse with probability ``p`` (a Bernoulli process);
    a pulse lasts ``duration`` and may run on into the next chunk. Pulses
    that overlap merge.

    Parameters
    ----------
    rate : float, optional
        Mean starts per second; ``p = rate / fs``.
    p : float, optional
        Start probability per sample, instead of ``rate``.
    duration : float, default 0.0
        Pulse length / s; at least one sample.
    seed : int, default 0

    Raises
    ------
    ValueError
        Not exactly one of ``rate`` and ``p``.

    Examples
    --------
    >>> from sigline import RandomPulses
    >>> RandomPulses(p=1.0, fs=1.0)(3), RandomPulses(p=0.0, fs=1.0)(3)
    (array([1., 1., 1.]), array([0., 0., 0.]))
    """

    def __init__(self, rate=None, p=None, duration=0.0, seed=0, **kw):
        if (rate is None) == (p is None):
            raise ValueError("give exactly one of rate= or p=")
        # not self.rate: rates of blocks are sample rates
        self.pulse_rate, self.p_start, self.duration, self.seed = (
            rate,
            p,
            duration,
            seed,
        )
        super().__init__(**kw)

    def reset(self):
        """Reseed; no pulse running."""
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self.remaining = 0  # samples of a pulse still to come
        self._next = None

    @property
    def p(self) -> float:
        """Start probability per sample."""
        return self.p_start if self.p_start is not None else self.pulse_rate / self.fs

    def _length(self):
        return max(1, round(self.duration * self.fs))

    def output(self, L):
        """Return the next ``L`` samples (drawn once per chunk)."""
        if self._next is None or len(self._next[0]) != L:
            n, i = self._length(), np.arange(L)
            # last start at or before each sample; a running pulse counts as
            # one that started remaining - n samples ago
            start = np.where(self.rng.random(L) < self.p, i, -n)
            start[0] = max(start[0], self.remaining - n)
            last = np.maximum.accumulate(start)
            on = (i - last < n).astype(float)
            self._next = (on, max(0, int(last[-1]) + n - L))
        return self._next[0]

    def update(self, L):
        """Advance to the next chunk."""
        self.output(L)
        super().update(L)
        self.remaining = self._next[1]
        self._next = None

    def info(self):
        """Return the start rate and the duration."""
        how = (
            f"{self.pulse_rate:g}/s"
            if self.pulse_rate is not None
            else f"p={self.p_start:g}"
        )
        return f"{how}, {si(self.duration, 's')}"
