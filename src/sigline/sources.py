"""Sources: blocks without inputs."""

from __future__ import annotations

import numpy as np

from .core import StateBlock

__all__ = ["Const", "FromArray", "Signal", "Source", "Step"]


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
