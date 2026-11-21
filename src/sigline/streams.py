"""Blocks that exchange samples with other threads: devices, sound cards, apps."""

from __future__ import annotations

import threading
import time

import numpy as np

from .core import Block

__all__ = ["FromQueue", "SampleBuffer", "ToQueue"]


class SampleBuffer:
    """Thread-safe FIFO of samples with a capacity.

    Parameters
    ----------
    capacity : int, default 2**20
        Samples kept; pushing more drops the oldest (counted in ``overflows``).
    dtype : numpy dtype, default float
        Of the zeros that fill an underrun before the first push.

    Attributes
    ----------
    overflows : int
        Samples dropped because the buffer was full.
    underruns : int
        Samples filled with zeros because none had arrived.
    """

    def __init__(self, capacity=2**20, dtype=float):
        self.capacity, self._dtype0 = int(capacity), np.dtype(dtype)
        self._cond = threading.Condition()
        self._woken = False
        self.clear()

    def clear(self):
        """Drop all samples and zero the counters."""
        with self._cond:
            self._chunks, self._n = [], 0
            self.dtype, self._shape = self._dtype0, ()
            self.overflows = self.underruns = 0

    def __len__(self):
        """Return the number of buffered samples."""
        return self._n

    def push(self, x):
        """Append samples (axis 0 is time); safe from any thread."""
        x = np.array(x, copy=True)
        with self._cond:
            if not self._n:
                self.dtype, self._shape = x.dtype, x.shape[1:]
            self._chunks.append(x)
            self._n += len(x)
            excess = self._n - self.capacity
            if excess > 0:
                self.overflows += excess
                self._drop(excess)
            self._cond.notify_all()

    def _drop(self, k):
        while k:
            first = self._chunks[0]
            if len(first) <= k:
                self._chunks.pop(0)
                self._n -= len(first)
                k -= len(first)
            else:
                self._chunks[0] = first[k:]
                self._n -= k
                k = 0

    def pull(self, n, timeout=0.0):
        """Return exactly ``n`` samples, zero-filled if they do not arrive in time.

        Parameters
        ----------
        n : int
        timeout : float or None, default 0.0
            Seconds to wait for missing samples; ``None`` waits until they
            arrive or :meth:`wake` is called.

        Returns
        -------
        numpy.ndarray
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._n < n and not self._woken:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    break
                self._cond.wait(left)
            k = min(n, self._n)
            out = np.zeros((n,) + self._shape, dtype=self.dtype)
            if k:
                data = np.concatenate(self._chunks)
                out[:k] = data[:k]
                self._chunks, self._n = [data[k:]], self._n - k
            self.underruns += n - k
            return out

    def wake(self):
        """End waiting pulls, now and later, until :meth:`rearm`."""
        with self._cond:
            self._woken = True
            self._cond.notify_all()

    def rearm(self):
        """Let :meth:`pull` wait again after :meth:`wake`."""
        with self._cond:
            self._woken = False


class FromQueue(Block):
    """Source fed from another thread with :meth:`push`.

    With ``timeout=None`` the diagram waits for samples, so their arrival
    paces it (``external_clock``); ``Diagram.start`` then skips wall-clock
    pacing.

    Parameters
    ----------
    timeout : float or None, default None
        Seconds to wait per chunk before filling with zeros.
    capacity : int, default 2**20
        Buffered samples before the oldest are dropped.
    dtype : numpy dtype, default float
        Of zeros before the first push.

    Examples
    --------
    >>> from sigline import FromQueue
    >>> q = FromQueue(timeout=0)
    >>> q.push([1.0, 2.0, 3.0])
    >>> q(2), q(2), q.underruns
    (array([1., 2.]), array([3., 0.]), 1)
    """

    inputs = ()

    def __init__(self, timeout=None, capacity=2**20, dtype=float, **kw):
        self.timeout = timeout
        self.buffer = SampleBuffer(capacity, dtype)
        super().__init__(**kw)

    @property
    def external_clock(self) -> bool:
        """``True`` when the block waits for data, so data paces the diagram."""
        return self.timeout is None

    def reset(self):
        """Drop buffered samples (stale input) and zero the counters."""
        self.buffer.clear()

    def push(self, x):
        """Append samples; call from any thread."""
        self.buffer.push(x)

    def open(self):
        """Allow waiting again after a stop."""
        self.buffer.rearm()

    def step_source(self, L):
        """Return the next ``L`` samples."""
        return self.buffer.pull(L, self.timeout)

    def wake(self):
        """Interrupt a wait; called by ``Diagram.stop``."""
        self.buffer.wake()

    @property
    def underruns(self) -> int:
        """Samples filled with zeros."""
        return self.buffer.underruns

    @property
    def overflows(self) -> int:
        """Samples dropped because the buffer was full."""
        return self.buffer.overflows


class ToQueue(Block):
    """Sink read from another thread with :meth:`pull`.

    Parameters
    ----------
    capacity : int, default 2**20
        Buffered samples before the oldest are dropped.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import ToQueue
    >>> q = ToQueue()
    >>> _ = q(np.array([1.0, 2.0]))
    >>> q.pull(3), q.underruns
    (array([1., 2., 0.]), 1)
    """

    outputs = ()

    def __init__(self, capacity=2**20, **kw):
        self.buffer = SampleBuffer(capacity)
        super().__init__(**kw)

    def reset(self):
        """Drop buffered samples and zero the counters."""
        self.buffer.clear()

    def step(self, x):
        """Buffer one chunk."""
        self.buffer.push(x)
        return {}

    def pull(self, n, timeout=0.0):
        """Return ``n`` samples, zero-filled if fewer are buffered; any thread.

        Parameters
        ----------
        n : int
        timeout : float or None, default 0.0
            Seconds to wait for missing samples.

        Returns
        -------
        numpy.ndarray
        """
        return self.buffer.pull(n, timeout)

    @property
    def underruns(self) -> int:
        """Samples pulled as zeros because none were buffered."""
        return self.buffer.underruns

    @property
    def overflows(self) -> int:
        """Samples dropped because nobody pulled them in time."""
        return self.buffer.overflows
