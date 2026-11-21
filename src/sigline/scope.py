"""Scope: records the signal on its input line."""

from __future__ import annotations

import numpy as np

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
        if label is not None:
            kw.setdefault("name", label)
        super().__init__(**kw)
        self.label = self.label or self.name

    def reset(self):
        """Forget the recording."""
        self._chunks, self._len, self._dropped, self._seen = [], 0, 0, 0

    def step(self, x):
        """Record one chunk."""
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

    @property
    def data(self) -> np.ndarray:
        """Recorded samples; the newest ``keep`` if set."""
        if not self._chunks:
            return np.empty(0)
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        d = self._chunks[0]
        return d[-self.keep :] if self.keep is not None else d

    @property
    def t(self) -> np.ndarray:
        """Time / s of each recorded sample, absolute since the start."""
        n = len(self.data)
        start = self._dropped + (self._len - n)
        return (start + np.arange(n)) * self.every / self.fs
