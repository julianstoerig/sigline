"""Sources: blocks without inputs."""

from __future__ import annotations

import numpy as np

from .core import StateBlock

__all__ = ["Const", "Source"]


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
