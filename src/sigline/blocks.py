"""Math and general-purpose blocks."""

from __future__ import annotations

from .core import Block

__all__ = ["Gain"]


class Gain(Block):
    """Multiply by ``g``.

    Parameters
    ----------
    g : float, complex or array_like
        Scalar, or an array broadcast against each chunk (e.g. one gain per
        vector channel).
    """

    def __init__(self, g, **kw):
        self.g = g
        super().__init__(**kw)

    def step(self, x):
        """Return ``g * x``."""
        return self.g * x
