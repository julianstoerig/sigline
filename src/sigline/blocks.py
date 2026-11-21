"""Math and general-purpose blocks."""

from __future__ import annotations

from .core import Block

__all__ = ["Gain", "Product", "Sum"]


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


class Sum(Block):
    """Add and subtract inputs ``x0, x1, ...`` by ``signs``.

    Parameters
    ----------
    signs : str, default "++"
        One ``+`` or ``-`` per input: ``Sum("+-")`` gives ``x0 - x1``.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Sum
    >>> Sum("+-+")(np.array([1.0]), np.array([2.0]), np.array([4.0]))
    array([3.])
    """

    same_shape_inputs = True

    def __init__(self, signs="++", **kw):
        self.signs = signs
        self.inputs = tuple(f"x{i}" for i in range(len(signs)))
        super().__init__(**kw)

    def step(self, **u):
        """Return the signed sum."""
        return sum(
            v if s == "+" else -v for s, v in zip(self.signs, u.values(), strict=True)
        )


class Product(Block):
    """Multiply inputs ``x0, x1, ...``.

    Parameters
    ----------
    n : int, default 2
        Number of inputs.
    """

    same_shape_inputs = True

    def __init__(self, n=2, **kw):
        self.inputs = tuple(f"x{i}" for i in range(n))
        super().__init__(**kw)

    def step(self, **u):
        """Return the product of all inputs."""
        out = 1
        for v in u.values():
            out = out * v
        return out
