"""Math and general-purpose blocks."""

from __future__ import annotations

import inspect

from .core import Block

__all__ = ["Func", "Gain", "Product", "Sum"]


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


class Func(Block):
    """Wrap a vectorised function; its required arguments are the inputs.

    Parameters
    ----------
    fn : callable
        Called once per chunk with one array per input.
    outputs : tuple of str, default ("y",)
        Output names; ``fn`` returns a tuple for several.
    inputs : tuple of str, optional
        Input names; default: ``fn``'s required positional arguments, or
        ``("x",)`` without a signature (NumPy ufuncs).

    Notes
    -----
    The default output is ``y``, so ``Func(lambda y: ...)`` names a port both
    input and output; use ``.inp("y")`` / ``.out("y")``.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Func
    >>> iq = Func(lambda i, q: i + 1j * q)
    >>> iq.inputs, iq(np.array([1.0]), np.array([2.0]))
    (('i', 'q'), array([1.+2.j]))
    >>> pol = Func(lambda z: (np.abs(z), np.angle(z)), outputs=("mag", "ph"))
    >>> pol(np.array([1j]))
    (array([1.]), array([1.57079633]))
    """

    def __init__(self, fn, outputs=("y",), inputs=None, **kw):
        self.fn = fn
        if inputs is None:
            try:
                ps = inspect.signature(fn).parameters.values()
                inputs = tuple(
                    p.name
                    for p in ps
                    if p.default is p.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                )
            except (TypeError, ValueError):  # numpy ufuncs have no signature
                inputs = ("x",)
        self.inputs, self.outputs = tuple(inputs), tuple(outputs)
        name = getattr(fn, "__name__", "<lambda>")
        kw.setdefault("name", None if name == "<lambda>" else name)
        super().__init__(**kw)

    def step(self, **u):
        """Return ``fn(*inputs)``."""
        return self.fn(*u.values())
