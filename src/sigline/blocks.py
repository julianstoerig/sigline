"""Math and general-purpose blocks."""

from __future__ import annotations

import inspect

import numpy as np

from .core import Block, StateBlock

TWO_PI = 2 * np.pi

__all__ = ["Delay", "Func", "Gain", "Mixer", "Product", "Sum"]


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

    Raises
    ------
    ValueError
        ``signs`` empty or not made of ``+`` and ``-``.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Sum
    >>> Sum("+-+")(np.array([1.0]), np.array([2.0]), np.array([4.0]))
    array([3.])
    """

    same_shape_inputs = True

    def __init__(self, signs="++", **kw):
        if not signs or set(signs) - {"+", "-"}:
            raise ValueError(f"signs must be '+' and '-' only, not {signs!r}")
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


class Delay(StateBlock):
    """Delay by ``n`` samples; breaks feedback loops when ``n >= chunk``.

    Parameters
    ----------
    n : int, default 1
    init : float, default 0.0
        Output for the first ``n`` samples.
    """

    def __init__(self, n=1, init=0.0, **kw):
        self.n, self.init = int(n), init
        super().__init__(**kw)

    def reset(self):
        """Fill the buffer with ``init``."""
        self.buf = np.full(self.n, self.init)

    def feedthrough(self, L):
        """Return ``n < L``: a short delay needs this chunk's input."""
        return self.n < L

    def _match(self, x):
        # grow the buffer to vector signals on first use
        if self.buf.ndim < x.ndim:
            shape = (self.n,) + x.shape[1:]
            self.buf = np.broadcast_to(
                self.buf.reshape((self.n,) + (1,) * (x.ndim - 1)), shape
            ).copy()

    def output(self, L):
        """Return the next ``L`` buffered samples."""
        return self.buf[:L]

    def update(self, L, x):
        """Shift ``x`` into the buffer."""
        self._match(x)
        self.buf = np.concatenate([self.buf[L:], x])

    def step(self, x):
        """Delay one chunk (any length)."""
        self._match(x)
        z = np.concatenate([self.buf, x])
        self.buf = z[len(x) :]
        return z[: len(x)]


class Mixer(Block):
    """Multiply by a phase-continuous local oscillator.

    Parameters
    ----------
    f : float
        LO frequency / Hz; changes between runs keep the phase continuous.
    lo : {"cos", "sin", "exp", "exp-"}, default "cos"
        ``exp`` is ``e^{+j phi}`` (up-conversion), ``exp-`` is ``e^{-j phi}``
        (down-conversion to complex baseband).
    phase : float, default 0.0
        Start phase / rad.
    """

    _LO = {
        "cos": np.cos,
        "sin": np.sin,
        "exp": lambda p: np.exp(1j * p),
        "exp-": lambda p: np.exp(-1j * p),
    }

    def __init__(self, f, lo="cos", phase=0.0, **kw):
        self.f, self.lo, self.phase0 = f, lo, phase
        super().__init__(**kw)

    def reset(self):
        """Restart at the start phase."""
        self.phi = self.phase0

    def step(self, x):
        """Return ``x`` times the LO."""
        L = len(x)
        ph = self.phi + TWO_PI * self.f / self.fs * np.arange(L)
        self.phi = (self.phi + TWO_PI * self.f / self.fs * L) % TWO_PI
        lo = self._LO[self.lo](ph)
        return x * lo.reshape((L,) + (1,) * (x.ndim - 1))
