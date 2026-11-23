"""Math and general-purpose blocks."""

from __future__ import annotations

import numpy as np

from ._format import si
from .core import Block, StateBlock, _arguments, _parameters

__all__ = ["Delay", "Func", "Gain", "Product", "Sum", "Unwrap"]


def _expand(vals):
    # signals with one value per sample, next to vector signals: repeat them
    # over the channels (plain broadcasting would align them with the
    # channels instead, wrongly, whenever the chunk is as long as the width)
    nd = max(np.ndim(v) for v in vals)
    return [
        np.reshape(v, np.shape(v) + (1,) * (nd - 1)) if np.ndim(v) == 1 else v
        for v in vals
    ]


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

    def info(self):
        """Return the gain."""
        if not np.isscalar(self.g):
            return "×g"
        if np.iscomplexobj(self.g) and np.real(self.g) == 0:
            return f"×{np.imag(self.g):g}j"
        return f"×{self.g:g}"


class Sum(Block):
    """Add and subtract inputs ``x0, x1, ...`` by ``signs``.

    Signals with one value per sample add to every channel of a vector
    signal.

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
    expands_scalars = True

    def __init__(self, signs="++", **kw):
        if not signs or set(signs) - {"+", "-"}:
            raise ValueError(f"signs must be '+' and '-' only, not {signs!r}")
        self.signs = signs
        self.inputs = tuple(f"x{i}" for i in range(len(signs)))
        super().__init__(**kw)

    def step(self, **u):
        """Return the signed sum."""
        vals = _expand([u[k] for k in self.inputs])  # port order, not keyword order
        return sum(v if s == "+" else -v for s, v in zip(self.signs, vals, strict=True))

    def info(self):
        """Return the signs."""
        return " ".join(self.signs)


class Product(Block):
    """Multiply and divide inputs ``x0, x1, ...``.

    Signals with one value per sample scale every channel of a vector
    signal.

    Parameters
    ----------
    ops : int or str, default 2
        Number of inputs to multiply, or one ``*`` or ``/`` per input:
        ``Product("*/")`` gives ``x0 / x1``.

    Raises
    ------
    ValueError
        ``ops`` empty or not made of ``*`` and ``/``.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Product
    >>> Product("**/")(np.array([6.0]), np.array([2.0]), np.array([4.0]))
    array([3.])
    """

    same_shape_inputs = True
    expands_scalars = True

    def __init__(self, ops=2, **kw):
        if isinstance(ops, int):
            ops = "*" * ops
        if not ops or set(ops) - {"*", "/"}:
            raise ValueError(f"ops must be '*' and '/' only, not {ops!r}")
        self.ops = ops
        self.inputs = tuple(f"x{i}" for i in range(len(ops)))
        super().__init__(**kw)

    def step(self, **u):
        """Return the product, dividing where ``ops`` says ``/``."""
        out = 1
        for op, v in zip(self.ops, _expand([u[k] for k in self.inputs]), strict=True):
            out = out * v if op == "*" else out / v
        return out

    def info(self):
        """Return ``×``, or the ops if some divide."""
        return "×" if "/" not in self.ops else " ".join(self.ops)


class Func(Block):
    """Wrap a vectorised function: called once per chunk, on whole arrays.

    Required arguments are the inputs; arguments with defaults are
    parameters, passed by keyword, as for :class:`Nonlinear`. ``Func`` suits
    what NumPy computes on arrays (``np.abs``, ``np.clip``), which is the
    fastest; ``Nonlinear`` takes one sample at a time, for branches and
    state. ``fn`` must treat samples independently, or results depend on the
    chunking.

    Parameters
    ----------
    fn : callable
        Called once per chunk with one array per input.
    outputs : tuple of str, default ("y",)
        Output names; ``fn`` returns a tuple for several.
    inputs : tuple of str, optional
        Input names; default: ``fn``'s required positional arguments, or
        ``("x",)`` without a signature (NumPy ufuncs).
    params : dict, optional
        Values for ``fn``'s parameters; ``blk.params`` changes them for the
        next chunk.

    Attributes
    ----------
    params : dict
        Parameter values.

    Raises
    ------
    TypeError
        ``params`` names an argument ``fn`` does not have.

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
    >>> scale = Func(lambda x, k=2.0: k * x)
    >>> scale.params["k"] = 3.0
    >>> scale(np.array([1.0]))
    array([3.])
    """

    def __init__(self, fn, outputs=("y",), inputs=None, params=None, **kw):
        self.fn = fn
        req, defaults = _arguments(fn) or (("x",), {})
        self.inputs = tuple(req if inputs is None else inputs)
        self.outputs = tuple(outputs)
        self.params = _parameters(fn, defaults, params, self.inputs)
        name = getattr(fn, "__name__", "<lambda>")
        kw.setdefault("name", None if name == "<lambda>" else name)
        super().__init__(**kw)

    def step(self, **u):
        """Return ``fn(*inputs, **params)``."""
        return self.fn(*(u[k] for k in self.inputs), **self.params)

    def info(self):
        """Return the function's name unless it is the block's name."""
        name = getattr(self.fn, "__name__", "")
        return "" if name in (self.name, "<lambda>") else name


class Delay(StateBlock):
    """Delay by ``n`` samples or ``tau`` seconds; breaks loops when ``n >= chunk``.

    Parameters
    ----------
    n : int, default 1
    init : float or array_like, default 0.0
        Output for the first ``n`` samples. An array, one value per channel,
        makes the output a vector signal from the start, e.g. in a loop of
        vector signals.
    tau : float, optional
        Delay / s instead of ``n``; ``tau * fs`` must be a whole number.

    Raises
    ------
    ValueError
        ``tau * fs`` is not a whole number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Delay
    >>> Delay(tau=0.002, fs=1000.0)(np.arange(1.0, 5.0))
    array([0., 0., 1., 2.])
    """

    def __init__(self, n=1, init=0.0, *, tau=None, **kw):
        self._n, self.init, self.tau = int(n), init, tau
        super().__init__(**kw)

    @property
    def n(self) -> int:
        """Delay / samples."""
        if self.tau is None:
            return self._n
        if self.fs is None:
            raise ValueError(f"{self.name}: tau needs fs; put the delay in a diagram")
        n = round(self.tau * self.fs)
        if abs(n - self.tau * self.fs) > 1e-6:
            raise ValueError(
                f"{self.name}: tau * fs = {self.tau * self.fs:g} is not a whole "
                "number of samples"
            )
        return n

    @n.setter
    def n(self, value):
        self._n, self.tau = int(value), None

    def reset(self):
        """Fill the buffer with ``init`` (once the length is known)."""
        self.buf = None

    def feedthrough(self, L):
        """Return ``n < L``: a short delay needs this chunk's input."""
        return self.n < L

    def _ensure(self):
        if self.buf is None:
            init = np.asarray(self.init)
            self.buf = np.broadcast_to(init, (self.n,) + init.shape).copy()

    def _match(self, x):
        # grow the buffer to vector signals on first use
        self._ensure()
        if self.buf.ndim < x.ndim:
            shape = (self.n,) + x.shape[1:]
            self.buf = np.broadcast_to(
                self.buf.reshape((self.n,) + (1,) * (x.ndim - 1)), shape
            ).copy()

    def output(self, L):
        """Return the next ``L`` buffered samples."""
        self._ensure()
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

    def info(self):
        """Return ``z^-n``, or the delay in seconds."""
        return f"z^-{self._n}" if self.tau is None else si(self.tau, "s")


class Unwrap(Block):
    """Unwrap a phase across chunks: remove its jumps of 2 pi.

    Equals ``numpy.unwrap`` of the whole signal, however it is chunked.

    Parameters
    ----------
    skip : int, default 0
        Pass the first ``skip`` samples through wrapped, e.g. a filter's
        start-up transient, and unwrap from there on.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Unwrap
    >>> phase = np.angle(np.exp(1j * np.arange(0.0, 12.0, 2.0)))
    >>> u = Unwrap()
    >>> np.round(np.concatenate([u(phase[:3]), u(phase[3:])]), 12)
    array([ 0.,  2.,  4.,  6.,  8., 10.])
    """

    real_inputs = True

    def __init__(self, skip=0, **kw):
        self.skip = int(skip)
        super().__init__(**kw)

    def reset(self):
        """Forget the last sample and the offset."""
        self._n, self._prev, self._offset = 0, None, None

    def step(self, x):
        """Return the unwrapped chunk."""
        x = np.asarray(x, dtype=float)
        if not len(x):
            return x.copy()
        prev = x[:1] if self._prev is None else self._prev[None]
        # as numpy.unwrap: corrections of 2 pi wherever a step exceeds pi
        d = np.diff(np.concatenate([prev, x]), axis=0)
        dm = np.mod(d + np.pi, 2 * np.pi) - np.pi
        dm[(dm == -np.pi) & (d > 0)] = np.pi
        corr = dm - d
        corr[np.abs(d) < np.pi] = 0.0
        corr[self._n + np.arange(len(x)) < self.skip] = 0.0
        off = np.zeros(x.shape[1:]) if self._offset is None else self._offset
        # summed on from the offset, so chunks add up as one signal would
        acc = np.cumsum(np.concatenate([off[None], corr]), axis=0)[1:]
        self._n += len(x)
        self._prev, self._offset = x[-1], acc[-1]
        return x + acc

    def info(self):
        """Return the skip."""
        return f"skip {self.skip}" if self.skip else ""
