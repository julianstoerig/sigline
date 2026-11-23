"""Per-sample nonlinear blocks, optionally stateful, compiled with numba.

Write the nonlinearity for one sample in plain scalar Python: ``if``/``else``,
``math``, ``cmath``, NumPy scalar functions::

    def soft_clip(x):                            # stateless: y = f(x)
        return math.tanh(x)

    def schmitt(x, state, hi=0.5, lo=-0.5):      # stateful: (y, state) = f(x, state)
        if x > hi:
            state = 1.0
        elif x < lo:
            state = -1.0
        return state, state

    Nonlinear(soft_clip)
    Nonlinear(schmitt, state=-1.0)               # blk.params["hi"] = 0.7

The block generates a loop over the chunk around ``f`` and compiles both
with numba when it can. The same loop runs in Python otherwise, so results
are identical.
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np

from ._numba import HAVE_NUMBA, callable_in_numba
from .core import Block, _arguments, _parameters

if HAVE_NUMBA:
    import numba

__all__ = ["Nonlinear", "state"]

_LOOPS: dict = {}


def _loop_source(n_in, n_out, stateful, pnames):
    xs = [f"x{i}" for i in range(n_in)]
    os_ = [f"o{j}" for j in range(n_out)]
    ps = [f"p{k}" for k in range(len(pnames))]
    args = ", ".join(f"{x}[i]" for x in xs)
    if stateful:
        args += ", s"
    args += "".join(f", {n}={p}" for n, p in zip(pnames, ps, strict=True))
    lines = [
        f"def loop({', '.join(xs + os_ + ['s'] + ps)}):",
        f"    for i in range({xs[0]}.shape[0]):",
        f"        r = f({args})",
    ]
    lines += ["        y = r[0]", "        s = r[1]"] if stateful else ["        y = r"]
    if n_out == 1:
        lines += ["        o0[i] = y"]
    else:
        lines += [f"        o{j}[i] = y[{j}]" for j in range(n_out)]
    lines += ["    return s"]
    return "\n".join(lines)


def _build(f, n_in, n_out, stateful, pnames, jit):
    # (f, loop over a chunk), compiled or not; cached
    key = (f, n_in, n_out, stateful, tuple(pnames), jit)
    if key not in _LOOPS:
        fn = callable_in_numba(f) if jit else f
        ns = {"f": fn}
        exec(_loop_source(n_in, n_out, stateful, pnames), ns)
        _LOOPS[key] = (fn, numba.njit(ns["loop"]) if jit else ns["loop"])
    return _LOOPS[key]


def state(**fields):
    """Return named state for :class:`Nonlinear`: ``state(phase=0.0, freq=0.0)``.

    Inside ``f``, read ``s.phase`` and return ``type(s)(new_phase, new_freq)``;
    numba builds namedtuples but has no ``_replace``.

    Parameters
    ----------
    **fields
        Initial values; ints become floats so numba types stay stable.

    Returns
    -------
    tuple
        A namedtuple ``State``.

    Examples
    --------
    >>> from sigline import state
    >>> s = state(phase=0, freq=0.5)
    >>> s.phase, s.freq
    (0.0, 0.5)
    """
    T = namedtuple("State", list(fields))
    return _copy_state(T(*fields.values()))


def _copy_state(s):
    # deep copy, so f, dtype probes and reset() never change the initial state
    if s is None:
        return None
    if isinstance(s, np.ndarray):
        return s.copy()
    if hasattr(s, "_fields"):
        return type(s)(*(_copy_state(v) for v in s))
    if isinstance(s, tuple | list):
        return tuple(_copy_state(v) for v in s)
    if isinstance(s, int | np.integer) and not isinstance(s, bool):
        return float(s)
    return s


class Nonlinear(Block):
    """Apply a scalar function ``f`` sample by sample, optionally with state.

    Parameters
    ----------
    f : callable
        Function of one sample. Leading arguments without defaults are the
        inputs, in order. Arguments with defaults are parameters.
    state : float, tuple, numpy.ndarray or namedtuple, optional
        Initial state. Then the last argument without a default receives the
        state, and ``f`` returns ``(y, new_state)``. Arrays may be changed in
        place; :func:`state` makes named state.
    outputs : tuple of str, default ("y",)
        Several outputs: ``f`` returns ``y`` as a tuple.
    inputs : tuple of str, optional
        Input names; default: ``f``'s required arguments (but the state).
    params : dict, optional
        Values for ``f``'s parameters. ``blk.params`` changes them for the
        next chunk.
    jit : {"auto", True, False}, default "auto"
        ``"auto"`` compiles with numba if installed and falls back to
        Python with a warning if ``f`` does not compile. True requires
        numba; False runs Python, e.g. for breakpoints in ``f``.
    dtype : dtype or tuple of dtype, optional
        Output types, one or one per output. Without it the block calls
        ``f`` once per chunk, on a copy of the state, to find them.

    Attributes
    ----------
    params : dict
        Parameter values.
    state
        The current state.
    mode : str or None
        ``"numba"`` or ``"python"`` after the first chunk.

    Raises
    ------
    ImportError
        ``jit=True`` without numba.
    TypeError
        A stateful ``f`` without ``(x, state)`` arguments, or unknown
        ``params``.
    ValueError
        ``dtype`` with the wrong number of entries.

    Notes
    -----
    Use :class:`Func` for functions NumPy vectorises (``np.tanh``,
    ``np.clip``): its SIMD kernels are faster. ``Nonlinear`` suits scalar
    logic and memory: hysteresis, PLLs, AGCs, slew limits.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Nonlinear
    >>> def schmitt(x, s, hi=0.5, lo=-0.5):
    ...     if x > hi:
    ...         s = 1.0
    ...     elif x < lo:
    ...         s = -1.0
    ...     return s, s
    >>> Nonlinear(schmitt, state=-1.0)(np.array([0.0, 0.7, 0.0, -0.7, 0.0]))
    array([-1.,  1.,  1., -1., -1.])
    """

    def __init__(
        self,
        f,
        state=None,
        *,
        outputs=("y",),
        inputs=None,
        params=None,
        jit="auto",
        dtype=None,
        **kw,
    ):
        if jit not in ("auto", True, False):
            raise ValueError(f"jit must be 'auto', True or False, not {jit!r}")
        if jit is True and not HAVE_NUMBA:
            raise ImportError("Nonlinear(jit=True) needs numba: pip install numba")
        self.f, self.state0, self.jit = f, state, jit
        if dtype is not None:
            dts = dtype if isinstance(dtype, list | tuple) else [dtype] * len(outputs)
            if len(dts) != len(outputs):
                raise ValueError(f"dtype needs one entry per output {tuple(outputs)}")
            self.dtype = [np.dtype(t) for t in dts]
        else:
            self.dtype = None
        self.stateful = state is not None
        req, defaults = _arguments(f) or (["x"], {})
        if self.stateful:
            if len(req) < 2:
                raise TypeError("a stateful f needs at least (x, state) arguments")
            req = req[:-1]
        self.inputs = tuple(req if inputs is None else inputs)
        self.outputs = tuple(outputs)
        self.params = _parameters(f, defaults, params, self.inputs)
        self._compiled = None  # (fn, loop, mode)
        kw.setdefault("name", None if f.__name__ == "<lambda>" else f.__name__)
        super().__init__(**kw)

    def reset(self):
        """Restore the initial state."""
        self.state = _copy_state(self.state0)

    def info(self):
        """Return the function's name unless it is the block's, and statefulness."""
        name = self.f.__name__
        txt = "" if name == self.name else name
        return f"{txt} (stateful)".strip() if self.stateful else txt

    @property
    def mode(self) -> str | None:
        """``"numba"`` or ``"python"`` once built, else None."""
        return self._compiled[2] if self._compiled else None

    def _get(self):
        if self._compiled is None:
            use = HAVE_NUMBA and self.jit is not False
            fn, loop = _build(
                self.f,
                len(self.inputs),
                len(self.outputs),
                self.stateful,
                list(self.params),
                use,
            )
            self._compiled = (fn, loop, "numba" if use else "python")
        return self._compiled

    def _fallback(self, err):
        if self.jit is True:
            raise err
        warnings.warn(
            f"{self.name}: numba could not compile {self.f.__name__!r}; runs in "
            f"Python (slow): {str(err).splitlines()[0]}",
            RuntimeWarning,
            stacklevel=4,
        )
        fn, loop = _build(
            self.f,
            len(self.inputs),
            len(self.outputs),
            self.stateful,
            list(self.params),
            False,
        )
        self._compiled = (fn, loop, "python")

    def _probe_dtypes(self, fn, xs):
        # call f once on a copy of the state: f may change array state
        st = _copy_state(self.state)
        r = fn(*[x[0] for x in xs], *([st] if self.stateful else []), **self.params)
        y = r[0] if self.stateful else r
        out = []
        for v in y if len(self.outputs) > 1 else (y,):
            dt = np.asarray(v).dtype
            out.append(np.dtype(float) if dt.kind in "iub" else dt)  # no truncation
        return out

    def step(self, **u):
        """Run ``f`` over the chunk; returns one array per output."""
        xs = [np.ascontiguousarray(u[k]) for k in self.inputs]
        L = len(xs[0])
        for attempt in range(2):
            fn, loop, mode = self._get()
            try:
                dts = self.dtype or (
                    self._probe_dtypes(fn, xs) if L else [float] * len(self.outputs)
                )
                outs = [np.empty(L, dt) for dt in dts]
                s = self.state if self.stateful else 0.0
                s_new = loop(*xs, *outs, s, *self.params.values())
                break
            except Exception as err:  # numba typing errors show at the first call
                if mode != "numba" or attempt or not _is_numba_error(err):
                    raise
                self._fallback(err)
        if self.stateful:
            self.state = s_new
        return outs[0] if len(outs) == 1 else tuple(outs)


def _is_numba_error(err):
    return HAVE_NUMBA and isinstance(err, numba.core.errors.NumbaError)
