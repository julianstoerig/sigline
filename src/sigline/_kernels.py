"""numba kernels: one block, one sample, for compiled loop clusters.

A kernel is ``step(S, P, *inputs) -> (outputs, S)`` and, for blocks without
feedthrough at one sample, ``out(S, P) -> outputs``. ``S`` is the block's
state, ``P`` its parameters. Around every compiled chunk the cluster reads
both from the block and writes the state back, so ``reset()``, inspection
and parameter changes between runs keep working. Signals are scalars.

Adding a block type to compiled loops means registering a kernel here with
:func:`kernel_for`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._numba import callable_in_numba

try:
    import numba
except ImportError:  # pragma: no cover - loops import this only with numba
    numba = None

from .blocks import Delay, Func, Gain, Mixer, Product, Sum
from .filters import FIR, IIR
from .nonlinear import Nonlinear, _copy_state

TWO_PI = 2 * np.pi
JIT = {"error_model": "numpy"}  # division by zero gives inf, as in numpy


class NotCompilable(Exception):
    """A block or signal that compiled loops do not support."""


@dataclass
class Kernel:
    """One block's kernel and how to move its state and parameters."""

    step: Callable
    out: Callable | None
    get_state: Callable
    set_state: Callable
    params: Callable


_KERNELS: dict[type, Callable] = {}
_cache: dict = {}


def kernel_for(cls):
    """Register the decorated ``make(block) -> Kernel`` for blocks of ``cls``."""

    def register(make):
        _KERNELS[cls] = make
        return make

    return register


def kernel(b) -> Kernel:
    """Return the kernel for block ``b``.

    Raises
    ------
    NotCompilable
        No kernel for its type, or parameters a kernel does not support.
    """
    make = _KERNELS.get(type(b))
    if make is None:
        raise NotCompilable(f"{b.name}: no compiled kernel for {type(b).__name__}")
    return make(b)


def _jit(key, src, **glob):
    # compile step (and out) from source once per key
    if key not in _cache:
        ns = {"np": np, "math": math, "TWO_PI": TWO_PI, **glob}
        exec(src, ns)
        jit = numba.njit(inline="always", **JIT)
        _cache[key] = (jit(ns["step"]), jit(ns["out"]) if "out" in ns else None)
    return _cache[key]


def _stateless(step, params):
    return Kernel(step, None, lambda: 0.0, lambda S: None, params)


def _scalar(b, arr):
    if arr is not None and np.ndim(arr) > 1:
        raise NotCompilable(f"{b.name}: vector signals do not compile")


@kernel_for(Gain)
def _gain(b):
    if not np.isscalar(b.g):
        raise NotCompilable(f"{b.name}: array gain")
    step, _ = _jit("gain", "def step(S, P, x):\n    return (P[0] * x,), S\n")
    return _stateless(step, lambda: (b.g,))


@kernel_for(Sum)
def _sum(b):
    terms = [("-" if s == "-" else "+") + f" x{i}" for i, s in enumerate(b.signs)]
    expr = " ".join(terms).removeprefix("+ ")
    args = ", ".join(f"x{i}" for i in range(len(b.signs)))
    src = f"def step(S, P, {args}):\n    return ({expr},), S\n"
    step, _ = _jit(("sum", b.signs), src)
    return _stateless(step, lambda: (0.0,))


@kernel_for(Product)
def _product(b):
    expr = "1"
    for i, op in enumerate(b.ops):
        expr = f"({expr} {op} x{i})"
    args = ", ".join(f"x{i}" for i in range(len(b.ops)))
    src = f"def step(S, P, {args}):\n    return ({expr},), S\n"
    step, _ = _jit(("product", b.ops), src)
    return _stateless(step, lambda: (0.0,))


@kernel_for(Func)
def _func(b):
    try:
        f = callable_in_numba(b.fn, **JIT)
    except Exception as err:
        raise NotCompilable(f"{b.name}: {err}") from None
    args = ", ".join(f"u{i}" for i in range(len(b.inputs)))
    ret = f"(f({args}),)" if len(b.outputs) == 1 else f"f({args})"
    ns = {"f": f}
    exec(f"def step(S, P, {args}):\n    return {ret}, S\n", ns)
    return _stateless(numba.njit(inline="always", **JIT)(ns["step"]), lambda: (0.0,))


@kernel_for(Delay)
def _delay(b):
    if b.n == 0:
        step, _ = _jit("delay0", "def step(S, P, x):\n    return (x,), S\n")
        return _stateless(step, lambda: (0.0,))
    _scalar(b, b.buf)
    # ring buffer: i is the oldest sample, the next output
    step, out = _jit(
        "delay",
        "def out(S, P):\n"
        "    return (S[0][S[1]],)\n"
        "def step(S, P, x):\n"
        "    buf, i = S\n"
        "    y = buf[i]\n"
        "    buf[i] = x\n"
        "    i += 1\n"
        "    if i == buf.size:\n"
        "        i = 0\n"
        "    return (y,), (buf, i)\n",
    )

    def put(S):
        buf, i = S
        b.buf = np.concatenate([buf[i:], buf[:i]])

    return Kernel(step, out, lambda: (b.buf.copy(), 0), put, lambda: (0.0,))


@kernel_for(FIR)
def _fir(b):
    _scalar(b, b.hist)
    if b.hist is None:
        raise NotCompilable(f"{b.name}: state not initialised")
    if len(b.taps) == 1:
        step, _ = _jit("fir0", "def step(S, P, x):\n    return (P[0][0] * x,), S\n")
        return _stateless(step, lambda: (np.asarray(b.taps),))
    # ring of the last K inputs: i is the oldest, overwritten next
    step, _ = _jit(
        "fir",
        "def step(S, P, x):\n"
        "    ring, i = S\n"
        "    h = P[0]\n"
        "    K = ring.size\n"
        "    y = h[0] * x\n"
        "    j = i\n"
        "    for k in range(1, K + 1):\n"
        "        j -= 1\n"
        "        if j < 0:\n"
        "            j += K\n"
        "        y += h[k] * ring[j]\n"
        "    ring[i] = x\n"
        "    i += 1\n"
        "    if i == K:\n"
        "        i = 0\n"
        "    return (y,), (ring, i)\n",
    )

    def get():
        b._fit_history(b.hist[:0])  # taps may have changed length
        return (b.hist.copy(), 0)

    def put(S):
        ring, i = S
        b.hist = np.concatenate([ring[i:], ring[:i]])

    return Kernel(step, None, get, put, lambda: (np.ascontiguousarray(b.taps),))


@kernel_for(IIR)
def _iir(b):
    if b.zi is None:
        raise NotCompilable(f"{b.name}: state not initialised")
    if b.sos is not None:
        _scalar(b, b.zi[:, 0])
        # transposed direct form II per section, as scipy.signal.sosfilt
        step, _ = _jit(
            "sos",
            "def step(S, P, x):\n"
            "    z = S[0]\n"
            "    sos = P[0]\n"
            "    y = x\n"
            "    for s in range(sos.shape[0]):\n"
            "        xs = y\n"
            "        y = sos[s, 0] * xs + z[s, 0]\n"
            "        z[s, 0] = sos[s, 1] * xs - sos[s, 4] * y + z[s, 1]\n"
            "        z[s, 1] = sos[s, 2] * xs - sos[s, 5] * y\n"
            "    return (y,), (z,)\n",
        )

        def params():
            sos = np.asarray(b.sos, float)
            return (sos / sos[:, 3:4],)

        return Kernel(
            step,
            None,
            lambda: (b.zi.copy(),),
            lambda S: setattr(b, "zi", S[0]),
            params,
        )
    _scalar(b, b.zi)
    # transposed direct form II, as scipy.signal.lfilter
    step, _ = _jit(
        "lfilter",
        "def step(S, P, x):\n"
        "    z = S[0]\n"
        "    bb, aa = P\n"
        "    n = bb.size - 1\n"
        "    y = bb[0] * x\n"
        "    if n > 0:\n"
        "        y += z[0]\n"
        "    for k in range(n - 1):\n"
        "        z[k] = bb[k + 1] * x + z[k + 1] - aa[k + 1] * y\n"
        "    if n > 0:\n"
        "        z[n - 1] = bb[n] * x - aa[n] * y\n"
        "    return (y,), (z,)\n",
    )

    def params():
        n = max(len(b.a), len(b.b))
        bb, aa = np.zeros(n), np.zeros(n)
        bb[: len(b.b)], aa[: len(b.a)] = b.b, b.a
        return bb / aa[0], aa / aa[0]

    return Kernel(
        step, None, lambda: (b.zi.copy(),), lambda S: setattr(b, "zi", S[0]), params
    )


@kernel_for(Mixer)
def _mixer(b):
    lo = {
        "cos": "np.cos(ph)",
        "sin": "np.sin(ph)",
        "exp": "np.exp(1j * ph)",
        "exp-": "np.exp(-1j * ph)",
    }[b.lo]
    step, _ = _jit(
        ("mixer", b.lo),
        "def step(S, P, x):\n"
        "    ph = S\n"
        f"    y = x * {lo}\n"
        "    return (y,), (ph + P[0]) % TWO_PI\n",
    )
    return Kernel(
        step,
        None,
        lambda: float(b.phi),
        lambda S: setattr(b, "phi", S),
        lambda: (TWO_PI * b.f / b.fs,),
    )


@kernel_for(Nonlinear)
def _nonlinear(b):
    if b.mode != "numba":
        raise NotCompilable(f"{b.name}: its function is not compiled with numba")
    f = b._get()[0]
    ins = "".join(f", u{i}" for i in range(len(b.inputs)))
    kws = "".join(f", {n}=P[{i}]" for i, n in enumerate(b.params))
    call = f"f({ins[2:]}{', S' if b.stateful else ''}{kws})"
    if b.stateful:
        ret = "(r[0],), r[1]" if len(b.outputs) == 1 else "r[0], r[1]"
    else:
        ret = "(r,), S" if len(b.outputs) == 1 else "r, S"
    ns = {"f": f}
    exec(f"def step(S, P{ins}):\n    r = {call}\n    return {ret}\n", ns)
    if b.stateful:
        get, put = (lambda: _copy_state(b.state)), (lambda S: setattr(b, "state", S))
    else:
        get, put = (lambda: 0.0), (lambda S: None)
    return Kernel(
        numba.njit(inline="always", **JIT)(ns["step"]),
        None,
        get,
        put,
        lambda: tuple(b.params.values()) or (0.0,),
    )
