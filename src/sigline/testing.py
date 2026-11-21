"""Test helpers for blocks.

>>> from sigline.testing import assert_chunk_invariant
"""

from __future__ import annotations

import numpy as np

from .diagram import Diagram
from .scope import Scope
from .sources import FromArray

__all__ = ["assert_chunk_invariant"]


def _build(make, inputs, fs):
    blk = make()
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            FromArray(v, fs=fs) >> blk.inp(k)
    elif inputs is not None and not isinstance(inputs, int | np.integer):
        FromArray(inputs, fs=fs) >> blk
    scopes = {o: Scope() for o in blk.outputs}
    if not scopes:
        raise ValueError("block has no outputs")
    for o, sc in scopes.items():
        blk.out(o) >> sc
    if not blk.inputs and blk.fs is None:
        blk.fs = fs
    return scopes, Diagram(*scopes.values(), fs=fs)


def assert_chunk_invariant(
    make, inputs, fs=1.0, chunks=(1, 7, 64), n_random=3, seed=0, atol=1e-10, rtol=1e-9
):
    """Assert that a block gives the same output for any chunking.

    Compares one run in a single chunk with: a run after ``reset()``, runs
    in each of ``chunks``, and ``n_random`` runs with random chunk sizes.

    Parameters
    ----------
    make : callable
        Returns a fresh block.
    inputs : array_like or dict or int
        Input signal, ``{port: signal}``, or a length for sources.
    fs : float, default 1.0
        Sample rate / Hz.
    chunks : tuple of int, default (1, 7, 64)
        Fixed chunk sizes, rounded to the granularity.
    n_random : int, default 3
        Runs with random chunk sizes.
    seed : int, default 0
        Seed for the random chunk sizes.
    atol, rtol : float
        Tolerances.

    Returns
    -------
    dict
        ``{output: array}`` of the single-chunk run.

    Raises
    ------
    AssertionError
        Names the run and the first differing sample.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Block
    >>> class Gain2(Block):
    ...     def step(self, x):
    ...         return 2 * x
    >>> ref = assert_chunk_invariant(Gain2, np.arange(10.0))
    >>> ref["y"][:3]
    array([0., 2., 4.])
    """
    if isinstance(inputs, dict):
        n = min(len(v) for v in inputs.values())
    elif isinstance(inputs, int | np.integer):
        n = int(inputs)
    else:
        n = len(inputs)
    scopes, d = _build(make, inputs, fs)
    g = d.granularity
    n = n // g * g
    if n == 0:
        raise ValueError("input shorter than one granule of the slowest rate")
    d.run(n=n, chunk=n)
    ref = {o: sc.data.copy() for o, sc in scopes.items()}

    def check(label, got):
        for o in ref:
            a, b = got[o], ref[o]
            if a.shape != b.shape:
                raise AssertionError(
                    f"{label}: output {o!r} shape {a.shape} != {b.shape}"
                )
            bad = ~np.isclose(a, b, atol=atol, rtol=rtol)
            if np.any(bad):
                i = int(np.flatnonzero(bad.reshape(len(bad), -1).any(axis=1))[0])
                raise AssertionError(
                    f"{label}: output {o!r} differs from the one-chunk run first at "
                    f"sample {i}: {a[i]} != {b[i]} "
                    f"(max error {np.max(np.abs(a - b)):.3g})"
                )

    def size(c):
        return max(g, c // g * g)

    d.reset()
    d.run(n=n, chunk=size(n // 3))
    check("after reset()", {o: sc.data for o, sc in scopes.items()})

    for c in chunks:
        sc2, d2 = _build(make, inputs, fs)
        d2.run(n=n, chunk=size(c))
        check(f"chunk={size(c)}", {o: s.data for o, s in sc2.items()})

    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        sc2, d2 = _build(make, inputs, fs)
        done, sizes = 0, []
        while done < n:
            c = min(n - done, size(int(rng.integers(1, max(2, n // 8))) * g))
            d2.run(n=c, chunk=c)
            sizes.append(c)
            done += c
        more = "..." if len(sizes) > 6 else ""
        check(f"random chunks {sizes[:6]}{more}", {o: s.data for o, s in sc2.items()})
    return ref
