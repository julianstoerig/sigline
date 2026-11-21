"""The diagram: runs connected blocks chunk by chunk.

Per chunk:

1. take the outputs of blocks without direct feedthrough (sources, delays of
   at least one chunk),
2. ``step`` the other blocks in topological order,
3. ``update`` the blocks from 1.

A feedback loop must pass through a block from 1, else
:class:`AlgebraicLoopError`.
"""

from __future__ import annotations

import warnings
from collections import defaultdict, deque
from fractions import Fraction
from math import lcm

import numpy as np

from ._checks import check_inputs, check_output, describe
from .core import Block, Port
from .scope import Scope

__all__ = ["AlgebraicLoopError", "Diagram"]


class AlgebraicLoopError(RuntimeError):
    """Feedback loop in which every block needs its input of the same chunk."""


def _discover(start):
    seen, todo = set(), list(start)
    while todo:
        b = todo.pop()
        if b in seen:
            continue
        seen.add(b)
        todo += [p.block for p in b._sources.values()]
        todo += [p.block for p in b._sinks]
    return sorted(seen, key=lambda b: b._id)


class Diagram:
    """All blocks connected to the given ones, run together.

    Parameters
    ----------
    *blocks : Block
        Any blocks of the graph; one is enough.
    fs : float
        Sample rate / Hz. Sources with their own ``fs`` keep it; rates
        propagate along the lines (:meth:`rates`).
    check : {"first", True, False}, default "first"
        Check signals on the first chunk after build or :meth:`reset`, on
        every chunk, or never. Names the first block that outputs NaN/Inf or
        drops an imaginary part, and applies opt-in input checks
        (``same_shape_inputs``, ``real_inputs``). Exceptions inside a block
        always get a note naming it.

    Attributes
    ----------
    blocks : list of Block
        In creation order.
    n : int
        Samples simulated so far, at ``fs``.
    stopped : str or None
        Why the last run ended early (:meth:`Block.request_stop`).
    granularity : int
        Run and chunk lengths are multiples of this, so every block gets
        whole samples.

    Raises
    ------
    ValueError
        Unconnected input, mismatched input rates, or ``fs`` contradicting
        the input rate.
    TypeError
        ``rate`` used for something other than a rate ratio.

    Examples
    --------
    >>> import numpy as np
    >>> import sigline as sl
    >>> class Ramp(sl.Block):
    ...     inputs = ()
    ...
    ...     def reset(self):
    ...         self.n = 0
    ...
    ...     def step_source(self, L):
    ...         self.n += L
    ...         return np.arange(self.n - L, self.n, dtype=float)
    >>> class Record(sl.Block):
    ...     outputs = ()
    ...
    ...     def reset(self):
    ...         self.data = np.empty(0)
    ...
    ...     def step(self, x):
    ...         self.data = np.concatenate([self.data, x])
    ...         return {}
    >>> rec = Record()
    >>> _ = Ramp() >> rec
    >>> d = sl.Diagram(rec, fs=8.0)
    >>> d.run(0.5, chunk=3).n           # seconds; or d.run(n=4)
    4
    >>> d.run(n=2).n                     # continues where it stopped
    6
    >>> rec.data
    array([0., 1., 2., 3., 4., 5.])
    """

    def __init__(self, *blocks: Block, fs: float, check: bool | str = "first"):
        if check not in ("first", True, False):
            raise ValueError(f"check must be 'first', True or False, not {check!r}")
        self.fs = float(fs)
        self.check = check
        self._checked = False
        self.blocks = _discover(blocks)
        for b in self.blocks:
            missing = [p for p in b.inputs if p not in b._sources]
            if missing:
                raise ValueError(f"unconnected input(s) {missing} on {b.name}")
        self.n = 0
        self.stopped: str | None = None
        self._schedules: dict[int, tuple[list[Block], list[Block]]] = {}
        self._probes: dict[tuple[Block, str], list[Scope]] = {}
        self._assign_rates()

    # sample rates
    def _ratio(self, fs):
        return (Fraction(fs) / Fraction(self.fs)).limit_denominator(10**6)

    def _assign_rates(self):
        # rates are fractions of the diagram rate: rin[b], rout[b]
        for b in self.blocks:
            if not isinstance(b.rate, Fraction | int):
                raise TypeError(
                    f"{b.name}: the attribute 'rate' is reserved for the "
                    f"sample-rate ratio (a Fraction), but is {b.rate!r}; "
                    "rename your attribute"
                )
        rin, rout = {}, {}
        for b in self.blocks:
            if not b.inputs:
                r = self._ratio(b.fs) if b.fs else Fraction(1)
                rin[b] = rout[b] = r
        changed = True
        while changed:
            changed = False
            for b in self.blocks:
                if not b.inputs or b in rin:
                    continue
                known = [rout[p.block] for p in b._sources.values() if p.block in rout]
                if known:
                    rin[b], rout[b] = known[0], known[0] * b.rate
                    changed = True
        for b in self.blocks:  # loops without any source run at the diagram rate
            if b not in rin:
                rin[b], rout[b] = Fraction(1), Fraction(b.rate)
        f = Fraction(self.fs)
        for b in self.blocks:
            srcs = {k: rout[p.block] for k, p in b._sources.items()}
            if len(set(srcs.values())) > 1:
                desc = ", ".join(f"{k}: {float(r * f):g} Hz" for k, r in srcs.items())
                raise ValueError(f"sample-rate mismatch at {b.name} ({desc})")
            fs_b = float(rin[b] * f)
            if b.inputs and b.fs is not None and abs(b.fs - fs_b) > 1e-9 * fs_b:
                raise ValueError(
                    f"{b.name}.fs={b.fs:g} but its inputs run at {fs_b:g} Hz"
                )
            b.fs = fs_b
        self._rin, self._rout = rin, rout
        dens = [r.denominator for r in [*rin.values(), *rout.values()]]
        self.granularity = lcm(*dens) if dens else 1

    def rates(self) -> dict[str, tuple[float, float]]:
        """Return each block's input and output rate.

        Sources: own ``fs`` or the diagram's. Others: input rate times
        ``rate``.

        Returns
        -------
        dict
            ``{name: (fs_in / Hz, fs_out / Hz)}``.
        """
        f = Fraction(self.fs)
        return {
            b.name: (float(self._rin[b] * f), float(self._rout[b] * f))
            for b in self.blocks
        }

    def _len(self, b, N):
        # (input length, output length) of b for a chunk of N diagram samples
        return int(N * self._rin[b]), int(N * self._rout[b])

    # scheduling
    def compile(self, L: int) -> tuple[list[Block], list[Block]]:
        """Return the schedule for chunks of ``L`` samples (cached).

        Parameters
        ----------
        L : int
            Chunk length at the diagram rate.

        Returns
        -------
        state : list of Block
            Blocks without direct feedthrough.
        order : list of Block
            The others, in execution order.

        Raises
        ------
        AlgebraicLoopError
            A loop without a block from ``state``.
        """
        if L in self._schedules:
            return self._schedules[L]
        ft = [b for b in self.blocks if b.feedthrough(self._len(b, L)[0])]
        nft = [b for b in self.blocks if not b.feedthrough(self._len(b, L)[0])]
        ftset = set(ft)
        indeg = {b: 0 for b in ft}
        children = defaultdict(list)
        for b in ft:
            for src in {p.block for p in b._sources.values()}:
                if src in ftset:
                    indeg[b] += 1
                    children[src].append(b)
        q = deque(b for b in ft if indeg[b] == 0)
        order = []
        while q:
            b = q.popleft()
            order.append(b)
            for c in children[b]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(ft):
            loop = [b.name for b in ft if indeg[b] > 0]
            raise AlgebraicLoopError(
                f"algebraic loop at chunk={L} through {loop}: insert a block "
                "without direct feedthrough, e.g. a delay of at least one chunk"
            )
        self._schedules[L] = (nft, order)
        return nft, order

    # execution
    def run(self, T: float | None = None, *, n: int | None = None, chunk: int = 1024):
        """Simulate ``T`` seconds or ``n`` samples, continuing the last run.

        Ends early after a chunk in which a block called
        :meth:`Block.request_stop`.

        Parameters
        ----------
        T : float, optional
            Duration / s.
        n : int, optional
            Samples at the diagram rate; a multiple of ``granularity``.
        chunk : int, default 1024
            Samples per chunk, rounded down to a multiple of ``granularity``.
            Changes speed, never results.

        Returns
        -------
        Diagram
            ``self``.

        Raises
        ------
        ValueError
            Neither ``T`` nor ``n``, or ``n`` not a multiple of ``granularity``.
        """
        g = self.granularity
        if n is None:
            if T is None:
                raise ValueError("give T (seconds) or n (samples)")
            n = round(T * self.fs / g) * g
        if n % g:
            raise ValueError(
                f"n={n} must be a multiple of {g} (the slowest rate divides fs by {g})"
            )
        chunk = max(g, int(chunk) // g * g)
        done = 0
        self.stopped = None
        while done < n:
            L = min(chunk, n - done)
            self._step(L)
            done += L
            req = [b for b in self.blocks if getattr(b, "_stop_request", None)]
            if req:
                self.stopped = f"{req[0].name}: {req[0]._stop_request}"
                for b in req:
                    b._stop_request = None
                break
        self.n += done
        return self

    def _step(self, N):
        nft, order = self.compile(N)
        vals: dict[tuple[Block, str], np.ndarray] = {}
        checking = self.check is True or (self.check == "first" and not self._checked)

        def gather(b):
            return {k: vals[(p.block, p.name)] for k, p in b._sources.items()}

        def store(b, res, L):
            for k, v in b._normalise(res, L).items():
                if checking:
                    check_output(b, k, v)
                vals[(b, k)] = v

        # positional-only, because ports may be named "b" or "fn"
        def call(b, fn, /, *args, **inputs):
            if checking:
                check_inputs(b, inputs)
            try:
                if checking:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", np.exceptions.ComplexWarning)
                        return fn(*args, **inputs)
                return fn(*args, **inputs)
            except np.exceptions.ComplexWarning:
                raise TypeError(
                    f"{b.name} ({type(b).__name__}) dropped the imaginary part of "
                    f"a complex signal; inputs: {describe(inputs)}"
                ) from None
            except Exception as err:
                if not getattr(err, "_sigline_noted", False):
                    err.add_note(
                        f"  in block {b.name} ({type(b).__name__}), "
                        f"inputs: {describe(inputs) or 'none'}"
                    )
                    err._sigline_noted = True
                raise

        for b in nft:
            Lin, Lout = self._len(b, N)
            store(b, call(b, b.output, Lout), Lout)
        for b in order:
            Lin, Lout = self._len(b, N)
            if b.inputs:
                res = call(b, b.step, **gather(b))
            else:
                res = call(b, b.step_source, Lout)
            store(b, res, Lout)
        for b in nft:
            Lin, Lout = self._len(b, N)
            call(b, b.update, Lin if b.inputs else Lout, **gather(b))
        for key, scopes in self._probes.items():
            for sc in scopes:
                sc.step(vals[key])
        self._checked = True

    def reset(self):
        """Reset every block and probe, ``n`` and the first-chunk check.

        Returns
        -------
        Diagram
            ``self``.
        """
        for b in self.blocks:
            b.reset()
        for scopes in self._probes.values():
            for sc in scopes:
                sc.reset()
        self._checked = False
        self.n = 0
        return self

    # probes
    def _resolve(self, target) -> tuple[Block, str]:
        # block, output port, "name" or "name:port" -> (block, port)
        if isinstance(target, Port):
            if target.kind != "out":
                raise ValueError(
                    f"{target} is an input; probe the output that drives it"
                )
            return target.block, target.name
        if isinstance(target, Block):
            if len(target.outputs) != 1:
                raise ValueError(
                    f"{target.name} has outputs {target.outputs}; use 'name:port'"
                )
            return target, target.outputs[0]
        name, _, port = str(target).partition(":")
        blk = self[name]
        return self._resolve(blk.out(port) if port else blk)

    def probe(self, target, name=None, keep=None, every=1) -> Scope:
        """Record a signal without drawing a line to a Scope.

        Parameters
        ----------
        target : Block or Port or str
            A block with one output, an output port, ``"name"`` or
            ``"name:port"``.
        name : str, optional
            Default ``"probe:<target>"``.
        keep, every : optional
            As for :class:`Scope`.

        Returns
        -------
        Scope
            The recording; not part of :attr:`blocks`.

        Raises
        ------
        ValueError
            Input port, ambiguous output, or a block of another diagram.
        """
        blk, port = self._resolve(target)
        if blk not in self.blocks:
            raise ValueError(f"{blk.name} is not part of this diagram")
        label = target if isinstance(target, str) else f"{blk.name}:{port}"
        sc = Scope(name=name or f"probe:{label}", keep=keep, every=every)
        sc.fs = float(self._rout[blk] * Fraction(self.fs))
        self._probes.setdefault((blk, port), []).append(sc)
        return sc

    def remove_probe(self, sc: Scope):
        """Stop recording with probe ``sc``."""
        for key, lst in list(self._probes.items()):
            if sc in lst:
                lst.remove(sc)
                if not lst:
                    del self._probes[key]

    # inspection
    @property
    def t(self) -> np.ndarray:
        """Time / s of the samples simulated so far."""
        return np.arange(self.n) / self.fs

    def summary(self, chunk: int = 1024) -> str:
        """Return the schedule for ``chunk`` and all lines as text.

        Parameters
        ----------
        chunk : int, default 1024
            Chunk length to schedule.

        Returns
        -------
        str
            ``state``/``step`` rows in execution order, then the lines.

        Raises
        ------
        AlgebraicLoopError
            Not schedulable at this chunk length.
        """
        nft, order = self.compile(chunk)
        rows = [f"Diagram fs={self.fs:g} Hz, {len(self.blocks)} blocks, chunk={chunk}"]
        rows += [f"  state  {b.name}" for b in nft]
        rows += [f"  step   {b.name}" for b in order]
        rows += ["lines:"]
        for b in self.blocks:
            for k, p in b._sources.items():
                rows.append(f"  {p} -> {b.name}[{k!r}]")
        return "\n".join(rows)

    def __getitem__(self, name: str) -> Block:
        """Return the block called ``name``.

        Raises
        ------
        KeyError
            No such block.
        """
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    def scopes(self) -> list[Scope]:
        """Return the Scope blocks, in creation order."""
        return [b for b in self.blocks if isinstance(b, Scope)]

    def plot(self, *, axes=None):
        """Plot every Scope over time, one axis each.

        Parameters
        ----------
        axes : sequence of matplotlib.axes.Axes, optional
            One per Scope; default: new stacked subplots.

        Returns
        -------
        sequence of matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        sc = self.scopes()
        if axes is None:
            _, axes = plt.subplots(
                len(sc), 1, sharex=True, squeeze=False, figsize=(8, 2.2 * len(sc))
            )
            axes = axes[:, 0]
        for ax, s in zip(axes, sc, strict=True):
            s.plot(ax=ax)
            ax.set_xlabel("")
        axes[-1].set_xlabel("t / s")
        return axes
