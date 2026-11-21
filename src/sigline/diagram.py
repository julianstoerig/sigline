"""The diagram: collects connected blocks and runs them chunk by chunk.

Execution model
---------------
A diagram runs at one sample rate ``fs`` and is advanced in chunks of ``L``
samples. Every signal on a wire is a NumPy array whose first axis is time.
Per chunk the scheduler

1. asks every block *without direct feedthrough* for its outputs (they only
   depend on internal state: sources, delays by at least one chunk),
2. calls ``step(**inputs)`` on all feedthrough blocks in topological order,
3. calls ``update(**inputs)`` on the non-feedthrough blocks to advance them.

Feedback loops are legal as long as every loop passes through a block without
direct feedthrough at the chosen chunk length; otherwise :meth:`Diagram.compile`
raises :class:`AlgebraicLoopError` and names the blocks of the loop.
"""

from __future__ import annotations

import warnings
from collections import defaultdict, deque
from fractions import Fraction
from math import lcm

import numpy as np

from ._checks import check_inputs, check_output, describe
from .core import Block

__all__ = ["AlgebraicLoopError", "Diagram"]


class AlgebraicLoopError(RuntimeError):
    """A feedback loop in which every block needs its input of the same chunk."""


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
    """Collects every block connected to the given ones and runs them.

    Parameters
    ----------
    *blocks : Block
        Any blocks of the graph; every block reachable from them through
        wires (in either direction) belongs to the diagram, so one is enough.
    fs : float
        Sample rate in Hz. Sources run at their own ``fs`` if they have one,
        otherwise at this rate; rates propagate along the wires (see
        :meth:`rates`) and each block's ``fs`` attribute is set to its input
        rate.
    check : {"first", True, False}, default "first"
        When to check the signals: on the first chunk after building or
        :meth:`reset`, on every chunk, or never. The checks name the first
        block that outputs NaN/Inf or drops the imaginary part of a complex
        signal, and enforce the input checks blocks opt in to
        (``same_shape_inputs``, ``real_inputs``). Independently of ``check``,
        an exception raised inside a block gets a note naming the block.

    Attributes
    ----------
    blocks : list of Block
        All blocks of the diagram, in creation order.
    fs : float
        Sample rate in Hz; ``n``, ``t`` and ``run`` count samples at this rate.
    n : int
        Number of samples simulated so far.
    stopped : str or None
        Why the last ``run`` ended early (a block called
        :meth:`Block.request_stop`), or ``None`` if it ran to the end.
    granularity : int
        Run lengths and chunk sizes are multiples of this number of samples,
        so that every block gets a whole number of samples per chunk.

    Raises
    ------
    ValueError
        If an input of a block is not connected, the inputs of a block arrive
        at different rates, or a block's ``fs`` contradicts its input rate.
    TypeError
        If a block uses the reserved attribute ``rate`` for something else.

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
        """Return the input and output sample rate of every block.

        Sources run at their own ``fs`` (default: the diagram's). Every other
        block takes the rate of its inputs, which must agree, and multiplies
        it by its ``rate`` attribute (output rate / input rate).

        Returns
        -------
        dict
            ``{block name: (input fs, output fs)}`` in Hz.
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
        """Return the schedule for chunks of ``L`` samples.

        Blocks with direct feedthrough at this chunk length are sorted
        topologically over the wires between them; the result is cached per
        chunk length.

        Parameters
        ----------
        L : int
            Chunk length in samples at the diagram rate.

        Returns
        -------
        state : list of Block
            Blocks without direct feedthrough: their outputs are taken first,
            their states updated last.
        order : list of Block
            The feedthrough blocks in the order they run.

        Raises
        ------
        AlgebraicLoopError
            If a loop of wires has no block without direct feedthrough.
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
        """Simulate ``T`` seconds or ``n`` samples, continuing from the last run.

        Parameters
        ----------
        T : float, optional
            Duration in seconds, rounded to a multiple of ``granularity``
            samples.
        n : int, optional
            Number of samples at the diagram rate (instead of ``T``); must be a
            multiple of ``granularity``.
        chunk : int, default 1024
            Samples per chunk at the diagram rate, rounded down to a multiple
            of ``granularity``. Results do not depend on it; larger chunks run
            faster, smaller ones react sooner to parameter changes.

        The run ends early, after the current chunk, when a block calls
        :meth:`Block.request_stop`; ``stopped`` then gives the reason.

        Returns
        -------
        Diagram
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If neither ``T`` nor ``n`` is given, or ``n`` is not a multiple of
            ``granularity``.
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
        self._checked = True

    def reset(self):
        """Restore the initial state of every block and set ``n`` to 0.

        With ``check="first"`` the next chunk is checked again.

        Returns
        -------
        Diagram
            ``self``, so ``d.reset().run(1.0)`` works.
        """
        for b in self.blocks:
            b.reset()
        self._checked = False
        self.n = 0
        return self

    # inspection
    @property
    def t(self) -> np.ndarray:
        """Time axis of the samples simulated so far, in seconds."""
        return np.arange(self.n) / self.fs

    def summary(self, chunk: int = 1024) -> str:
        """Describe the schedule for ``chunk`` and list all wires.

        Parameters
        ----------
        chunk : int, default 1024
            Chunk length whose schedule is shown.

        Returns
        -------
        str
            One line per block (``state`` for blocks without feedthrough,
            ``step`` for the others, in execution order), then one line per
            wire.

        Raises
        ------
        AlgebraicLoopError
            If the diagram cannot be scheduled at this chunk length.
        """
        nft, order = self.compile(chunk)
        lines = [f"Diagram fs={self.fs:g} Hz, {len(self.blocks)} blocks, chunk={chunk}"]
        lines += [f"  state  {b.name}" for b in nft]
        lines += [f"  step   {b.name}" for b in order]
        lines += ["wires:"]
        for b in self.blocks:
            for k, p in b._sources.items():
                lines.append(f"  {p} -> {b.name}[{k!r}]")
        return "\n".join(lines)

    def __getitem__(self, name: str) -> Block:
        """Return the block called ``name``.

        Raises
        ------
        KeyError
            If no block has that name.
        """
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)
