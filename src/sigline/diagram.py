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

from collections import defaultdict, deque

import numpy as np

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
        Sample rate in Hz. Each block's ``fs`` attribute is set to it.

    Attributes
    ----------
    blocks : list of Block
        All blocks of the diagram, in creation order.
    fs : float
        Sample rate in Hz.
    n : int
        Number of samples simulated so far.

    Raises
    ------
    ValueError
        If an input of a block is not connected.

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

    def __init__(self, *blocks: Block, fs: float):
        self.fs = float(fs)
        self.blocks = _discover(blocks)
        for b in self.blocks:
            missing = [p for p in b.inputs if p not in b._sources]
            if missing:
                raise ValueError(f"unconnected input(s) {missing} on {b.name}")
        for b in self.blocks:
            if b.fs is not None and abs(b.fs - self.fs) > 1e-9 * self.fs:
                raise ValueError(
                    f"{b.name}.fs={b.fs:g} Hz differs from the diagram rate "
                    f"{self.fs:g} Hz"
                )
            b.fs = self.fs
        self.n = 0
        self._schedules: dict[int, tuple[list[Block], list[Block]]] = {}

    # scheduling
    def compile(self, L: int) -> tuple[list[Block], list[Block]]:
        """Return the schedule for chunks of ``L`` samples.

        Blocks with direct feedthrough at this chunk length are sorted
        topologically over the wires between them; the result is cached per
        chunk length.

        Parameters
        ----------
        L : int
            Chunk length in samples.

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
        ft = [b for b in self.blocks if b.feedthrough(L)]
        nft = [b for b in self.blocks if not b.feedthrough(L)]
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
            Duration in seconds, rounded to whole samples.
        n : int, optional
            Number of samples (instead of ``T``).
        chunk : int, default 1024
            Samples per chunk. Results do not depend on it; larger chunks run
            faster, smaller ones react sooner to parameter changes.

        Returns
        -------
        Diagram
            ``self``, so calls can be chained.

        Raises
        ------
        ValueError
            If neither ``T`` nor ``n`` is given.
        """
        if n is None:
            if T is None:
                raise ValueError("give T (seconds) or n (samples)")
            n = round(T * self.fs)
        chunk = max(1, int(chunk))
        done = 0
        while done < n:
            L = min(chunk, n - done)
            self._step(L)
            done += L
        self.n += done
        return self

    def _step(self, L):
        nft, order = self.compile(L)
        vals: dict[tuple[Block, str], np.ndarray] = {}

        def gather(b):
            return {k: vals[(p.block, p.name)] for k, p in b._sources.items()}

        def store(b, res):
            for k, v in b._normalise(res, L).items():
                vals[(b, k)] = v

        for b in nft:
            store(b, b.output(L))
        for b in order:
            store(b, b.step(**gather(b)) if b.inputs else b.step_source(L))
        for b in nft:
            b.update(L, **gather(b))

    def reset(self):
        """Restore the initial state of every block and set ``n`` to 0.

        Returns
        -------
        Diagram
            ``self``, so ``d.reset().run(1.0)`` works.
        """
        for b in self.blocks:
            b.reset()
        self.n = 0
        return self

    # inspection
    @property
    def t(self) -> np.ndarray:
        """Time axis of the samples simulated so far, in seconds."""
        return np.arange(self.n) / self.fs

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
