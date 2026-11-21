"""The diagram: collects connected blocks and runs them chunk by chunk.

Execution model
---------------
A diagram runs at one sample rate ``fs`` and is advanced in chunks of ``L``
samples. Every signal on a wire is a NumPy array whose first axis is time.
Per chunk the scheduler calls every block once, in topological order, so each
block sees the outputs of its drivers for the same chunk.
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
        self._schedules: dict[int, list[Block]] = {}

    # scheduling
    def compile(self, L: int) -> list[Block]:
        """Return the execution order for chunks of ``L`` samples.

        Blocks are sorted topologically over the wires; the result is cached
        per chunk length.

        Parameters
        ----------
        L : int
            Chunk length in samples.

        Returns
        -------
        list of Block
            The blocks in the order they run.

        Raises
        ------
        AlgebraicLoopError
            If the wires form a loop.
        """
        if L in self._schedules:
            return self._schedules[L]
        indeg = {b: 0 for b in self.blocks}
        children = defaultdict(list)
        for b in self.blocks:
            for src in {p.block for p in b._sources.values()}:
                indeg[b] += 1
                children[src].append(b)
        q = deque(b for b in self.blocks if indeg[b] == 0)
        order = []
        while q:
            b = q.popleft()
            order.append(b)
            for c in children[b]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(self.blocks):
            loop = [b.name for b in self.blocks if indeg[b] > 0]
            raise AlgebraicLoopError(f"algebraic loop at chunk={L} through {loop}")
        self._schedules[L] = order
        return order

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
        vals: dict[tuple[Block, str], np.ndarray] = {}
        for b in self.compile(L):
            if b.inputs:
                ins = {k: vals[(p.block, p.name)] for k, p in b._sources.items()}
                res = b.step(**ins)
            else:
                res = b.step_source(L)
            for k, v in b._normalise(res, L).items():
                vals[(b, k)] = v

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
