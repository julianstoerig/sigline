"""Graphs of diagrams: layout, text, matplotlib and Graphviz DOT.

One layout serves all renderings: feedback lines are found by a depth-first
search from the sources, blocks go into columns by the longest path along
the other lines, and a few barycenter sweeps order each column to reduce
crossings.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ._format import si
from .core import Block, StateBlock

__all__ = ["Edge", "Node", "layout"]


@dataclass(eq=False)
class Node:
    """A block as drawn."""

    block: Block
    name: str
    type: str
    inputs: tuple
    outputs: tuple
    kind: str
    info: str
    fs: float | None
    fs_out: float | None

    def rate_text(self) -> str:
        """Return ``"fs"`` or ``"fs_in → fs_out"``."""
        if not self.fs:
            return ""
        if self.fs_out and abs(self.fs_out - self.fs) > 1e-9 * self.fs:
            return f"{si(self.fs, 'Hz')} → {si(self.fs_out, 'Hz')}"
        return si(self.fs, "Hz")


@dataclass(eq=False)
class Edge:
    """A line from an output port to an input port."""

    src: Node
    sport: str
    dst: Node
    dport: str
    feedback: bool = False


def _kind(b) -> str:
    if not b.inputs:
        return "source"
    if not b.outputs:
        return "sink"
    if isinstance(b, StateBlock):
        return "state"
    return "block"


def view(d) -> tuple[list[Node], list[Edge]]:
    """Return the nodes and lines of diagram ``d``, in creation order."""
    nodes = {}
    for b in d.blocks:
        fs_out = b.fs * float(b.resampling_ratio) if b.fs and b.inputs else b.fs
        nodes[b] = Node(
            b, b.name, type(b).__name__, tuple(b.inputs), tuple(b.outputs),
            _kind(b), b.info(), b.fs, fs_out,
        )  # fmt: skip
    edges = [
        Edge(nodes[p.block], p.name, nodes[b], k)
        for b in d.blocks
        for k, p in b._sources.items()
    ]
    edges.sort(
        key=lambda e: (
            e.src.block._id,
            e.src.outputs.index(e.sport),
            e.dst.block._id,
            e.dst.inputs.index(e.dport),
        )
    )
    return list(nodes.values()), edges


def layout(d) -> tuple[list[list[Node]], list[Edge]]:
    """Place the blocks of ``d`` in columns.

    Parameters
    ----------
    d : Diagram

    Returns
    -------
    columns : list of list of Node
        Left to right, each ordered top to bottom.
    edges : list of Edge
        All lines; ``feedback`` marks those that run backwards.
    """
    return _layout(*view(d))


def _layout(nodes, edges):
    succ, indeg = defaultdict(list), defaultdict(int)
    for e in edges:
        succ[e.src].append(e)
        indeg[e.dst] += 1

    # feedback lines: back edges of an iterative depth-first search from sources
    white, grey, black = 0, 1, 2
    color = dict.fromkeys(nodes, white)
    for root in [n for n in nodes if not indeg[n]] + nodes:
        if color[root] != white:
            continue
        color[root] = grey
        stack = [(root, iter(succ[root]))]
        while stack:
            node, it = stack[-1]
            e = next(it, None)
            if e is None:
                color[node] = black
                stack.pop()
            elif color[e.dst] == grey:
                e.feedback = True
            elif color[e.dst] == white:
                color[e.dst] = grey
                stack.append((e.dst, iter(succ[e.dst])))

    # columns by longest path over forward lines
    fwd = [e for e in edges if not e.feedback]
    col = dict.fromkeys(nodes, 0)
    for _ in range(len(nodes)):
        changed = False
        for e in fwd:
            if col[e.dst] < col[e.src] + 1:
                col[e.dst] = col[e.src] + 1
                changed = True
        if not changed:
            break
    # sources sit right before their first consumer: no long lines
    for n in nodes:
        outs = [e for e in fwd if e.src is n]
        if not indeg[n] and outs:
            col[n] = min(col[e.dst] for e in outs) - 1
    ncol = max(col.values()) + 1 if col else 0
    cols = [[n for n in nodes if col[n] == c] for c in range(ncol)]

    # order within columns by barycenters of neighbours, sweeping both ways
    for sweep in range(6):
        pos = {n: i for c in cols for i, n in enumerate(c)}
        forward = sweep % 2 == 0
        for c in range(1, ncol) if forward else range(ncol - 2, -1, -1):

            def bary(n, c=c, pos=pos, forward=forward):
                if forward:
                    nb = [pos[e.src] for e in fwd if e.dst is n and col[e.src] < c]
                else:
                    nb = [pos[e.dst] for e in fwd if e.src is n and col[e.dst] > c]
                return sum(nb) / len(nb) if nb else pos[n]

            cols[c].sort(key=lambda n: (bary(n), n.block._id))
    return cols, edges
