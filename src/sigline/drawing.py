"""Graphs of diagrams: layout, text, matplotlib and Graphviz DOT.

One layout serves all renderings: feedback lines are found by a depth-first
search from the sources, blocks go into columns by the longest path along
the other lines, a line spanning several columns gets a lane in each column
it passes, and a few barycenter sweeps order each column to reduce
crossings. Subsystems show as one block, or opened (``expand=True``) as
their blocks, kept together in each column and framed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise, takewhile

from ._format import si
from .core import Block, StateBlock

__all__ = ["Edge", "Node", "draw", "layout", "to_dot", "to_mermaid", "to_text"]


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
    groups: tuple = ()  # opened subsystems holding the block, outermost first

    def label(self) -> str:
        """Return the name as drawn: in a frame, without the frame's path."""
        if not self.groups:
            return self.name
        frame = self.groups[-1]
        pre = "/".join(g.name for g in (*_groups(frame), frame)) + "/"
        name = self.block.name
        return name[len(pre) :] if name.startswith(pre) else name

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
    lanes: list = field(default_factory=list, repr=False)


@dataclass(eq=False)
class _Lane:
    # where a line passes a column it skips; ordered like a block
    edge: Edge
    col: int
    groups: tuple = ()  # frames it passes through there


def _tie(n, lanes_first):
    # stable order among equal barycenters; lanes before or after blocks
    if isinstance(n, Node):
        return (1, n.block._id)
    return (0 if lanes_first else 2, n.edge.dst.block._id)


def _crossings(cols, segs, col):
    # pairs of lines between adjacent columns that cross; lines from or to
    # the same port do not
    pos = {n: i for c in cols for i, n in enumerate(c)}
    by_col = defaultdict(list)
    for a, fa, b, fb in segs:
        by_col[col[a]].append((pos[a] + fa, pos[b] + fb))
    count = 0
    for lines in by_col.values():
        for i, (u1, v1) in enumerate(lines):
            for u2, v2 in lines[i + 1 :]:
                count += (u1 - u2) * (v1 - v2) < 0
    return count


def _blocks(cols):
    return [[n for n in c if isinstance(n, Node)] for c in cols]


def _kind(b) -> str:
    if not b.inputs:
        return "source"
    if not b.outputs:
        return "sink"
    if isinstance(b, StateBlock):
        return "state"
    return "block"


def _top(b, root=None):
    # the outermost subsystem below root holding b, or b itself
    while getattr(b, "_parent", None) is not root:
        b = b._parent
    return b


def _groups(b, root=None):
    # the subsystems below root holding b, outermost first
    chain = []
    while getattr(b, "_parent", None) is not root:
        b = b._parent
        chain.append(b)
    return tuple(reversed(chain))


def _lane_groups(e, c, first, last):
    # the frames a line passes at column c: those of its ends spanning c
    into = tuple(takewhile(lambda g: first[g] <= c, e.dst.groups))
    out = tuple(takewhile(lambda g: last[g] >= c, e.src.groups))
    return max(into, out, key=len)


def _together(items, depth=0):
    # the members of each opened subsystem next to each other, at their
    # mean place; nested subsystems likewise inside
    runs = {}
    for i, n in enumerate(items):
        g = n.groups
        runs.setdefault(g[depth] if len(g) > depth else n, []).append(i)
    out = []
    for k in sorted(runs, key=lambda k: sum(runs[k]) / len(runs[k])):
        members = [items[i] for i in runs[k]]
        out += members if isinstance(k, Node | _Lane) else _together(members, depth + 1)
    return out


def _shown(b, port, kind, root=None, expand=False):
    # (shown object, port) for a port of block b; None if hidden inside a
    # subsystem
    if expand:
        return b, port
    top = _top(b, root)
    if top is b:
        return b, port
    name = top._port_name(b, port, kind)
    return None if name is None else (top, name)


def _rel(name, root):
    # name inside the subsystem drawn alone, without its path
    pre = f"{root.name}/" if root is not None else ""
    return name[len(pre) :] if pre and name.startswith(pre) else name


def view(d, expand=False) -> tuple[list[Node], list[Edge]]:
    """Return the nodes and lines of diagram ``d``, in creation order.

    Parameters
    ----------
    d : Diagram
    expand : bool, default False
        Show the blocks inside subsystems, each with the subsystems holding
        it (``Node.groups``). Otherwise a subsystem is one node and the
        lines inside it are left out.

    Returns
    -------
    nodes : list of Node
    edges : list of Edge
    """
    root = getattr(d, "root", None)
    nodes = {}
    for b in d.blocks:
        top = b if expand else _top(b, root)
        if top is not b:
            if top not in nodes:
                ins = [ps[0] for ps in top._in.values()]
                outs = list(top._out.values())
                fs = ins[0].block.fs if ins else outs[0].block.fs
                ob = outs[0].block if outs else None
                fs_out = ob.fs * float(ob.resampling_ratio) if ob and ob.fs else fs
                nodes[top] = Node(
                    top, _rel(top.name, root), top.type, top.inputs, top.outputs,
                    "subsystem", top.info(), fs, fs_out,
                )  # fmt: skip
            continue
        fs_out = b.fs * float(b.resampling_ratio) if b.fs and b.inputs else b.fs
        nodes[b] = Node(
            b, _rel(b.name, root), type(b).__name__, tuple(b.inputs), tuple(b.outputs),
            _kind(b), b.info(), b.fs, fs_out, _groups(b, root) if expand else (),
        )  # fmt: skip
    edges, shown, seen = [], set(d.blocks), set()
    for b in d.blocks:
        for k, p in b._sources.items():
            if p.block not in shown:  # from outside a subsystem drawn alone
                continue
            src = _shown(p.block, p.name, "out", root, expand)
            dst = _shown(b, k, "in", root, expand)
            if src is None or dst is None or src[0] is dst[0]:
                continue
            key = (src[0], src[1], dst[0], dst[1])
            if key in seen:  # one line into a subsystem input feeding several
                continue
            seen.add(key)
            edges.append(Edge(nodes[src[0]], src[1], nodes[dst[0]], dst[1]))
    edges.sort(
        key=lambda e: (
            e.src.block._id,
            e.src.outputs.index(e.sport),
            e.dst.block._id,
            e.dst.inputs.index(e.dport),
        )
    )
    return list(nodes.values()), edges


def layout(d, expand=False) -> tuple[list[list[Node]], list[Edge]]:
    """Place the blocks of ``d`` in columns.

    Parameters
    ----------
    d : Diagram
    expand : bool, default False
        Place the blocks inside subsystems, as :func:`view`.

    Returns
    -------
    columns : list of list of Node
        Left to right, each ordered top to bottom.
    edges : list of Edge
        All lines; ``feedback`` marks those that run backwards.
    """
    cols, edges = _layout(*view(d, expand))
    return _blocks(cols), edges


def _frac(ports, port):
    # position of a port within its block, in [0, 1)
    return ports.index(port) / len(ports)


def _columns(nodes, edges, framed):
    # columns by longest path over forward lines; framed, an opened
    # subsystem also starts after the blocks feeding it and ends before the
    # blocks it feeds, so no other block shares its columns. None if lines
    # leave a subsystem and come back, which these rules cannot meet
    fwd = [e for e in edges if not e.feedback]
    fed = {e.dst for e in edges}
    members = defaultdict(list)
    for n in nodes:
        for g in n.groups if framed else ():
            members[g].append(n)
    rules = [
        (e, [g for g in e.dst.groups if g not in e.src.groups and g in members],
            [g for g in e.src.groups if g not in e.dst.groups and g in members])
        for e in fwd
    ]  # fmt: skip
    col = dict.fromkeys(nodes, 0)

    def lift(n, c):
        if col[n] < c:
            col[n] = c
            return True
        return False

    for _ in range(len(nodes) + 1):
        changed = False
        for e, into, out_of in rules:
            changed |= lift(e.dst, col[e.src] + 1)
            for g in into:
                for m in members[g]:
                    changed |= lift(m, col[e.src] + 1)
            for g in out_of:
                changed |= lift(e.dst, max(col[m] for m in members[g]) + 1)
        if not changed:
            break
    else:
        return None

    def entry(v, s):
        # the column where v's outermost subsystem without s begins
        g = next((g for g in v.groups if g in members and g not in s.groups), None)
        return col[v] if g is None else min(col[m] for m in members[g])

    # sources sit right before their first consumer: no long lines
    for n in nodes:
        outs = [e for e in fwd if e.src is n]
        if n not in fed and outs:
            col[n] = min(entry(e.dst, n) for e in outs) - 1
    # one landing in the columns of a subsystem it is not part of gets a
    # column of its own after it
    moved = True
    while moved:
        moved = False
        for n in nodes:
            if n in fed:
                continue
            for g, ms in members.items():
                first, last = min(col[m] for m in ms), max(col[m] for m in ms)
                if g not in n.groups and first <= col[n] <= last:
                    for m in nodes:
                        if col[m] > last:
                            col[m] += 1
                    col[n], moved = last + 1, True
                    break
    return col


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

    fwd = [e for e in edges if not e.feedback]
    col = _columns(nodes, edges, True) or _columns(nodes, edges, False)
    ncol = max(col.values()) + 1 if col else 0

    # lanes: a line skipping columns passes each of them in a slot of its
    # own, so it is never drawn through a block
    first, last = {}, {}
    for n in nodes:
        for g in n.groups:
            first[g] = min(first.get(g, col[n]), col[n])
            last[g] = max(last.get(g, col[n]), col[n])
    segs = []  # (a, port fraction at a, b, port fraction at b), adjacent columns
    lanes = []
    for e in fwd:
        way = [
            _Lane(e, c, _lane_groups(e, c, first, last))
            for c in range(col[e.src] + 1, col[e.dst])
        ]
        e.lanes = way
        lanes += way
        pts = [(e.src, _frac(e.src.outputs, e.sport)), *[(w, 0.0) for w in way],
               (e.dst, _frac(e.dst.inputs, e.dport))]  # fmt: skip
        segs += [(a, fa, b, fb) for (a, fa), (b, fb) in pairwise(pts)]
    for w in lanes:
        col[w] = w.col
    start = [[n for n in [*nodes, *lanes] if col[n] == c] for c in range(ncol)]
    grouped = any(n.groups for n in nodes)
    if grouped:
        start = [_together(c) for c in start]

    # order within columns by barycenters of neighbours, sweeping both ways;
    # port fractions refine them, so a block feeding x0 sorts above one
    # feeding x1. ties put lanes first or last; the order with the fewest
    # crossings seen wins
    into, out = defaultdict(list), defaultdict(list)
    for a, fa, b, fb in segs:
        into[b].append((a, fa))
        out[a].append((b, fb))
    best, best_x = start, _crossings(start, segs, col)
    for lanes_first in (True, False):
        cols = [list(c) for c in start]
        pos = {n: i for c in cols for i, n in enumerate(c)}
        for sweep in range(6):
            nbs = into if sweep % 2 == 0 else out
            order = range(1, ncol) if sweep % 2 == 0 else range(ncol - 2, -1, -1)
            for c in order:

                def bary(n, nbs=nbs, pos=pos):  # pos: updated in place
                    nb = [pos[m] + f for m, f in nbs[n]]
                    return sum(nb) / len(nb) if nb else pos[n]

                cols[c].sort(key=lambda n: (bary(n), _tie(n, lanes_first)))
                if grouped:
                    cols[c] = _together(cols[c])
                pos.update((n, i) for i, n in enumerate(cols[c]))
            x = _crossings(cols, segs, col)
            if x < best_x:
                best, best_x = [list(c) for c in cols], x
    return best, edges


# text


def _pname(n, port, kind):
    ports = n.inputs if kind == "in" else n.outputs
    return f"{n.name}[{port}]" if len(ports) > 1 else n.name


def to_text(d, expand=False) -> str:
    """Return the diagram as text: signal chains, feedback, probes, blocks.

    Parameters
    ----------
    d : Diagram
        Or a subsystem's inside.
    expand : bool, default False
        List the blocks inside subsystems, by their path (``"fm/scale"``).

    Returns
    -------
    str
    """
    nodes, edges = view(d, expand)
    cols, edges = _layout(nodes, edges)
    cols = _blocks(cols)
    indeg = defaultdict(int)
    out_of = defaultdict(list)
    for e in edges:
        indeg[e.dst] += 1
        if not e.feedback:
            out_of[e.src].append(e)
    nfb = sum(e.feedback for e in edges)
    root = getattr(d, "root", None)
    what = "Diagram" if root is None else f"{root.name} ({root.type})"
    rate = f"  fs = {si(d.fs, 'Hz')}" if d.fs else ""
    rows = [f"{what}{rate}  |  {len(nodes)} blocks, {len(edges)} lines, {nfb} feedback"]

    # chains: blocks with one input and one forward line out print in a row
    def passthrough(n):
        return indeg[n] == 1 and len(n.outputs) == 1 and len(out_of[n]) == 1

    used, chains = set(), []

    def follow(e):
        chain = [e]
        used.add(id(e))
        while passthrough(chain[-1].dst):
            nxt = out_of[chain[-1].dst][0]
            if id(nxt) in used:
                break
            chain.append(nxt)
            used.add(id(nxt))
        return chain

    colof = {n: c for c, ns in enumerate(cols) for n in ns}
    order = sorted(
        edges,
        key=lambda e: (e.feedback, colof[e.src], passthrough(e.src), e.src.block._id),
    )
    for e in order:
        if id(e) not in used and (e.feedback or not passthrough(e.src)):
            chains.append(follow(e))
    for e in edges:  # loops made only of pass-through blocks
        if id(e) not in used:
            chains.append(follow(e))

    def fmt(chain):
        s = _pname(chain[0].src, chain[0].sport, "out")
        for e in chain:
            s += (" ══⟲══▶ " if e.feedback else " ──▶ ") + _pname(e.dst, e.dport, "in")
        return s

    rows += ["", "signal flow:"]
    rows += ["  " + fmt(c) for c in chains if not c[0].feedback]
    fb = [c for c in chains if c[0].feedback]
    if fb:
        rows += ["feedback:"] + ["  " + fmt(c) for c in fb]
    if d._probes:
        rows.append("probes:")
        for (blk, port), scopes in d._probes.items():
            rows += [f"  {sc.name}  <-  {blk.name}[{port}]" for sc in scopes]
    if root is not None:
        rows.append("ports:")
        for name, p in root._exposed("in"):
            rows.append(f"  {name} ──▶ {_rel(p.block.name, root)}[{p.name}]")
        for name, p in root._exposed("out"):
            rows.append(f"  {_rel(p.block.name, root)}[{p.name}] ──▶ {name}")

    rows += ["", "blocks:"]
    w = max(len(n.name) for n in nodes)
    wt = max(len(n.type) for n in nodes)
    multirate = len({n.fs for n in nodes if n.fs}) > 1
    wr = max(len(n.rate_text()) for n in nodes)
    for c, column in enumerate(cols):
        for n in column:
            rate = f"{n.rate_text():>{wr}}  " if multirate else ""
            rows.append(
                f"  [{c}] {n.name:<{w}}  {n.type:<{wt}}  {rate}{n.info}".rstrip()
            )
    return "\n".join(rows)


# matplotlib

_FILL = {
    "source": "#dff0d8",
    "sink": "#ececec",
    "state": "#efe3f7",
    "block": "#ffffff",
    "subsystem": "#e3eefa",
}


def _gain_text(g):
    # a gain in three digits, for the triangle
    import numpy as np

    if not np.isscalar(g):
        return "g"
    if np.iscomplexobj(g) and np.real(g) == 0:
        return f"{np.imag(g):.3g}j"
    return f"{g:.3g}"


def _symbol(n):
    # (shape, width, height) of a block drawn as a symbol, else None
    from .blocks import Gain, Product, Sum
    from .sources import Const

    kind = type(n.block)
    if kind in (Sum, Product):
        r = 0.2 + 0.06 * max(len(n.inputs) - 2, 0)
        return "circle", 2 * r, 2 * r
    if kind is Gain:
        # nearly equilateral; wider only for long gains
        return "triangle", max(0.46, 0.075 * len(_gain_text(n.block.g)) + 0.15), 0.5
    if kind is Const:
        return "const", max(0.45, 0.08 * len(n.info) + 0.2), 0.3
    return None


def _lines(n, info=False):
    # (first, second) lines of a box: name and type, or the type alone for a
    # name the library gave; info joins the type line, never a third
    kind = f"{n.type}: {n.info}" if info and n.info else n.type
    if _default_name(n):
        return (n.type, n.info) if info and n.info else (n.type, None)
    return n.label(), kind


def _default_name(n):
    # True for names the library gave: the type plus a counter
    return re.fullmatch(rf"{re.escape(n.type)}\d+", n.name.rsplit("/", 1)[-1])


def _frame(piece, by_col, cols, extent, pad, head, dx, box_w):
    # outline of a subsystem over adjacent columns: (points, rectangle?);
    # head: room above for the titles of frames inside
    spans = []
    for c in piece:
        ext = [extent(n) for n in by_col[c]]
        x, w = c * dx, box_w / 2 + pad
        lo, hi = min(e[0] for e in ext) - pad, max(e[1] for e in ext) + pad + head
        spans.append((x - w, x + w, lo, hi))
    lo, hi = min(s[2] for s in spans), max(s[3] for s in spans)
    inside = {id(n) for c in piece for n in by_col[c]}
    others = [extent(n) for c in piece for n in cols[c] if id(n) not in inside]
    if all(top < lo - 0.05 or bottom > hi + 0.05 for bottom, top in others):
        x0, x1 = spans[0][0], spans[-1][1]
        return [(x0, hi), (x1, hi), (x1, lo), (x0, lo)], True
    upper = [(spans[0][0], spans[0][3])]
    lower = [(spans[0][0], spans[0][2])]
    for a, b in pairwise(spans):
        xm = (a[1] + b[0]) / 2
        upper += [(xm, a[3]), (xm, b[3])]
        lower += [(xm, a[2]), (xm, b[2])]
    upper.append((spans[-1][1], spans[-1][3]))
    lower.append((spans[-1][1], spans[-1][2]))
    return upper + lower[::-1], False


def draw(
    d,
    ax=None,
    *,
    save=None,
    info=False,
    expand=False,
    box_w=None,
    gap_x=1.0,
    row_h=1.0,
    fontsize=9,
):
    """Draw the diagram with matplotlib.

    Blocks are boxes with their name and type; ``Sum`` and ``Product`` are
    circles with their signs, ``Gain`` a triangle with its gain, ``Const`` a
    small box with its value, named only when you gave them a name.
    Subsystems are boxes, or opened: their blocks in a dashed frame named
    after the subsystem.

    Parameters
    ----------
    d : Diagram
    ax : matplotlib.axes.Axes, optional
        Default: a new figure sized to the diagram.
    save : str or path, optional
        Also write the figure to this file (any matplotlib format).
    info : bool, default False
        Also show each box's :meth:`Block.info`.
    expand : bool, default False
        Open subsystems, nested ones too.
    box_w : float, optional
        Box width; default: from the longest name.
    gap_x, row_h : float
        Gap between columns, height per row.
    fontsize : float, default 9

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
    from matplotlib.path import Path

    nodes, edges = view(d, expand)
    cols, edges = _layout(nodes, edges)
    multirate = len({n.fs for n in nodes if n.fs}) > 1
    sym = {n: _symbol(n) for n in nodes}
    sym = {n: v for n, v in sym.items() if v is not None}
    if box_w is None:
        # bold first lines are wider per character than the small second
        longest = max(
            [len(_lines(n, info)[0]) for n in nodes if n not in sym]
            + [0.8 * len(_lines(n, info)[1] or "") for n in nodes if n not in sym]
            + [8]
        )
        box_w = min(max(0.085 * longest + 0.35, 1.1), 2.4)
    nports = {n: max(len(n.inputs), len(n.outputs), 1) for n in nodes}
    hbox = {
        n: sym[n][2]
        if n in sym
        else (0.46 if _lines(n, info)[1] else 0.34) + 0.26 * (nports[n] - 1)
        for n in nodes
    }

    def slot(n):
        # vertical space of a block, or of a lane a line passes through
        if isinstance(n, Node):
            named = n in sym and not _default_name(n)
            return max(hbox[n] + 0.16 * named, 0.6) + row_h - 0.6
        return 0.4

    # frames of opened subsystems: margin around blocks, and room above for
    # the titles of the frames inside
    below = defaultdict(int)  # nesting levels inside each frame
    for n in nodes:
        for i, g in enumerate(n.groups):
            below[g] = max(below[g], len(n.groups) - 1 - i)

    def margins(g):
        return 0.1 + 0.08 * below[g], 0.18 * below[g]

    def sep(a, b):
        # room between neighbours for the frames that end or start there
        ga, gb = set(a.groups), set(b.groups)
        if ga == gb:
            return 0.0
        end = max((margins(g)[0] for g in ga - gb), default=0.0)
        start = max((sum(margins(g)) + 0.15 for g in gb - ga), default=0.0)
        return end + start

    # block and lane centres, columns centred vertically
    xy = {}
    col_h = [
        sum(slot(n) for n in c) + sum(sep(a, b) for a, b in pairwise(c)) for c in cols
    ]
    H = max(col_h) if col_h else 1
    for ci, c in enumerate(cols):
        y = col_h[ci] / 2
        for i, n in enumerate(c):
            y -= sep(c[i - 1], n) if i else 0.0
            h = slot(n)
            xy[n] = (ci * (box_w + gap_x), y - h / 2)
            y -= h
    blocks = {n: v for n, v in xy.items() if isinstance(n, Node)}

    def extent(n):
        # (bottom, top) of a block, its name under a symbol included, or a lane
        y = xy[n][1]
        if not isinstance(n, Node):
            return y - 0.05, y + 0.05
        below = 0.22 if n in sym and not _default_name(n) else 0.0
        return y - hbox[n] / 2 - below, y + hbox[n] / 2

    # opened subsystems: a rectangle around their blocks, or, where that
    # would take in other blocks, an outline stepping from column to column
    runs = defaultdict(lambda: defaultdict(list))
    for ci, c in enumerate(cols):
        for n in c:
            for g in n.groups:
                runs[g][ci].append(n)
    frames = []  # (depth, outline, rectangle?, subsystem)
    root = getattr(d, "root", None)
    for g, by_col in runs.items():
        depth = len(_groups(g, root))
        pad, head = margins(g)
        piece = []
        for ci in [*sorted(by_col), None]:
            if piece and (ci is None or ci != piece[-1] + 1):
                outline, rect = _frame(
                    piece, by_col, cols, extent, pad, head, box_w + gap_x, box_w
                )
                frames.append((depth, outline, rect, g))
                piece = []
            if ci is not None:
                piece.append(ci)

    def port_xy(n, port, kind):
        x, y = xy[n]
        ports = n.inputs if kind == "in" else n.outputs
        i, k = ports.index(port), len(ports)
        if n in sym:
            shape, w, h = sym[n]
            if kind == "out":
                return x + w / 2, y
            span = 0.6 * h if k > 1 else 0.0
            py = y + span / 2 - (span * i / (k - 1) if k > 1 else 0.0)
            if shape == "circle":
                return x - ((w / 2) ** 2 - (py - y) ** 2) ** 0.5, py
            return x - w / 2, py
        span = hbox[n] - 0.26
        py = y + span / 2 - (span * i / (k - 1) if k > 1 else span / 2)
        return (x - box_w / 2 if kind == "in" else x + box_w / 2), py

    W = max(len(cols) * (box_w + gap_x), 3)
    n_fb = sum(e.feedback for e in edges)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(min(1.6 + 1.25 * W, 48), 1.0 + 1.0 * H + 0.35 * n_fb)
        )
    ax.set_aspect("equal")
    ax.axis("off")

    small = {"ha": "center", "va": "center", "zorder": 5}
    for n, (x, y) in blocks.items():
        if n in sym:
            shape, w, h = sym[n]
            style = {"fc": _FILL[n.kind], "ec": "#333333", "lw": 1.1, "zorder": 3}
            if shape == "circle":
                ax.add_patch(Circle((x, y), w / 2, **style))
                ops = getattr(n.block, "signs", None) or n.block.ops
                marks = {"+": "+", "-": "−", "*": "×", "/": "÷"}
                if set(ops) == {"*"}:
                    ax.text(x, y, "×", fontsize=fontsize + 1, **small)
                else:
                    for p, op in zip(n.inputs, ops, strict=True):
                        px, py = port_xy(n, p, "in")
                        ax.text(px + 0.1, py, marks[op], fontsize=fontsize - 1, **small)
            elif shape == "triangle":
                ax.add_patch(Polygon([(x - w / 2, y + h / 2), (x - w / 2, y - h / 2),
                                      (x + w / 2, y)], **style))  # fmt: skip
                ax.text(x - w / 6, y, _gain_text(n.block.g),
                        fontsize=fontsize - 3, **small)  # fmt: skip
            else:
                ax.add_patch(FancyBboxPatch(
                    (x - w / 2, y - h / 2), w, h,
                    boxstyle="round,pad=0.01,rounding_size=0.05", **style))  # fmt: skip
                ax.text(x, y, n.info, fontsize=fontsize - 2, **small)
            if not _default_name(n):
                ax.text(x, y - h / 2 - 0.09, n.label(), ha="center", va="top",
                        fontsize=fontsize - 3, color="#555555", zorder=5)  # fmt: skip
            for kind, ports in (("in", n.inputs), ("out", n.outputs)):
                for p in ports:
                    ax.plot(*port_xy(n, p, kind), marker="o", ms=2.5,
                            color="#333333", zorder=6)  # fmt: skip
            continue
        h = hbox[n]
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - h / 2), box_w, h,
                boxstyle="round,pad=0,rounding_size=0.08",
                fc=_FILL[n.kind], ec="#333333", lw=1.1, zorder=3,
            )
        )  # fmt: skip
        first, second = _lines(n, info)
        rows = [(first, {"fontsize": fontsize, "weight": "bold"})]
        if second:
            style = {"fontsize": fontsize - 2, "color": "#555555", "style": "italic"}
            rows.append((second, style))
        top = y + 0.09 * (len(rows) - 1) + 0.02
        for i, (txt, style) in enumerate(rows):
            ax.text(x, top - 0.18 * i, txt, ha="center", va="center", zorder=5, **style)
        for kind, ports in (("in", n.inputs), ("out", n.outputs)):
            for p in ports:
                px, py = port_xy(n, p, kind)
                ax.plot(px, py, marker="o", ms=3, color="#333333", zorder=6)
                if len(ports) > 1:
                    ax.text(
                        px + (0.06 if kind == "in" else -0.06), py, p,
                        ha="left" if kind == "in" else "right", va="center",
                        fontsize=fontsize - 3, color="#333333", zorder=5,
                    )  # fmt: skip

    titled = set()
    for depth, outline, rect, g in frames:
        style = {"fc": "#f3f7fc" if depth % 2 == 0 else "#e8f0f9", "ec": "#7a9cc6",
                 "lw": 0.9, "ls": "--", "zorder": 0.5 + 0.1 * depth}  # fmt: skip
        if rect:
            (x0, y1), (x1, _), _, (_, y0) = outline
            box = "round,pad=0,rounding_size=0.1"
            ax.add_patch(
                FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=box, **style)
            )
        else:
            ax.add_patch(Polygon(outline, closed=True, **style))
        if g not in titled:
            titled.add(g)
            ax.text(outline[0][0] + 0.06, outline[0][1] + 0.03, _frame_title(g),
                    ha="left", va="bottom", fontsize=fontsize - 2, color="#1f4e79",
                    zorder=5)  # fmt: skip

    # lines: forward as curves, feedback dashed below the diagram
    bottom = min(
        y - (hbox[n] / 2 if isinstance(n, Node) else 0.15) for n, (_, y) in xy.items()
    )
    bottom = min([bottom] + [y for f in frames for _, y in f[1]])
    fanout = defaultdict(int)
    for e in edges:
        fanout[(e.src, e.sport)] += 1
    fb_i = 0
    for e in edges:
        x1, y1 = port_xy(e.src, e.sport, "out")
        x2, y2 = port_xy(e.dst, e.dport, "in")
        if not e.feedback and x2 > x1:
            # curves between columns, straight through each lane
            pts = [(x1, y1)]
            for w in e.lanes:
                xl, yl = xy[w]
                pts += [(xl - box_w / 2, yl), (xl + box_w / 2, yl)]
            pts.append((x2, y2))
            verts, codes = [pts[0]], [Path.MOVETO]
            for i in range(0, len(pts) - 1, 2):
                (xa, ya), (xb, yb) = pts[i], pts[i + 1]
                dx = max((xb - xa) * 0.5, 0.3)
                verts += [(xa + dx, ya), (xb - dx, yb), (xb, yb)]
                codes += [Path.CURVE4] * 3
                if i + 2 < len(pts):
                    verts.append(pts[i + 2])
                    codes.append(Path.LINETO)
            style = {"ec": "#222222", "lw": 1.2}
        else:
            fb_i += 1
            yb = bottom - 0.3 * fb_i
            verts = [(x1, y1), (x1 + 0.25, y1), (x1 + 0.25, yb), (x2 - 0.25, yb),
                     (x2 - 0.25, y2), (x2, y2)]  # fmt: skip
            codes = [Path.MOVETO] + [Path.LINETO] * 5
            style = {"ec": "#c0392b", "lw": 1.2, "ls": "--"}
        ax.add_patch(PathPatch(Path(verts, codes), fc="none", zorder=2, **style))
        ax.annotate(
            "", xy=(x2, y2), xytext=(x2 - 0.12, y2),
            arrowprops={"arrowstyle": "-|>", "color": style["ec"], "lw": 1.2,
                        "shrinkA": 0, "shrinkB": 2},
            zorder=4,
        )  # fmt: skip
        if fanout[(e.src, e.sport)] > 1:
            ax.plot(x1 + 0.08, y1, "o", ms=4, color="#222222", zorder=4)

    # rates: in a multirate diagram, a label under each line leaving a block
    # that changes the rate (or a source at its own rate)
    if multirate:
        labelled = set()
        for e in edges:
            n = e.src
            if e.feedback or (n, e.sport) in labelled or not n.fs_out:
                continue
            ref = n.fs if n.inputs else d.fs
            if ref and abs(n.fs_out - ref) <= 1e-9 * ref:
                continue
            labelled.add((n, e.sport))
            x1, y1 = port_xy(n, e.sport, "out")
            ax.text(x1 + 0.06, y1 - 0.05, si(n.fs_out, "Hz"), ha="left", va="top",
                    fontsize=fontsize - 3, color="#7a4f01", zorder=5)  # fmt: skip

    # probes: a tag on the probed line, its text upright in the gap between
    # columns, where no box sits; upper ports point up, lower ports down, so
    # tags never cross a block's other lines
    node_of = {n.block: n for n in nodes}
    for (blk, port), scopes in d._probes.items():
        shown = _shown(blk, port, "out", getattr(d, "root", None), expand)
        if shown is None:
            continue  # inside a subsystem: not drawn
        n, port = node_of[shown[0]], shown[1]
        px, py = port_xy(n, port, "out")
        i, k = n.outputs.index(port), len(n.outputs)
        s = -1 if k > 1 and i >= k / 2 else 1
        tx = px + 0.18
        ax.plot(
            tx,
            py + 0.1 * s,
            marker="v" if s > 0 else "^",
            ms=5,
            color="#1f77b4",
            zorder=7,
        )
        ax.text(tx, py + 0.2 * s, ", ".join(sc.name for sc in scopes),
                rotation=90, ha="left" if s > 0 else "right", va="center",
                rotation_mode="anchor", fontsize=fontsize - 3, color="#1f77b4",
                zorder=7)  # fmt: skip

    # a subsystem drawn alone: its ports as short labelled lines in and out
    root = getattr(d, "root", None)
    if root is not None:
        node_of = {n.block: n for n in nodes}
        for kind, sign in (("in", -1), ("out", 1)):
            for name, p in root._exposed(kind):
                shown = _shown(p.block, p.name, kind, root, expand)
                if shown is None:
                    continue
                px, py = port_xy(node_of[shown[0]], shown[1], kind)
                x0 = px + 0.45 * sign
                start, end = (
                    ((x0, py), (px, py)) if kind == "in" else ((px, py), (x0, py))
                )
                ax.annotate(
                    "", xy=end, xytext=start, zorder=4,
                    arrowprops={"arrowstyle": "-|>", "color": "#1f4e79", "lw": 1.2,
                                "shrinkA": 0, "shrinkB": 2},
                )  # fmt: skip
                ax.text(
                    x0 + 0.05 * sign, py, name, ha="right" if sign < 0 else "left",
                    va="center", fontsize=fontsize - 1, color="#1f4e79",
                    weight="bold", zorder=5,
                )  # fmt: skip

    xs = [x for x, _ in xy.values()]
    ax.set_xlim(min(xs) - box_w / 2 - 0.5, max(xs) + box_w / 2 + 0.5)
    top = max(
        y + (hbox[n] / 2 if isinstance(n, Node) else 0.15) for n, (_, y) in xy.items()
    )
    top = max([top] + [y + 0.2 for f in frames for _, y in f[1]])
    ax.set_ylim(bottom - 0.3 * fb_i - 0.35, top + 0.35)
    ax.set_title(
        f"fs = {si(d.fs, 'Hz')}" if d.fs else getattr(d, "root", d).name,
        fontsize=fontsize,
        loc="left",
        color="#555555",
    )
    if save:
        ax.figure.savefig(save, bbox_inches="tight", dpi=150)
    return ax


# graphviz and mermaid


def _tree(nodes):
    # nodes and child frames per frame, the frame given by its path
    members, children = defaultdict(list), defaultdict(list)
    for n in nodes:
        members[n.groups].append(n)
        for k in range(len(n.groups)):
            if n.groups[: k + 1] not in children[n.groups[:k]]:
                children[n.groups[:k]].append(n.groups[: k + 1])
    return members, children


def _frame_title(g):
    # a frame's name, with its type unless the name is the type numbered
    name = _rel(g.name, g._parent)
    if g.type == "Subsystem" or re.fullmatch(rf"{re.escape(g.type)}\d+", name):
        return name
    return f"{name} · {g.type}"


def to_dot(d, info=False, expand=False) -> str:
    """Return the diagram as Graphviz DOT source.

    Render with ``dot -Tpdf`` or Typst's ``diagraph`` package.

    Parameters
    ----------
    d : Diagram
    info : bool, default False
        Show each block's :meth:`Block.info`.
    expand : bool, default False
        Open subsystems as clusters.

    Returns
    -------
    str
    """
    nodes, edges = view(d, expand)
    _layout(nodes, edges)  # marks feedback lines

    def esc(s):
        for c in '\\"{}|<>':
            s = str(s).replace(c, "\\" + c)
        return s

    def nid(n):
        return f"b{n.block._id}"

    out = [
        "digraph sigline {",
        "  rankdir=LR;",
        "  splines=spline;",
        '  node [shape=record, fontname="Helvetica", fontsize=10,',
        '        style="rounded,filled"];',
        "  edge [arrowsize=0.7];",
    ]

    def node(n):
        first, second = _lines(n, info)
        mid = esc(first) + (f"\\n{esc(second)}" if second else "")
        parts = []
        if n.inputs:
            show = len(n.inputs) > 1
            parts.append(
                "{" + "|".join(f"<i_{p}> {esc(p) * show}" for p in n.inputs) + "}"
            )
        parts.append(mid)
        if n.outputs:
            show = len(n.outputs) > 1
            parts.append(
                "{" + "|".join(f"<o_{p}> {esc(p) * show}" for p in n.outputs) + "}"
            )
        return f'{nid(n)} [label="{{{"|".join(parts)}}}", fillcolor="{_FILL[n.kind]}"];'

    members, children = _tree(nodes)

    def emit(path, ind):
        out.extend(ind + node(n) for n in members[path])
        for child in children[path]:
            title = _frame_title(child[-1]).replace("\\", "\\\\").replace('"', '\\"')
            out.append(f"{ind}subgraph cluster_{child[-1]._id} {{")
            out.append(f'{ind}  label="{title}"; labeljust=l; fontsize=9;')
            out.append(f'{ind}  style="rounded,dashed"; color="#7a9cc6";')
            out.append(f'{ind}  fontcolor="#1f4e79";')
            emit(child, ind + "  ")
            out.append(ind + "}")

    emit((), "  ")
    for e in edges:
        attr = (
            ' [color="#c0392b", style=dashed, constraint=false]' if e.feedback else ""
        )
        out.append(f"  {nid(e.src)}:o_{e.sport}:e -> {nid(e.dst)}:i_{e.dport}:w{attr};")
    out.append("}")
    return "\n".join(out)


# words mermaid reads as syntax, not as a node id
_MERMAID_WORDS = {
    "call", "class", "classdef", "click", "default", "direction", "end",
    "flowchart", "graph", "linkstyle", "style", "subgraph",
}  # fmt: skip


def _mermaid_ids(named):
    # ids from (key, name) pairs: letters, digits and _, unique
    ids, used = {}, set()
    for key, name in named:
        base = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "block"
        if base[0].isdigit() or base.lower() in _MERMAID_WORDS:
            base = f"b_{base}"
        nid, k = base, 2
        while nid in used:
            nid, k = f"{base}_{k}", k + 1
        ids[key] = nid
        used.add(nid)
    return ids


def to_mermaid(d, expand=False) -> str:
    """Return the diagram as a Mermaid flowchart (GitHub renders these).

    Node ids are the block names made safe for Mermaid (``"mix i"`` becomes
    ``mix_i``), so the source reads like the diagram.

    Parameters
    ----------
    d : Diagram
    expand : bool, default False
        Open subsystems as subgraphs.

    Returns
    -------
    str
    """
    nodes, edges = view(d, expand)
    _layout(nodes, edges)
    members, children = _tree(nodes)
    frames = [path[-1] for paths in children.values() for path in paths]
    root = getattr(d, "root", None)
    ids = _mermaid_ids(
        [(n.block, n.name) for n in nodes] + [(g, _rel(g.name, root)) for g in frames]
    )

    def nid(n):
        return ids[n.block]

    out = ["flowchart LR"]

    def emit(path, ind):
        for n in members[path]:
            first, second = _lines(n)
            small = f"<br/><small>{second}</small>" if second else ""
            label = f"{first}{small}".replace('"', "'")
            out.append(f'{ind}{nid(n)}["{label}"]')
        for child in children[path]:
            title = _frame_title(child[-1]).replace('"', "'")
            out.append(f'{ind}subgraph {ids[child[-1]]} ["{title}"]')
            emit(child, ind + "  ")
            out.append(f"{ind}end")

    emit((), "  ")
    for e in edges:
        arrow = "-.->" if e.feedback else "-->"
        ports = []
        if len(e.src.outputs) > 1:
            ports.append(e.sport)
        if len(e.dst.inputs) > 1:
            ports.append(e.dport)
        label = f"|{'→'.join(ports)}|" if ports else ""
        out.append(f"  {nid(e.src)} {arrow}{label} {nid(e.dst)}")
    return "\n".join(out)
