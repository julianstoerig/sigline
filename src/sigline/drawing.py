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


def _frac(ports, port):
    # position of a port within its block, in [0, 1)
    return ports.index(port) / len(ports)


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
                # neighbour positions, refined by the port used, so a block
                # feeding input x0 sorts above one feeding x1
                if forward:
                    nb = [pos[e.src] + _frac(e.src.outputs, e.sport)
                          for e in fwd if e.dst is n and col[e.src] < c]  # fmt: skip
                else:
                    nb = [pos[e.dst] + _frac(e.dst.inputs, e.dport)
                          for e in fwd if e.src is n and col[e.dst] > c]  # fmt: skip
                return sum(nb) / len(nb) if nb else pos[n]

            cols[c].sort(key=lambda n: (bary(n), n.block._id))
    return cols, edges


# text


def _pname(n, port, kind):
    ports = n.inputs if kind == "in" else n.outputs
    return f"{n.name}[{port}]" if len(ports) > 1 else n.name


def to_text(d) -> str:
    """Return the diagram as text: signal chains, feedback, probes, blocks.

    Parameters
    ----------
    d : Diagram

    Returns
    -------
    str
    """
    nodes, edges = view(d)
    cols, edges = _layout(nodes, edges)
    indeg = defaultdict(int)
    out_of = defaultdict(list)
    for e in edges:
        indeg[e.dst] += 1
        if not e.feedback:
            out_of[e.src].append(e)
    nfb = sum(e.feedback for e in edges)
    rows = [
        f"Diagram  fs = {si(d.fs, 'Hz')}  |  {len(nodes)} blocks, "
        f"{len(edges)} lines, {nfb} feedback"
    ]

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

_FILL = {"source": "#dff0d8", "sink": "#ececec", "state": "#efe3f7", "block": "#ffffff"}


def draw(
    d, ax=None, *, save=None, info=True, box_w=1.7, gap_x=1.1, row_h=1.25, fontsize=9
):
    """Draw the diagram with matplotlib.

    Parameters
    ----------
    d : Diagram
    ax : matplotlib.axes.Axes, optional
        Default: a new figure sized to the diagram.
    save : str or path, optional
        Also write the figure to this file (any matplotlib format).
    info : bool, default True
        Show each block's :meth:`Block.info`.
    box_w, gap_x, row_h : float
        Box width, gap between columns, height per row.
    fontsize : float, default 9

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, PathPatch
    from matplotlib.path import Path

    nodes, edges = view(d)
    cols, edges = _layout(nodes, edges)
    multirate = len({n.fs for n in nodes if n.fs}) > 1
    nports = {n: max(len(n.inputs), len(n.outputs), 1) for n in nodes}
    hbox = {
        n: 0.55
        + 0.28 * (nports[n] - 1)
        + (0.2 if info and n.info else 0)
        + (0.2 if multirate and n.fs else 0)
        for n in nodes
    }

    # box centres, columns centred vertically
    xy = {}
    col_h = [sum(max(hbox[n], 0.7) + row_h - 0.7 for n in c) for c in cols]
    H = max(col_h) if col_h else 1
    for ci, c in enumerate(cols):
        y = col_h[ci] / 2
        for n in c:
            h = max(hbox[n], 0.7) + row_h - 0.7
            xy[n] = (ci * (box_w + gap_x), y - h / 2)
            y -= h

    def port_xy(n, port, kind):
        x, y = xy[n]
        ports = n.inputs if kind == "in" else n.outputs
        i, k = ports.index(port), len(ports)
        span = hbox[n] - 0.3
        py = y + span / 2 - (span * i / (k - 1) if k > 1 else span / 2)
        return (x - box_w / 2 if kind == "in" else x + box_w / 2), py

    W = max(len(cols) * (box_w + gap_x), 3)
    n_fb = sum(e.feedback for e in edges)
    if ax is None:
        _, ax = plt.subplots(
            figsize=(min(1.6 + 1.25 * W, 18), 1.0 + 0.9 * H + 0.35 * n_fb)
        )
    ax.set_aspect("equal")
    ax.axis("off")

    for n, (x, y) in xy.items():
        h = hbox[n]
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - h / 2), box_w, h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=_FILL[n.kind], ec="#333333", lw=1.1, zorder=3,
            )
        )  # fmt: skip
        rows = [
            (n.name, {"fontsize": fontsize, "weight": "bold"}),
            (n.type, {"fontsize": fontsize - 2, "color": "#555555", "style": "italic"}),
        ]
        if info and n.info:
            rows.append((n.info, {"fontsize": fontsize - 2, "color": "#1f4e79"}))
        if multirate and n.fs:
            rows.append(
                (n.rate_text(), {"fontsize": fontsize - 2.5, "color": "#7a4f01"})
            )
        top = y + 0.1 * (len(rows) - 1) + 0.03
        for i, (txt, style) in enumerate(rows):
            ax.text(x, top - 0.2 * i, txt, ha="center", va="center", zorder=5, **style)
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

    # lines: forward as curves, feedback dashed below the diagram
    bottom = min(y - hbox[n] / 2 for n, (_, y) in xy.items())
    fanout = defaultdict(int)
    for e in edges:
        fanout[(e.src, e.sport)] += 1
    fb_i = 0
    for e in edges:
        x1, y1 = port_xy(e.src, e.sport, "out")
        x2, y2 = port_xy(e.dst, e.dport, "in")
        if not e.feedback and x2 > x1:
            dx = max((x2 - x1) * 0.5, 0.3)
            verts = [(x1, y1), (x1 + dx, y1), (x2 - dx, y2), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
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

    # probes: a tag on the probed line, its text upright in the gap between
    # columns, where no box sits; upper ports point up, lower ports down, so
    # tags never cross a block's other lines
    node_of = {n.block: n for n in nodes}
    for (blk, port), scopes in d._probes.items():
        n = node_of[blk]
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

    xs = [x for x, _ in xy.values()]
    ax.set_xlim(min(xs) - box_w / 2 - 0.5, max(xs) + box_w / 2 + 0.5)
    top = max(y + hbox[n] / 2 for n, (_, y) in xy.items())
    ax.set_ylim(bottom - 0.3 * fb_i - 0.35, top + 0.35)
    ax.set_title(
        f"fs = {si(d.fs, 'Hz')}", fontsize=fontsize, loc="left", color="#555555"
    )
    if save:
        ax.figure.savefig(save, bbox_inches="tight", dpi=150)
    return ax


# graphviz and mermaid


def to_dot(d, info=True) -> str:
    """Return the diagram as Graphviz DOT source.

    Render with ``dot -Tpdf`` or Typst's ``diagraph`` package.

    Parameters
    ----------
    d : Diagram
    info : bool, default True
        Show each block's :meth:`Block.info`.

    Returns
    -------
    str
    """
    nodes, edges = view(d)
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
    for n in nodes:
        mid = f"{esc(n.name)}\\n{esc(n.type)}"
        if info and n.info:
            mid += f"\\n{esc(n.info)}"
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
        out.append(
            f'  {nid(n)} [label="{{{"|".join(parts)}}}", fillcolor="{_FILL[n.kind]}"];'
        )
    for e in edges:
        attr = (
            ' [color="#c0392b", style=dashed, constraint=false]' if e.feedback else ""
        )
        out.append(f"  {nid(e.src)}:o_{e.sport}:e -> {nid(e.dst)}:i_{e.dport}:w{attr};")
    out.append("}")
    return "\n".join(out)


def to_mermaid(d) -> str:
    """Return the diagram as a Mermaid flowchart (GitHub renders these).

    Parameters
    ----------
    d : Diagram

    Returns
    -------
    str
    """
    nodes, edges = view(d)
    _layout(nodes, edges)

    def nid(n):
        return f"b{n.block._id}"

    out = ["flowchart LR"]
    for n in nodes:
        label = f"{n.name}<br/><small>{n.type}</small>".replace('"', "'")
        out.append(f'  {nid(n)}["{label}"]')
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
