"""Subsystems: a piece of diagram used as one block, and open to change.

A plain function wires ordinary blocks and returns them as a subsystem,
naming which inner ports are its inputs and outputs::

    def upconverter(fc, fs, cutoff):
        mix = sl.Product(name="mix")
        sl.Sine(fc, name="lo") >> mix["x1"]
        hpf = mix >> sl.FIR(sig.firwin(101, cutoff, pass_zero=False, fs=fs), name="hpf")
        return sl.Subsystem(inputs={"x": mix["x0"]}, outputs={"y": hpf})

    up = upconverter(10e3, 48e3, 8e3)
    src >> up >> out                                # one block from outside
    up.lo.f = 12e3                                  # inner blocks by name
    up.replace(up.hpf, sl.IIR(sos=...))             # swap one, lines kept

The ports are the inner blocks' own ports, so a diagram runs exactly the
blocks inside: no extra blocks, no overhead. Members are the blocks behind
the ports when the subsystem is made: connected to them, but not through
lines leaving an output port, which lead outside. Inner blocks are renamed
by path (``"upconverter1/hpf"``).
Drawings show a subsystem as one box; ``up.draw()`` shows its inside.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from .core import (
    Block,
    Port,
    PortGroup,
    _Arithmetic,
    _as_in_port,
    _as_out_port,
    _connect,
    _disconnect,
    _fan_in,
)

__all__ = ["Subsystem"]


def _key(port):
    return (port.block, port.name, port.kind)


def _inside(in_ports, out_ports):
    # blocks reachable over lines from the ports' blocks, but not over lines
    # leaving through an exposed output: those lead outside (lines into the
    # exposed inputs cannot exist yet)
    leaving = {(p.block, p.name) for p in out_ports}
    seen, todo = set(), [p.block for p in [*in_ports, *out_ports]]
    while todo:
        b = todo.pop()
        if b in seen:
            continue
        seen.add(b)
        todo += [p.block for p in b._sources.values()]
        todo += [
            q.block
            for q in b._sinks
            if (b, q.block._sources[q.name].name) not in leaving
        ]
    return seen


class Subsystem(_Arithmetic):
    """Blocks used as one block with named inputs and outputs.

    Parameters
    ----------
    inputs : dict, optional
        ``{name: inner input}``: an input port, a block with one input, or a
        list of these, all fed by the line to this input.
    outputs : dict, optional
        ``{name: inner output}``: an output port, or a block with one
        output.
    name : str, optional
        Default: the calling function's name plus a counter, e.g.
        ``"upconverter1"``.
    info : str or callable, optional
        One-line summary for drawings, or a function returning it, so it
        follows parameter changes.

    Attributes
    ----------
    name, type : str
        ``type`` is the name without the counter.
    inputs, outputs : tuple of str
    members : list
        Blocks and subsystems directly inside.

    Raises
    ------
    ValueError
        A port given is already connected from outside, or no ports.
    """

    _counter: defaultdict[str, int] = defaultdict(int)

    def __init__(self, inputs=None, outputs=None, *, name=None, info=""):
        inputs, outputs = dict(inputs or {}), dict(outputs or {})
        if not inputs and not outputs:
            raise ValueError("a subsystem needs at least one input or output")
        caller = sys._getframe(1).f_code.co_name
        self.type = caller if caller.isidentifier() else "Subsystem"
        Subsystem._counter[self.type] += 1
        Block._counter["__all__"] += 1
        self._id = Block._counter["__all__"]
        self.name = name or f"{self.type}{Subsystem._counter[self.type]}"
        self._info, self._parent = info, None
        self._in = {k: self._ports(v) for k, v in inputs.items()}
        self._out = {k: _as_out_port(v) for k, v in outputs.items()}
        for k, ps in self._in.items():
            for p in ps:
                if p.name in p.block._sources:
                    raise ValueError(f"input {k!r} ({p}) is already connected")
        self.inputs, self.outputs = tuple(self._in), tuple(self._out)
        # adopt the blocks behind the ports, or the outermost subsystem
        # holding them
        self._members = []
        inner = [p for ps in self._in.values() for p in ps]
        for b in sorted(_inside(inner, self._out.values()), key=lambda b: b._id):
            top = b
            while top._parent is not None:
                top = top._parent
            if top is not self and top not in self._members:
                top._parent = self
                self._members.append(top)
        for b in self.blocks:
            b.name = f"{self.name}/{b.name}"

    @staticmethod
    def _ports(v):
        # one inner input, or several fed together -> list of ports
        vs = v if isinstance(v, list | tuple) and not isinstance(v, PortGroup) else [v]
        out = []
        for x in vs:
            p = _as_in_port(x)
            out += list(p) if isinstance(p, PortGroup) else [p]
        return out

    # contents
    @property
    def members(self) -> list:
        """Blocks and subsystems directly inside."""
        return list(self._members)

    @property
    def blocks(self) -> list:
        """Every block inside, nested ones included, by creation."""
        out = []
        for m in self._members:
            out += m.blocks if isinstance(m, Subsystem) else [m]
        return sorted(out, key=lambda b: b._id)

    def __getattr__(self, short):
        """Return the member called ``short``, e.g. ``up.hpf``."""
        if short.startswith("_"):
            raise AttributeError(short)
        for m in self.__dict__.get("_members", ()):
            if m.name.rsplit("/", 1)[-1] == short:
                return m
        raise AttributeError(f"{self.name} has no member {short!r}")

    def replace(self, old, new):
        """Put block ``new`` in the place of member ``old``, with its lines.

        Ports are matched by name, else by position. Diagrams pick the new
        block up at their next run; probes on the old one are dropped.

        Parameters
        ----------
        old : Block or str
            A block inside, or a member's short name.
        new : Block
            Unconnected.

        Returns
        -------
        Block
            ``new``.

        Raises
        ------
        ValueError
            ``old`` is not a block of this subsystem, ``new`` is connected,
            or the ports do not match.
        """
        old = getattr(self, old) if isinstance(old, str) else old
        if old not in self.blocks:
            raise ValueError(f"{getattr(old, 'name', old)} is not in {self.name}")
        if new._sources or new._sinks:
            raise ValueError(f"{new.name} is already connected")

        def match(port, old_ports, new_ports):
            if port in new_ports:
                return port
            i = old_ports.index(port)
            if i >= len(new_ports):
                raise ValueError(f"{new.name} has no port for {old.name}[{port!r}]")
            return new_ports[i]

        imap = {k: match(k, old.inputs, new.inputs) for k in old.inputs}
        omap = {k: match(k, old.outputs, new.outputs) for k in old.outputs}
        for k in list(old._sources):
            _connect(_disconnect(old.inp(k)), new.inp(imap[k]))
        for q in list(old._sinks):
            _connect(new.out(omap[_disconnect(q).name]), q)
        # exposed ports, membership, name
        for sub in self._chain(old):
            sub._in = {k: [new.inp(imap[p.name]) if p.block is old else p
                           for p in ps]
                       for k, ps in sub._in.items()}  # fmt: skip
            sub._out = {k: new.out(omap[p.name]) if p.block is old else p
                        for k, p in sub._out.items()}  # fmt: skip
        parent = old._parent
        parent._members[parent._members.index(old)] = new
        new._parent, old._parent = parent, None
        new.name = f"{parent.name}/{new.name}"
        return new

    def _chain(self, b):
        # subsystems from b's parent up to and including self
        out, sub = [], b._parent
        while sub is not None:
            out.append(sub)
            if sub is self:
                break
            sub = sub._parent
        return out

    def _exposed(self, kind):
        # (name, inner port) pairs; an input may have several
        if kind == "in":
            return [(k, p) for k, ps in self._in.items() for p in ps]
        return list(self._out.items())

    def _exposes(self, port) -> bool:
        return any(_key(p) == _key(port) for _, p in self._exposed(port.kind))

    def _port_name(self, block, port, kind):
        # the name under which this subsystem exposes block[port], or None
        for k, p in self._exposed(kind):
            if p.block is block and p.name == port:
                return k
        return None

    # ports, as for a block
    def __getitem__(self, port) -> Port:
        """Return port ``port``, input or output."""
        if port in self.inputs and port in self.outputs:
            raise KeyError(f"{port!r} is both input and output; use .inp() / .out()")
        if port in self.inputs:
            return self.inp(port)
        if port in self.outputs:
            return self._out[port]
        raise KeyError(f"{self.name} has no port {port!r}")

    def inp(self, port) -> Port | PortGroup:
        """Return input ``port``: one inner port, or all it feeds."""
        ps = self._in[port]
        return ps[0] if len(ps) == 1 else PortGroup(ps)

    def out(self, port) -> Port:
        """Return output ``port``."""
        return self._out[port]

    def _port(self, kind):
        ports = self.inputs if kind == "in" else self.outputs
        if len(ports) != 1:
            raise ValueError(f"{self.name} has {kind}puts {ports}; pick one by name")
        return self[ports[0]]

    def __rshift__(self, other):
        """Draw a line from the only output to ``other``."""
        return _connect(self, other)

    def __rrshift__(self, other):
        """Feed a tuple of signals to the inputs in order: ``(a, b) >> sub``."""
        if not isinstance(other, tuple | list):
            return NotImplemented
        _fan_in(other, self)
        return self

    def __lshift__(self, other):
        """Draw a line from ``other`` to the only input; return ``other``."""
        from .core import _connect_back

        return _connect_back(self, other)

    # standalone use as a plain function, like a block
    def __call__(self, *args, fs=None, **inputs):
        """Run the blocks inside on one chunk, outside a diagram.

        Like a block: inputs positionally or by name, a subsystem without
        inputs takes the length; state carries over between calls. It runs
        as a small diagram, which needs a sample rate.

        Parameters
        ----------
        *args, **inputs : numpy.ndarray or int
            One array per input, or the length.
        fs : float, optional
            Input sample rate / Hz; default: that of the blocks the inputs
            feed, or the one rate all blocks inside have.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            One array, or a tuple for several outputs.

        Raises
        ------
        ValueError
            Lines to blocks outside, or no single sample rate.
        """
        from .diagram import Diagram
        from .scope import Scope
        from .sources import FromArray

        blocks = self.blocks
        inner = set(blocks)
        for b in blocks:
            ends = [p.block for p in b._sources.values()] + [q.block for q in b._sinks]
            if any(e not in inner for e in ends):
                raise ValueError(f"{self.name} is connected; run its diagram instead")
        if fs is None:
            # the input rate: of the blocks the inputs feed, else of them all
            entry = [p.block for _, p in self._exposed("in")] or blocks
            rates = {b.fs for b in entry if b.fs is not None}
            rates = rates or {b.fs for b in blocks if b.fs is not None}
            if len(rates) != 1:
                have = "several sample rates" if rates else "no sample rate"
                raise ValueError(f"{self.name}: its blocks have {have}; give fs=")
            fs = rates.pop()
        if not self.inputs and len(args) == 1 and not inputs:
            n, feeds = int(args[0]), {}
        else:
            inputs.update(zip(self.inputs, args, strict=False))
            feeds = {k: np.asarray(v) for k, v in inputs.items()}
            n = len(next(iter(feeds.values())))
        saved = {b: b.fs for b in blocks}
        scopes = {k: Scope() for k in self.outputs}
        try:
            for k, x in feeds.items():
                _connect(FromArray(x), self.inp(k))
            for k, sc in scopes.items():
                _connect(self.out(k), sc)
            Diagram(*scopes.values(), *blocks, fs=fs).run(n=n, chunk=n)
        finally:
            for k in feeds:
                for p in self._in[k]:
                    if p.name in p.block._sources:
                        _disconnect(p)
            for sc in scopes.values():
                if "x" in sc._sources:
                    _disconnect(sc.inp("x"))
            for b, f in saved.items():
                b.fs = f
        vals = tuple(scopes[k].data for k in self.outputs)
        return vals[0] if len(vals) == 1 else vals

    # inspection
    def info(self) -> str:
        """Return the summary given, or the number of blocks."""
        if callable(self._info):
            return self._info()
        n = len(self.blocks)
        return self._info or f"{n} block{'s' * (n != 1)}"

    def reset(self):
        """Reset every block inside."""
        for b in self.blocks:
            b.reset()

    def draw(self, ax=None, **kw):
        """Draw the inside: members, nested subsystems as boxes, ports marked.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        **kw
            As for :meth:`Diagram.draw`.
        """
        from .drawing import draw

        return draw(_Inside(self), ax, **kw)

    def to_text(self) -> str:
        """Return the inside as text, like :meth:`Diagram.to_text`."""
        from .drawing import to_text

        return to_text(_Inside(self))

    def to_dot(self, info=False) -> str:
        """Return the inside as Graphviz DOT, like :meth:`Diagram.to_dot`."""
        from .drawing import to_dot

        return to_dot(_Inside(self), info=info)

    def to_mermaid(self) -> str:
        """Return the inside as a Mermaid flowchart, like :meth:`Diagram.to_mermaid`."""
        from .drawing import to_mermaid

        return to_mermaid(_Inside(self))

    def __str__(self):
        """Return :meth:`to_text`."""
        return self.to_text()

    def _repr_svg_(self):
        # jupyter shows a subsystem as its inside
        import io

        import matplotlib.pyplot as plt

        fig = self.draw().figure
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def __repr__(self):
        """Return ``<Subsystem name in=... out=...>``."""
        return f"<Subsystem {self.name} in={self.inputs} out={self.outputs}>"


class _Inside:
    # what drawing needs from a diagram, for a subsystem's inside
    def __init__(self, sub):
        self.root, self.blocks, self._probes = sub, sub.blocks, {}
        self.fs = next((b.fs for b in self.blocks if b.fs), None)
