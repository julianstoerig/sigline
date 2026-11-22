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
blocks inside: no extra blocks, no overhead. Members are the blocks
connected to the ports when the subsystem is made; wrap first, then connect
outside. Inner blocks are renamed by path (``"upconverter1/hpf"``).
Drawings show a subsystem as one box; ``up.draw()`` shows its inside.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from .core import Block, Port, _Arithmetic, _as_in_port, _as_out_port, _connect

__all__ = ["Subsystem"]


def _key(port):
    return (port.block, port.name, port.kind)


def _connected(ports):
    # blocks reachable over lines from the given ports' blocks
    seen, todo = set(), [p.block for p in ports]
    while todo:
        b = todo.pop()
        if b in seen:
            continue
        seen.add(b)
        todo += [p.block for p in b._sources.values()]
        todo += [p.block for p in b._sinks]
    return seen


class Subsystem(_Arithmetic):
    """Blocks used as one block with named inputs and outputs.

    Parameters
    ----------
    inputs : dict, optional
        ``{name: inner input}``: an input port, or a block with one input.
    outputs : dict, optional
        ``{name: inner output}``: an output port, or a block with one
        output.
    name : str, optional
        Default: the calling function's name plus a counter, e.g.
        ``"upconverter1"``.
    info : str, optional
        One-line summary for drawings.

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
        self._in = {k: _as_in_port(v) for k, v in inputs.items()}
        self._out = {k: _as_out_port(v) for k, v in outputs.items()}
        for k, p in self._in.items():
            if p.name in p.block._sources:
                raise ValueError(f"input {k!r} ({p}) is already connected")
        self.inputs, self.outputs = tuple(self._in), tuple(self._out)
        # adopt everything connected: blocks, or the outermost subsystem
        # holding them
        self._members = []
        for b in sorted(_connected([*self._in.values(), *self._out.values()]),
                        key=lambda b: b._id):  # fmt: skip
            top = b
            while top._parent is not None:
                top = top._parent
            if top is not self and top not in self._members:
                top._parent = self
                self._members.append(top)
        for b in self.blocks:
            b.name = f"{self.name}/{b.name}"

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

        Ports are matched by name, else by position. Build diagrams after
        replacing; a diagram made before keeps the old block.

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
        for k, src in list(old._sources.items()):
            src.block._sinks = [q for q in src.block._sinks
                                if not (q.block is old and q.name == k)]  # fmt: skip
            del old._sources[k]
            _connect(src, new.inp(imap[k]))
        for q in list(old._sinks):
            port = q.block._sources[q.name].name
            old._sinks.remove(q)
            del q.block._sources[q.name]
            _connect(new.out(omap[port]), q)
        # exposed ports, membership, name
        for sub in self._chain(old):
            sub._in = {k: new.inp(imap[p.name]) if p.block is old else p
                       for k, p in sub._in.items()}  # fmt: skip
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

    def _exposes(self, port) -> bool:
        ports = self._in.values() if port.kind == "in" else self._out.values()
        return any(_key(p) == _key(port) for p in ports)

    def _port_name(self, block, port, kind):
        # the name under which this subsystem exposes block[port], or None
        for k, p in (self._in if kind == "in" else self._out).items():
            if p.block is block and p.name == port:
                return k
        return None

    # ports, as for a block
    def __getitem__(self, port) -> Port:
        """Return port ``port``, input or output."""
        if port in self.inputs and port in self.outputs:
            raise KeyError(f"{port!r} is both input and output; use .inp() / .out()")
        if port in self.inputs:
            return self._in[port]
        if port in self.outputs:
            return self._out[port]
        raise KeyError(f"{self.name} has no port {port!r}")

    def inp(self, port) -> Port:
        """Return input ``port``."""
        return self._in[port]

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

    def __lshift__(self, other):
        """Draw a line from ``other`` to the only input; return ``other``."""
        from .core import _connect_back

        return _connect_back(self, other)

    # inspection
    def info(self) -> str:
        """Return the summary given, or the number of blocks."""
        n = len(self.blocks)
        return self._info or f"{n} block{'s' * (n != 1)}"

    def reset(self):
        """Reset every block inside."""
        for b in self.blocks:
            b.reset()

    def draw(self, ax=None, **kw):
        """Draw the inside: members, nested subsystems as boxes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        **kw
            As for :meth:`Diagram.draw`.
        """
        from .drawing import draw

        return draw(_Inside(self), ax, **kw)

    def __repr__(self):
        """Return ``<Subsystem name in=... out=...>``."""
        return f"<Subsystem {self.name} in={self.inputs} out={self.outputs}>"


class _Inside:
    # what drawing needs from a diagram, for a subsystem's inside
    def __init__(self, sub):
        self.root, self.blocks, self._probes = sub, sub.blocks, {}
        self.fs = next((b.fs for b in self.blocks if b.fs), None)
