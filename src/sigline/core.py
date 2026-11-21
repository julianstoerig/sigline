"""Blocks, ports and the ``>>`` wiring between them.

A block is an object with named input and output ports. It owns its state and
processes one chunk of samples per call of :meth:`Block.step`. Blocks are wired
with ``>>`` into a graph that a diagram runs, and every block also works on its
own as a plain function: ``block(x)`` processes one chunk.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

__all__ = ["Block", "Port", "StateBlock"]


@dataclass(frozen=True, eq=False)
class Port:
    """One named input or output of a block.

    Ports are created on demand by ``block["name"]``, :meth:`Block.inp` and
    :meth:`Block.out`; they are lightweight handles used for wiring.

    Parameters
    ----------
    block : Block
        The block the port belongs to.
    name : str
        Port name, one of ``block.inputs`` or ``block.outputs``.
    kind : {"in", "out"}
        Whether the port is an input or an output.
    """

    block: Block
    name: str
    kind: str

    def __rshift__(self, other):
        """Connect this output port to ``other`` (see :meth:`Block.__rshift__`)."""
        return _connect(self, other)

    def __repr__(self):
        """Return ``block_name['port']``."""
        return f"{self.block.name}[{self.name!r}]"


def _as_out_port(obj) -> Port:
    if isinstance(obj, Port):
        if obj.kind != "out":
            raise ValueError(f"{obj} is an input port; cannot be a connection source")
        return obj
    if isinstance(obj, Block):
        if len(obj.outputs) != 1:
            raise ValueError(
                f"{obj.name} has outputs {obj.outputs}; pick one with block['name']"
            )
        return obj.out(obj.outputs[0])
    raise TypeError(f"cannot connect from {obj!r}")


def _as_in_port(obj) -> Port:
    if isinstance(obj, Port):
        if obj.kind != "in":
            raise ValueError(f"{obj} is an output port; cannot be a connection target")
        return obj
    if isinstance(obj, Block):
        if len(obj.inputs) == 1:
            return obj.inp(obj.inputs[0])
        if not obj.inputs:
            raise ValueError(f"{obj.name} has no inputs")
        if all(p in obj._sources for p in obj.inputs):
            raise ValueError(f"all inputs of {obj.name} are connected")
        raise ValueError(
            f"{obj.name} has inputs {obj.inputs}; pick one with block['name']"
        )
    raise TypeError(f"cannot connect to {obj!r}")


def _connect(src, dst):
    s, d = _as_out_port(src), _as_in_port(dst)
    if d.name in d.block._sources:
        raise ValueError(f"{d} is already driven by {d.block._sources[d.name]}")
    d.block._sources[d.name] = s
    s.block._sinks.append(d)
    # return the destination so chains read left to right: a >> b >> c
    return d.block


class Block:
    """Base class of all blocks.

    Subclass it, store the parameters as attributes, call
    ``super().__init__(**kw)`` (which calls :meth:`reset`) and implement
    :meth:`step`. The class attributes ``inputs`` and ``outputs`` name the
    ports; the defaults are one input ``x`` and one output ``y``. A block with
    ``inputs = ()`` is a source.

    Parameters
    ----------
    name : str, optional
        Name shown in messages and used to look the block up in a diagram.
        Defaults to the class name plus a counter, e.g. ``"Gain1"``.
    fs : float, optional
        Sample rate in Hz. A diagram sets it when it is built, so blocks can
        read ``self.fs`` from their first ``step`` on.

    Examples
    --------
    >>> import numpy as np
    >>> from sigline import Block
    >>> class Scale(Block):
    ...     def __init__(self, k, **kw):
    ...         self.k = k
    ...         super().__init__(**kw)
    ...
    ...     def step(self, x):
    ...         return self.k * x
    >>> Scale(3)(np.array([1.0, 2.0]))
    array([3., 6.])
    """

    inputs: tuple[str, ...] = ("x",)
    outputs: tuple[str, ...] = ("y",)
    _counter: defaultdict[str, int] = defaultdict(int)

    def __init__(self, *, name: str | None = None, fs: float | None = None):
        cls = type(self).__name__
        Block._counter[cls] += 1
        Block._counter["__all__"] += 1
        self._id = Block._counter["__all__"]  # creation order, for stable orderings
        self.name = name or f"{cls}{Block._counter[cls]}"
        self.fs = fs
        self._sources: dict[str, Port] = {}  # input name -> driving output port
        self._sinks: list[Port] = []  # input ports this block drives
        self.reset()

    # user api
    def step(self, **inputs):
        """Process one chunk.

        Parameters
        ----------
        **inputs : numpy.ndarray
            One array per input port, keyed by port name; the first axis is time.

        Returns
        -------
        numpy.ndarray or tuple or dict
            One array for a single output, a tuple in ``outputs`` order, or a
            dict ``{port: array}``. A scalar is broadcast over the chunk.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement step()")

    def step_source(self, L):
        """Produce one chunk of ``L`` samples (blocks without inputs).

        The default calls ``step()``; override it when the source needs ``L``.

        Parameters
        ----------
        L : int
            Chunk length in samples.

        Returns
        -------
        numpy.ndarray or tuple or dict
            As for :meth:`step`.
        """
        return self.step()

    def reset(self):
        """(Re)initialise the state; called by ``__init__`` and on diagram reset."""

    def feedthrough(self, L: int) -> bool:
        """Tell whether this chunk's outputs depend on this chunk's inputs.

        The scheduler orders blocks over feedthrough connections only, so a
        block that returns ``False`` breaks feedback loops. Such a block
        implements ``output`` and ``update`` as :class:`StateBlock` does.

        Parameters
        ----------
        L : int
            Input chunk length in samples.

        Returns
        -------
        bool
            ``True`` for ordinary blocks.
        """
        return True

    # wiring
    def __getitem__(self, port: str) -> Port:
        """Return the input or output port named ``port``.

        Raises
        ------
        KeyError
            If there is no such port, or the name is both an input and an output
            (then use :meth:`inp` or :meth:`out`).
        """
        if port in self.inputs and port in self.outputs:
            raise KeyError(
                f"{self.name}: {port!r} is both an input and an output; "
                f"use .inp({port!r}) or .out({port!r})"
            )
        if port in self.inputs:
            return Port(self, port, "in")
        if port in self.outputs:
            return Port(self, port, "out")
        raise KeyError(
            f"{self.name} has no port {port!r} "
            f"(inputs={self.inputs}, outputs={self.outputs})"
        )

    def inp(self, port: str) -> Port:
        """Return the input port named ``port`` (unambiguous ``block[port]``).

        Raises
        ------
        KeyError
            If the block has no such input.
        """
        if port not in self.inputs:
            raise KeyError(f"{self.name} has no input {port!r} (inputs={self.inputs})")
        return Port(self, port, "in")

    def out(self, port: str) -> Port:
        """Return the output port named ``port`` (unambiguous ``block[port]``).

        Raises
        ------
        KeyError
            If the block has no such output.
        """
        if port not in self.outputs:
            raise KeyError(
                f"{self.name} has no output {port!r} (outputs={self.outputs})"
            )
        return Port(self, port, "out")

    def __rshift__(self, other):
        """Connect this block's only output to ``other``.

        ``other`` is an input port (``b["x"]``) or a block with exactly one
        input. Each input has exactly one driver; an output can drive many
        inputs.

        Returns
        -------
        Block
            The destination block, so chains read left to right:
            ``a >> b >> c["x"]``.

        Raises
        ------
        ValueError
            If a port is ambiguous, has the wrong direction or is already driven.
        """
        return _connect(self, other)

    # standalone use as a plain function: fir(x), mix(a, b), src(L)
    def __call__(self, *args, **kwargs):
        """Process one chunk outside a diagram.

        Sources take the chunk length, ``src(L)``; other blocks take their
        inputs positionally (in ``inputs`` order) or by port name. State
        carries over between calls, exactly as between diagram chunks.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The output, or a tuple in ``outputs`` order for several outputs.
        """
        if not self.inputs and len(args) == 1 and not kwargs:
            L = int(args[0])
            out = self._normalise(self.step_source(L), L)
        else:
            kwargs.update(zip(self.inputs, args, strict=False))
            L = len(next(iter(kwargs.values())))
            out = self._normalise(self.step(**kwargs), L)
        vals = tuple(out[k] for k in self.outputs)
        return vals[0] if len(vals) == 1 else vals

    # helpers
    def _normalise(self, res, L) -> dict:
        if isinstance(res, dict):
            out = res
        elif isinstance(res, tuple):
            out = dict(zip(self.outputs, res, strict=True))
        else:
            out = {self.outputs[0]: res}
        for k, v in out.items():
            v = np.asarray(v)
            if v.ndim == 0:
                v = np.full(L, v)
            if v.shape[0] != L:
                raise ValueError(
                    f"{self.name}[{k!r}] returned {v.shape[0]} samples, expected {L}"
                )
            out[k] = v
        return out

    def __repr__(self):
        """Return ``<ClassName name>``."""
        return f"<{type(self).__name__} {self.name}>"


class StateBlock(Block):
    """Block whose output depends only on its state (no direct feedthrough).

    Implement :meth:`output`, which must not change the state, and
    :meth:`update`, which advances the state by one chunk. A diagram asks all
    state blocks for their outputs first and updates them once their inputs
    exist, so they break feedback loops. Sources with internal state and
    delays are typical state blocks.

    Examples
    --------
    A delay by one chunk of ``n`` samples:

    >>> import numpy as np
    >>> from sigline import StateBlock
    >>> class Delay(StateBlock):
    ...     def __init__(self, n, **kw):
    ...         self.n = n
    ...         super().__init__(**kw)
    ...
    ...     def reset(self):
    ...         self.buf = np.zeros(self.n)
    ...
    ...     def output(self, L):
    ...         return self.buf[:L]
    ...
    ...     def update(self, L, x):
    ...         self.buf = np.concatenate([self.buf, x])[L:]
    >>> d = Delay(2)
    >>> d(np.array([1.0, 2.0])), d(np.array([3.0, 4.0]))
    (array([0., 0.]), array([1., 2.]))
    """

    def feedthrough(self, L):
        """Return ``False``: the output never depends on the current input."""
        return False

    def output(self, L):
        """Return the next ``L`` output samples without changing the state.

        Parameters
        ----------
        L : int
            Output chunk length in samples.

        Returns
        -------
        numpy.ndarray or tuple or dict
            As for :meth:`Block.step`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement output()")

    def update(self, L, **inputs):
        """Advance the state by one chunk.

        Parameters
        ----------
        L : int
            Chunk length in samples (the input length; for a source, the
            output length).
        **inputs : numpy.ndarray
            This chunk's inputs, keyed by port name.
        """

    def step(self, **inputs):
        """Return :meth:`output` for this chunk, then :meth:`update` the state."""
        L = len(next(iter(inputs.values())))
        y = self.output(L)
        self.update(L, **inputs)
        return y

    def step_source(self, L):
        """Return :meth:`output` for ``L`` samples, then :meth:`update` (sources)."""
        y = self.output(L)
        self.update(L)
        return y
