"""Blocks, ports and the ``>>`` lines between them.

A block has named ports, owns its state and processes one chunk per
:meth:`Block.step`. It also works alone as a function: ``block(x)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = ["Block", "Port", "StateBlock"]


@dataclass(frozen=True, eq=False)
class Port:
    """Named input or output of a block, created by ``block["name"]``.

    Parameters
    ----------
    block : Block
        Owner.
    name : str
        One of ``block.inputs`` or ``block.outputs``.
    kind : {"in", "out"}
        Direction.
    """

    block: Block
    name: str
    kind: str

    def __rshift__(self, other):
        """Draw a line from this output to ``other``."""
        return _connect(self, other)

    def __lshift__(self, other):
        """Draw a line from ``other`` to this input; return ``other``'s block."""
        return _connect_back(self, other)

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


def _connect_back(dst, src):
    # dst << src: the same line as src >> dst, but returns the source
    if isinstance(src, tuple | list):
        _fan_in(src, dst)
        return tuple(_as_out_port(s).block for s in src)
    s = _as_out_port(src)
    _connect(s, dst)
    return s.block


def _fan_in(srcs, dst):
    # (a, b) >> blk: one line per input, in order
    if not isinstance(dst, Block):
        raise TypeError(f"a tuple of signals needs a block, not {dst!r}")
    if len(srcs) != len(dst.inputs):
        raise ValueError(
            f"{len(srcs)} signals for the {len(dst.inputs)} inputs {dst.inputs} "
            f"of {dst.name}"
        )
    for s, port in zip(srcs, dst.inputs, strict=True):
        _connect(s, dst.inp(port))
    return dst


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

    Store parameters as attributes, call ``super().__init__(**kw)`` (it calls
    :meth:`reset`), implement :meth:`step`. ``inputs`` and ``outputs`` name the
    ports (default ``("x",)`` and ``("y",)``); ``inputs = ()`` makes a source.

    ``resampling_ratio`` is output rate / input rate as a
    :class:`~fractions.Fraction`: 1 by default, ``Fraction(4)`` for an
    upsampler by 4, ``Fraction(1, 2)`` for a downsampler by 2.

    Parameters
    ----------
    name : str, optional
        Default: class name plus a counter, e.g. ``"Gain1"``.
    fs : float, optional
        Sample rate / Hz. A diagram sets it to the input rate; a source given
        ``fs`` runs at that rate.

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
    resampling_ratio: Fraction = Fraction(1)
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
            One array per input port; first axis is time.

        Returns
        -------
        numpy.ndarray or tuple or dict
            An array, a tuple in ``outputs`` order or ``{port: array}``.
            Scalars are broadcast over the chunk.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement step()")

    def step_source(self, L):
        """Produce ``L`` samples (sources). Default: ``step()``.

        Parameters
        ----------
        L : int
            Chunk length.

        Returns
        -------
        numpy.ndarray or tuple or dict
            As for :meth:`step`.
        """
        return self.step()

    def reset(self):
        """Initialise the state; called by ``__init__`` and ``Diagram.reset``."""

    def info(self) -> str:
        """Return a one-line parameter summary for drawings; override it."""
        return ""

    def open(self):
        """Acquire external resources (devices, files); called before running."""

    def close(self):
        """Release what :meth:`open` acquired; called by ``Diagram.close``."""

    def wake(self):
        """Interrupt a wait inside ``step``; called by ``Diagram.stop``."""

    def request_stop(self, reason: str = "stop requested"):
        """Stop the diagram after the current chunk.

        ``Diagram.stopped`` becomes ``"<name>: <reason>"``; the next run
        continues normally.

        Parameters
        ----------
        reason : str, default "stop requested"
            Text for ``Diagram.stopped``.
        """
        self._stop_request = reason

    def feedthrough(self, L: int) -> bool:
        """Tell whether this chunk's outputs depend on this chunk's inputs.

        Blocks returning ``False`` break feedback loops and implement
        ``output`` and ``update`` like :class:`StateBlock`.

        Parameters
        ----------
        L : int
            Input chunk length.

        Returns
        -------
        bool
            ``True`` here.
        """
        return True

    # lines
    def __getitem__(self, port: str) -> Port:
        """Return the port named ``port``.

        Raises
        ------
        KeyError
            No such port, or the name is both input and output (use
            :meth:`inp` / :meth:`out`).
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
        """Return the input port named ``port``.

        Raises
        ------
        KeyError
            No such input.
        """
        if port not in self.inputs:
            raise KeyError(f"{self.name} has no input {port!r} (inputs={self.inputs})")
        return Port(self, port, "in")

    def out(self, port: str) -> Port:
        """Return the output port named ``port``.

        Raises
        ------
        KeyError
            No such output.
        """
        if port not in self.outputs:
            raise KeyError(
                f"{self.name} has no output {port!r} (outputs={self.outputs})"
            )
        return Port(self, port, "out")

    def __rshift__(self, other):
        """Draw a line from the only output to ``other``.

        ``other`` is an input port or a block with one input. An input has one
        driver; an output may drive many inputs.

        Returns
        -------
        Block
            The destination, so ``a >> b >> c["x"]`` chains.

        Raises
        ------
        ValueError
            Ambiguous port, wrong direction or input already driven.
        """
        return _connect(self, other)

    def __lshift__(self, other):
        """Draw a line from ``other`` to this block's only input.

        ``b << a`` is ``a >> b``, but returns the source, so
        ``c << b << a`` chains right to left.

        A tuple ``b << (x, y)`` feeds the inputs in order.

        Returns
        -------
        Block or tuple of Block
            ``other``'s block(s).
        """
        return _connect_back(self, other)

    def __rrshift__(self, other):
        """Feed a tuple of signals to the inputs in order: ``(a, b) >> blk``.

        Returns
        -------
        Block
            ``self``.

        Raises
        ------
        ValueError
            Not one signal per input.
        """
        if isinstance(other, tuple | list):
            return _fan_in(other, self)
        return NotImplemented

    # standalone use as a plain function: fir(x), mix(a, b), src(L)
    def __call__(self, *args, **kwargs):
        """Process one chunk outside a diagram; state carries over.

        Sources take the length, ``src(L)``; other blocks take their inputs
        positionally or by port name.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            One array, or a tuple for several outputs.
        """
        if not self.inputs and len(args) == 1 and not kwargs:
            L = int(args[0])
            out = self._normalise(self.step_source(L), L)
        else:
            kwargs.update(zip(self.inputs, args, strict=False))
            L = len(next(iter(kwargs.values())))
            out = self._normalise(self.step(**kwargs), int(L * self.resampling_ratio))
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
    """Block whose output depends only on its state.

    Implement :meth:`output` (must not change state) and :meth:`update`. A
    diagram takes all outputs first and updates last, so state blocks break
    feedback loops.

    Examples
    --------
    Delay by ``n`` samples (chunks of at most ``n``):

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
        """Return ``False``."""
        return False

    def output(self, L):
        """Return the next ``L`` samples without changing the state.

        Parameters
        ----------
        L : int
            Output chunk length.

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
            Input chunk length (output length for sources).
        **inputs : numpy.ndarray
            This chunk's inputs.
        """

    def step(self, **inputs):
        """Return :meth:`output`, then :meth:`update`."""
        L = len(next(iter(inputs.values())))
        y = self.output(L)
        self.update(L, **inputs)
        return y

    def step_source(self, L):
        """Return :meth:`output`, then :meth:`update` (sources)."""
        y = self.output(L)
        self.update(L)
        return y
