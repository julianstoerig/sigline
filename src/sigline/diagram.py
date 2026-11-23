"""The diagram: runs connected blocks chunk by chunk.

Per chunk:

1. take the outputs of blocks without direct feedthrough (sources, delays of
   at least one chunk),
2. ``step`` the other blocks in topological order,
3. ``update`` the blocks from 1.

A feedback loop through a block from 1 runs vectorised. Other loops, e.g.
through a delay shorter than the chunk, run sample by sample as loop
clusters (:mod:`sigline.loops`); a loop that no block delays at all raises
:class:`AlgebraicLoopError`.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import warnings
from fractions import Fraction
from math import lcm

import numpy as np

from . import core
from ._checks import check_inputs, check_output, describe
from .core import Block, Port
from .loops import (
    AlgebraicLoopError,
    LoopCluster,
    block_deps,
    cycles,
    topological,
)
from .scope import Scope

__all__ = ["AlgebraicLoopError", "Diagram"]


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


def _place(b, fs):
    # run b at fs, unless the user pinned it to another rate
    given = b._fs_given
    if b.inputs and given is not None and abs(given - fs) > 1e-9 * fs:
        raise ValueError(f"{b.name}.fs={given:g} but its inputs run at {fs:g} Hz")
    b._fs_run = fs


class Diagram:
    """All blocks connected to the given ones, run together.

    Parameters
    ----------
    *blocks : Block
        Any blocks of the graph; one is enough.
    fs : float
        Sample rate / Hz. Sources given an ``fs`` keep it; rates propagate
        along the lines (:meth:`rates`) and set ``fs`` on the blocks.
    check : {"first", True, False}, default "first"
        Check signals on the first chunk after build or :meth:`reset`, on
        every chunk, or never. Names the first block that outputs NaN/Inf or
        drops an imaginary part, and applies opt-in input checks
        (``same_shape_inputs``, ``real_inputs``). Exceptions inside a block
        always get a note naming it.
    compile_loops : {True, "python", False}, default True
        Loops through delays shorter than the chunk run sample by sample
        (:meth:`loops`), compiled with numba where every block has a kernel.
        ``"python"`` never compiles; False raises :class:`AlgebraicLoopError`
        instead.

    Attributes
    ----------
    blocks : list of Block
        In creation order.
    n : int
        Samples simulated so far, at ``fs``.
    stopped : str or None
        Why the last run ended early (:meth:`Block.request_stop`).
    overruns : int
        Chunks of the last real-time :meth:`start` that finished after their
        deadline.
    granularity : int
        Run and chunk lengths are multiples of this, so every block gets
        whole samples.

    Raises
    ------
    ValueError
        Unconnected input, mismatched input rates, or ``fs`` contradicting
        the input rate.
    TypeError
        ``resampling_ratio`` not a Fraction or int.

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

    def __init__(
        self,
        *blocks: Block,
        fs: float,
        check: bool | str = "first",
        compile_loops: bool | str = True,
    ):
        if check not in ("first", True, False):
            raise ValueError(f"check must be 'first', True or False, not {check!r}")
        self.fs = float(fs)
        self.check = check
        self._checked = False
        self._start = blocks
        self.n = 0
        self.stopped: str | None = None
        self.overruns = 0
        self._pending: queue.SimpleQueue = queue.SimpleQueue()
        self._opened = False
        self._schedules: dict[int, tuple[list[Block], list]] = {}
        self._clusters: dict[frozenset, LoopCluster] = {}
        self._probes: dict[tuple[Block, str], list[Scope]] = {}
        self.compile_loops = compile_loops
        self._build()

    def _build(self):
        # discover blocks, check inputs, assign rates; again after edits
        # a subsystem stands for the blocks inside
        start = [
            x for b in self._start for x in (b.blocks if hasattr(b, "_port") else [b])
        ]
        blocks = _discover(start)
        for b in blocks:
            missing = [p for p in b.inputs if p not in b._sources]
            if missing:
                raise ValueError(f"unconnected input(s) {missing} on {b.name}")
        self.blocks = blocks
        self._assign_rates()
        self._edits = core._EDITS[0]
        self._schedules.clear()
        self._checked = False
        for key in [k for k in self._probes if k[0] not in blocks]:
            del self._probes[key]  # its block was replaced

    def _refresh(self):
        # lines drawn or moved since the build: discover again, but never
        # under a running background thread
        if self._edits != core._EDITS[0] and not self.running:
            self._build()

    @property
    def compile_loops(self) -> bool | str:
        """How loops that need their own output within a chunk run."""
        return self._compile_loops

    @compile_loops.setter
    def compile_loops(self, value):
        if not (value is True or value is False or value == "python"):
            raise ValueError(
                f"compile_loops must be True, 'python' or False, not {value!r}"
            )
        self._compile_loops = value
        self._schedules.clear()

    # sample rates
    def _ratio(self, fs):
        return (Fraction(fs) / Fraction(self.fs)).limit_denominator(10**6)

    def _assign_rates(self):
        # rates are fractions of the diagram rate: rin[b], rout[b]
        for b in self.blocks:
            if not isinstance(b.resampling_ratio, Fraction | int):
                raise TypeError(
                    f"{b.name}.resampling_ratio must be a Fraction or int, "
                    f"not {b.resampling_ratio!r}"
                )
        rin, rout = {}, {}
        for b in self.blocks:
            if not b.inputs:
                given = b._fs_given
                r = self._ratio(given) if given else Fraction(1)
                rin[b] = rout[b] = r
        changed = True
        while changed:
            changed = False
            for b in self.blocks:
                if not b.inputs or b in rin:
                    continue
                known = [rout[p.block] for p in b._sources.values() if p.block in rout]
                if known:
                    rin[b], rout[b] = known[0], known[0] * b.resampling_ratio
                    changed = True
        for b in self.blocks:  # loops without any source run at the diagram rate
            if b not in rin:
                rin[b], rout[b] = Fraction(1), Fraction(b.resampling_ratio)
        f = Fraction(self.fs)
        for b in self.blocks:
            srcs = {k: rout[p.block] for k, p in b._sources.items()}
            if len(set(srcs.values())) > 1:
                desc = ", ".join(f"{k}: {float(r * f):g} Hz" for k, r in srcs.items())
                raise ValueError(f"sample-rate mismatch at {b.name} ({desc})")
            _place(b, float(rin[b] * f))
        self._rin, self._rout = rin, rout
        dens = [r.denominator for r in [*rin.values(), *rout.values()]]
        self.granularity = lcm(*dens) if dens else 1

    def rates(self) -> dict[str, tuple[float, float]]:
        """Return each block's input and output rate.

        Sources: own ``fs`` or the diagram's. Others: input rate times
        ``resampling_ratio``.

        Returns
        -------
        dict
            ``{name: (fs_in / Hz, fs_out / Hz)}``.
        """
        self._refresh()
        f = Fraction(self.fs)
        return {
            b.name: (float(self._rin[b] * f), float(self._rout[b] * f))
            for b in self.blocks
        }

    def _len(self, b, N):
        # (input length, output length) of b for a chunk of N diagram samples
        return int(N * self._rin[b]), int(N * self._rout[b])

    # scheduling
    def schedule(self, L: int) -> tuple[list[Block], list]:
        """Return the schedule for chunks of ``L`` samples (cached).

        Parameters
        ----------
        L : int
            Chunk length at the diagram rate.

        Returns
        -------
        state : list of Block
            Blocks without direct feedthrough.
        order : list of Block or LoopCluster
            The others, in execution order; loops that need their own
            output within the chunk as one :class:`LoopCluster` each.

        Raises
        ------
        AlgebraicLoopError
            A loop that no block delays, or any loop without a block from
            ``state`` when ``compile_loops`` is False.
        """
        if L in self._schedules:
            return self._schedules[L]
        ft = [b for b in self.blocks if b.feedthrough(self._len(b, L)[0])]
        nft = [b for b in self.blocks if not b.feedthrough(self._len(b, L)[0])]
        order = topological(ft, block_deps(ft))
        if order is None:
            if not self.compile_loops:
                names = ", ".join(b.name for b in cycles(ft)[0])
                raise AlgebraicLoopError(
                    f"loop at chunk={L} through {names} needs its own output "
                    "within the chunk: shorten the chunk to the loop's Delay, "
                    "or set compile_loops=True"
                )
            order = self._with_clusters(ft)
        self._schedules[L] = (nft, order)
        return nft, order

    def _with_clusters(self, ft):
        # each loop becomes one node; kept, with its compiled code, across
        # chunk lengths and rebuilds that leave it unchanged
        owner = {}
        for comp in cycles(ft):
            key, new = frozenset(comp), LoopCluster(comp, self)
            if not new.same_as(self._clusters.get(key)):
                self._clusters[key] = new
            for b in comp:
                owner[b] = self._clusters[key]
        nodes = list(dict.fromkeys(owner.get(b, b) for b in ft))
        deps = {n: set() for n in nodes}
        for b, srcs in block_deps(ft).items():
            me = owner.get(b, b)
            deps[me] |= {owner.get(x, x) for x in srcs} - {me}
        return topological(nodes, deps)

    def loops(self, chunk: int = 1024) -> list[LoopCluster]:
        """Return the loop clusters for chunks of ``chunk`` samples.

        Parameters
        ----------
        chunk : int, default 1024

        Returns
        -------
        list of LoopCluster
            With ``name``, ``members`` and ``mode``.
        """
        self._refresh()
        _, order = self.schedule(self._chunk(chunk))
        return [x for x in order if isinstance(x, LoopCluster)]

    # execution
    def run(self, T: float | None = None, *, n: int | None = None, chunk: int = 1024):
        """Simulate ``T`` seconds or ``n`` samples, continuing the last run.

        Ends early after a chunk in which a block called
        :meth:`Block.request_stop`.

        Parameters
        ----------
        T : float, optional
            Duration / s.
        n : int, optional
            Samples at the diagram rate; a multiple of ``granularity``.
        chunk : int, default 1024
            Samples per chunk, rounded down to a multiple of ``granularity``.
            Changes speed, never results.

        Returns
        -------
        Diagram
            ``self``.

        Raises
        ------
        ValueError
            Neither ``T`` nor ``n``, or ``n`` not a multiple of ``granularity``.
        RuntimeError
            Running in the background.
        """
        self._refresh()
        if n is None and T is None:
            raise ValueError("give T (seconds) or n (samples)")
        self._check_idle()
        n, chunk = self._length(T, n), self._chunk(chunk)
        self.open()
        self.stopped = None
        self._chunks(n, chunk)
        return self

    def _chunks(self, n, chunk, realtime=False, halt=None):
        # run n samples (None: until halted) in chunks; the one loop of run
        # and start. in the background, apply queued changes and pace to
        # the wall clock
        t0, done = time.perf_counter(), 0
        while not (halt and halt.is_set()) and (n is None or done < n):
            L = chunk if n is None else min(chunk, n - done)
            if halt:
                self._apply_pending()
            self._step(L)
            done += L
            self.n += L
            if self._take_stop_request():
                break
            if realtime:
                lag = t0 + done / self.fs - time.perf_counter()
                if lag > 0:
                    halt.wait(lag)
                else:
                    self.overruns += 1

    def _length(self, T, n):
        # validated run length in samples, None for "forever"
        g = self.granularity
        if n is None:
            if T is None:
                return None
            n = round(T * self.fs / g) * g
        if n % g:
            raise ValueError(
                f"n={n} must be a multiple of {g} (the slowest rate divides fs by {g})"
            )
        return n

    def _chunk(self, chunk):
        g = self.granularity
        return max(g, int(chunk) // g * g)

    def _take_stop_request(self) -> bool:
        probes = [p for ps in self._probes.values() for p in ps]
        req = [b for b in self.blocks + probes if getattr(b, "_stop_request", None)]
        if req:
            self.stopped = f"{req[0].name}: {req[0]._stop_request}"
            for b in req:
                b._stop_request = None
        return bool(req)

    def _check_idle(self):
        if self.running:
            raise RuntimeError(
                "the diagram is running in the background; stop() it first"
            )

    # background runs
    @property
    def running(self) -> bool:
        """``True`` while a background run started by :meth:`start` is going."""
        th = getattr(self, "_thread", None)
        return th is not None and th.is_alive()

    def start(self, T=None, *, n=None, chunk=1024, realtime=None):
        """Run in a background thread and return at once.

        Without ``T`` or ``n`` the run goes on until :meth:`stop` or a stop
        request. Read scopes with :meth:`Scope.snapshot` meanwhile.

        Parameters
        ----------
        T : float, optional
            Duration / s.
        n : int, optional
            Samples at the diagram rate.
        chunk : int, default 1024
            Samples per chunk; also the latency of parameter changes.
        realtime : bool, optional
            Pace chunks to the wall clock: chunk ``k`` is done no earlier
            than ``k * chunk / fs`` after the start. ``False`` runs as fast as
            possible, or as fast as a device delivers samples. Default:
            ``True`` unless a block has ``external_clock = True``.

        Returns
        -------
        Diagram
            ``self``.

        Raises
        ------
        RuntimeError
            Already running.
        """
        self._refresh()
        self._check_idle()
        n, chunk = self._length(T, n), self._chunk(chunk)
        if realtime is None:
            realtime = not any(getattr(b, "external_clock", False) for b in self.blocks)
        self.open()
        self.stopped, self.overruns, self._error = None, 0, None
        self._halt = threading.Event()
        self._thread = threading.Thread(
            target=self._loop,
            args=(n, chunk, realtime),
            name="sigline-run",
            daemon=True,
        )
        self._thread.start()
        return self

    def _loop(self, n, chunk, realtime):
        try:
            self._chunks(n, chunk, realtime, self._halt)
        except BaseException as err:  # re-raised in the caller's thread
            self._error = err
        finally:
            try:
                self._apply_pending()
            except BaseException as err:
                self._error = self._error or err
            finally:
                self.close()

    # resources
    def open(self):
        """Call ``open()`` on every block, once; ``run`` and ``start`` do this."""
        if not self._opened:
            for b in self.blocks:
                b.open()
            self._opened = True

    def close(self):
        """Call ``close()`` on every block; a background run does this at its end."""
        if self._opened:
            self._opened = False
            for b in self.blocks:
                b.close()

    def __enter__(self):
        """Return ``self``; ``with`` closes the blocks at the end."""
        return self

    def __exit__(self, *exc):
        """Stop a background run and close the blocks."""
        self.stop()
        self.close()

    def _apply_pending(self):
        while True:
            try:
                fn, args, kw = self._pending.get_nowait()
            except queue.Empty:
                return
            fn(*args, **kw)

    def call(self, fn, *args, **kw):
        """Call ``fn(*args, **kw)`` between chunks of a running diagram.

        Without a background run it is called at once. Use it for changes
        that must not land mid-chunk.

        Parameters
        ----------
        fn : callable
        *args, **kw
            Passed to ``fn``.
        """
        if self.running:
            self._pending.put((fn, args, kw))
        else:
            fn(*args, **kw)

    def set(self, target, **values):
        """Set block attributes between chunks, e.g. ``d.set("lo", f=2e3)``.

        Parameters
        ----------
        target : Block or str
            Block or block name.
        **values
            Attribute values.

        Raises
        ------
        AttributeError
            The block has no such attribute (likely a typo).
        """
        blk = self[target] if isinstance(target, str) else target
        for k in values:
            if not hasattr(blk, k):
                raise AttributeError(f"{blk.name} has no attribute {k!r}")

        def apply():
            for k, v in values.items():
                setattr(blk, k, v)

        self.call(apply)

    def wait(self, timeout=None):
        """Wait for a background run to end; re-raise an error from it.

        Parameters
        ----------
        timeout : float, optional
            Seconds; ``None`` waits until the run ends.

        Returns
        -------
        bool
            ``True`` if the run has ended.
        """
        th = getattr(self, "_thread", None)
        if th is None:
            return True
        try:
            th.join(timeout)
        except KeyboardInterrupt:
            self.stop()
            raise
        if th.is_alive():
            return False
        err, self._error = getattr(self, "_error", None), None
        if err is not None:
            raise err
        return True

    def stop(self):
        """End a background run after the current chunk and wait for it.

        Returns
        -------
        Diagram
            ``self``.
        """
        halt = getattr(self, "_halt", None)
        if halt is not None:
            halt.set()
            for b in self.blocks:
                b.wake()
        self.wait()
        return self

    def _step(self, N):
        nft, order = self.schedule(N)
        vals: dict[tuple[Block, str], np.ndarray] = {}
        checking = self.check is True or (self.check == "first" and not self._checked)

        def gather(b):
            # in port order, not in the order the lines were drawn
            src = b._sources
            return {k: vals[(src[k].block, src[k].name)] for k in b.inputs}

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
            if isinstance(b, LoopCluster):
                b.run(vals, N, call, compile=self.compile_loops is True)
                if checking:
                    for m, k in b.ext_out:
                        check_output(m, k, vals[(m, k)])
                continue
            Lin, Lout = self._len(b, N)
            if b.inputs:
                res = call(b, b.step, **gather(b))
            else:
                res = call(b, b.step_source, Lout)
            store(b, res, Lout)
        for b in nft:
            Lin, Lout = self._len(b, N)
            call(b, b.update, Lin if b.inputs else Lout, **gather(b))
        for key, scopes in self._probes.items():
            for sc in scopes:
                sc.step(**{sc.inputs[0]: vals[key]})
        self._checked = True

    def reset(self):
        """Reset every block and probe, ``n`` and the first-chunk check.

        Returns
        -------
        Diagram
            ``self``.

        Raises
        ------
        RuntimeError
            While running in the background.
        """
        self._check_idle()
        for b in self.blocks:
            b.reset()
        for c in self._clusters.values():
            c.reset()
        for scopes in self._probes.values():
            for sc in scopes:
                sc.reset()
        self._checked = False
        self.n = 0
        return self

    # probes
    def _resolve(self, target) -> tuple[Block, str]:
        # block, output port, subsystem, "name" or "name:port" -> (block, port)
        if hasattr(target, "_port"):
            target = target._port("out")
        if isinstance(target, Port):
            if target.kind != "out":
                raise ValueError(
                    f"{target} is an input; probe the output that drives it"
                )
            return target.block, target.name
        if isinstance(target, Block):
            if len(target.outputs) != 1:
                raise ValueError(
                    f"{target.name} has outputs {target.outputs}; use 'name:port'"
                )
            return target, target.outputs[0]
        name, _, port = str(target).partition(":")
        blk = self[name]
        return self._resolve(blk.out(port) if port else blk)

    def probe(self, target, name=None, keep=None, every=1, *, sink=None) -> Block:
        """Record or measure a signal without drawing a line to it.

        Parameters
        ----------
        target : Block or Port or Subsystem or str
            A block or subsystem with one output, an output port, ``"name"``
            or ``"name:port"``.
        name : str, optional
            Default ``"probe:<block>"``, or ``"probe:<block>:<port>"`` for a
            block with several outputs (only for the default Scope).
        keep, every : optional
            As for :class:`Scope`.
        sink : Block, optional
            Any block with one input and no outputs, e.g. a
            ``SpectrumAnalyzer``; default: a new :class:`Scope`.

        Returns
        -------
        Block
            The probe; not part of :attr:`blocks`.

        Raises
        ------
        ValueError
            Input port, ambiguous output, a block of another diagram, or a
            sink with outputs or several inputs.
        """
        self._refresh()
        blk, port = self._resolve(target)
        if blk not in self.blocks:
            raise ValueError(f"{blk.name} is not part of this diagram")
        if sink is None:
            if isinstance(target, str):
                label = target
            elif len(blk.outputs) == 1:
                label = blk.name  # the port adds nothing
            else:
                label = f"{blk.name}:{port}"
            sink = Scope(name=name or f"probe:{label}", keep=keep, every=every)
        elif len(sink.inputs) != 1 or sink.outputs:
            raise ValueError(
                f"{sink.name} needs one input and no outputs to be a probe"
            )
        _place(sink, float(self._rout[blk] * Fraction(self.fs)))
        self._probes.setdefault((blk, port), []).append(sink)
        return sink

    def remove_probe(self, sc: Block):
        """Stop recording or measuring with probe ``sc``."""
        for key, lst in list(self._probes.items()):
            if sc in lst:
                lst.remove(sc)
                if not lst:
                    del self._probes[key]

    # inspection
    @property
    def t(self) -> np.ndarray:
        """Time / s of the samples simulated so far."""
        return np.arange(self.n) / self.fs

    def summary(self, chunk: int = 1024) -> str:
        """Return the schedule for ``chunk`` and all lines as text.

        Parameters
        ----------
        chunk : int, default 1024
            Chunk length to schedule.

        Returns
        -------
        str
            ``state``/``step`` rows in execution order, then the lines.

        Raises
        ------
        AlgebraicLoopError
            Not schedulable at this chunk length.
        """
        self._refresh()
        nft, order = self.schedule(chunk)
        rows = [f"Diagram fs={self.fs:g} Hz, {len(self.blocks)} blocks, chunk={chunk}"]
        rows += [f"  state  {b.name}" for b in nft]
        rows += [
            f"  {'loop' if isinstance(b, LoopCluster) else 'step'}   {b.name}"
            for b in order
        ]
        rows += ["lines:"]
        for b in self.blocks:
            for k, p in b._sources.items():
                rows.append(f"  {p} -> {b.name}[{k!r}]")
        return "\n".join(rows)

    def __getitem__(self, name: str):
        """Return the block or subsystem called ``name``.

        Raises
        ------
        KeyError
            No such block or subsystem.
        """
        self._refresh()
        for b in self.blocks:
            if b.name == name:
                return b
        for b in self.blocks:
            sub = b._parent
            while sub is not None:
                if sub.name == name:
                    return sub
                sub = sub._parent
        raise KeyError(name)

    def scopes(self) -> list[Scope]:
        """Return the Scope blocks, in creation order."""
        return [b for b in self.blocks if isinstance(b, Scope)]

    def plot(self, *, axes=None):
        """Plot every Scope over time, one axis each.

        Parameters
        ----------
        axes : sequence of matplotlib.axes.Axes, optional
            One per Scope; default: new stacked subplots.

        Returns
        -------
        sequence of matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        sc = self.scopes()
        if axes is None:
            _, axes = plt.subplots(
                len(sc), 1, sharex=True, squeeze=False, figsize=(8, 2.2 * len(sc))
            )
            axes = axes[:, 0]
        for ax, s in zip(axes, sc, strict=True):
            s.plot(ax=ax)
            ax.set_xlabel("")
        axes[-1].set_xlabel("t / s")
        return axes

    # drawings
    def __str__(self):
        """Return :meth:`to_text`."""
        return self.to_text()

    def to_text(self, expand=False) -> str:
        """Return the diagram as text: chains, feedback, probes, blocks.

        ``expand=True`` lists the blocks inside subsystems.
        """
        self._refresh()
        from .drawing import to_text

        return to_text(self, expand)

    def draw(self, ax=None, *, save=None, info=False, expand=False, **kw):
        """Draw the diagram with matplotlib; ``d.draw("rx.pdf")`` also saves it.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or str or path, optional
            Axes to draw into, or a file to save to (short for ``save=``).
        save : str or path, optional
            File to write, any matplotlib format.
        info : bool, default False
            Also show each block's :meth:`Block.info`.
        expand : bool, default False
            Open subsystems: their blocks in a frame.
        **kw
            See :func:`sigline.drawing.draw`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self._refresh()
        from .drawing import draw

        if isinstance(ax, str | os.PathLike):
            ax, save = None, ax
        return draw(self, ax=ax, save=save, info=info, expand=expand, **kw)

    def _repr_svg_(self):
        # jupyter shows a diagram as its drawing
        import io

        import matplotlib.pyplot as plt

        fig = self.draw().figure
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def to_dot(self, info=False, expand=False) -> str:
        """Return Graphviz DOT source (``dot -Tpdf``, Typst ``diagraph``).

        ``expand=True`` opens subsystems as clusters.
        """
        self._refresh()
        from .drawing import to_dot

        return to_dot(self, info=info, expand=expand)

    def to_mermaid(self, expand=False) -> str:
        """Return a Mermaid flowchart (rendered by GitHub and many editors).

        ``expand=True`` opens subsystems as subgraphs.
        """
        self._refresh()
        from .drawing import to_mermaid

        return to_mermaid(self, expand)
