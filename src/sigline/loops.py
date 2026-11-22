"""Feedback loops run sample by sample: loop clusters.

A loop through a delay shorter than the chunk cannot run vectorised: its
blocks need each other's output of the same chunk. The diagram groups such a
loop's blocks (a strongly connected component of the blocks with direct
feedthrough) into a :class:`LoopCluster`. Everything else stays vectorised;
the cluster steps its members one sample at a time inside the chunk. Its
inputs from outside are complete arrays already, because they are outside
the loop.
"""

from __future__ import annotations

import warnings
from collections import defaultdict, deque

import numpy as np

from ._numba import HAVE_NUMBA

__all__ = ["AlgebraicLoopError", "LoopCluster", "cycles"]


class AlgebraicLoopError(RuntimeError):
    """Feedback loop in which every block needs its input of the same sample."""


def _strongly_connected(nodes, succ):
    # tarjan, iterative
    index, low, on, stack, out = {}, {}, set(), [], []
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on.add(root)
        work = [(root, iter(succ[root]))]
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is None:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    comp = []
                    while True:
                        u = stack.pop()
                        on.discard(u)
                        comp.append(u)
                        if u is v:
                            break
                    out.append(comp)
            elif w not in index:
                index[w] = low[w] = len(index)
                stack.append(w)
                on.add(w)
                work.append((w, iter(succ[w])))
            elif w in on:
                low[v] = min(low[v], index[w])
    return out


def _successors(blocks):
    # lines between the given blocks: src -> [dst, ...]
    bset, succ = set(blocks), {b: [] for b in blocks}
    for b in blocks:
        for src in {p.block for p in b._sources.values()}:
            if src in bset:
                succ[src].append(b)
    return succ


def cycles(blocks) -> list[list]:
    """Return the feedback loops among ``blocks``.

    Parameters
    ----------
    blocks : list of Block
        Only lines between these count.

    Returns
    -------
    list of list of Block
        Strongly connected components with a cycle, each sorted by creation.
    """
    succ = _successors(blocks)
    loops = [
        c for c in _strongly_connected(blocks, succ) if len(c) > 1 or c[0] in succ[c[0]]
    ]
    return [sorted(c, key=lambda b: b._id) for c in loops]


def topological(nodes, deps) -> list | None:
    """Return ``nodes`` with each after its ``deps``, or None on a cycle.

    Parameters
    ----------
    nodes : list
    deps : dict
        ``node -> set`` of nodes it needs first.
    """
    indeg = {n: len(deps[n]) for n in nodes}
    children = defaultdict(list)
    for n in nodes:
        for dep in deps[n]:
            children[dep].append(n)
    q, order = deque(n for n in nodes if indeg[n] == 0), []
    while q:
        n = q.popleft()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return order if len(order) == len(nodes) else None


def block_deps(blocks):
    """Return ``{block: set of blocks among blocks that feed it}``."""
    bset = set(blocks)
    return {
        b: {p.block for p in b._sources.values() if p.block in bset} for b in blocks
    }


def algebraic_error(loop, where) -> AlgebraicLoopError:
    """Return the error for a loop through ``loop`` that no block delays."""
    names = ", ".join(b.name for b in loop)
    return AlgebraicLoopError(
        f"algebraic loop {where} through {names}: no block in it delays its "
        "input; insert a Delay or another block without direct feedthrough"
    )


class LoopCluster:
    """The blocks of one feedback loop, run one sample at a time.

    Made by :meth:`Diagram.compile`; listed by :meth:`Diagram.loops`. When
    every member has a numba kernel, the whole per-sample loop is generated
    as one compiled function: tens of ns per sample instead of tens of us.
    The first chunk after creation or :meth:`reset` runs in Python, which
    creates the members' state and records the output types; compiled
    chunks follow. Both give the same results.

    Parameters
    ----------
    members : list of Block
        A strongly connected set of blocks with direct feedthrough at the
        chunk length.
    diagram : Diagram

    Attributes
    ----------
    name : str
        ``"loop[a, b, ...]"``.
    members : list of Block
        Sorted by creation.
    mode : str
        ``"numba"`` while running compiled, else ``"python"``.
    source : str or None
        The generated loop function, once compiled.

    Raises
    ------
    AlgebraicLoopError
        No block in the loop delays even one sample, or the loop spans
        several sample rates.
    """

    def __init__(self, members, diagram):
        self.members = sorted(members, key=lambda b: b._id)
        mset = set(self.members)
        self.name = "loop[" + ", ".join(b.name for b in self.members) + "]"
        rates = {diagram._rin[b] for b in self.members}
        rates |= {diagram._rout[b] for b in self.members}
        if len(rates) != 1:
            raise AlgebraicLoopError(
                f"{self.name} spans several sample rates; it needs a delay of "
                "at least one chunk"
            )
        self.ratio = rates.pop()
        # the schedule at one sample: state blocks, then the others in order
        self.state = [b for b in self.members if not b.feedthrough(1)]
        ft = [b for b in self.members if b.feedthrough(1)]
        self.order = topological(ft, block_deps(ft))
        if self.order is None:
            raise algebraic_error(cycles(ft)[0], "at one sample")
        # inputs: inner value or outside array; outputs: every member port,
        # so probes and outside lines find them
        self._plan = {
            b: [(k, (p.block, p.name), p.block in mset) for k, p in b._sources.items()]
            for b in self.members
        }
        self.ext_in = [
            (b, k) for b in self.members for k, _, inside in self._plan[b] if not inside
        ]
        self.ext_out = [(b, k) for b in self.members for k in b.outputs]
        self.mode = "python"
        self.source = None
        self._compiled = self._kernels = self._dtypes = self._why_python = None
        self._warm = False

    def same_as(self, other) -> bool:
        """Return whether ``other`` runs the same blocks and lines."""
        return isinstance(other, LoopCluster) and (
            self._plan,
            self.state,
            self.order,
            self.ratio,
        ) == (other._plan, other.state, other.order, other.ratio)

    def reset(self):
        """Run the next chunk in Python, so the members' state is made anew."""
        self._warm = False

    def run(self, vals, N, call, compile=False):
        """Run a chunk of ``N`` diagram samples.

        Parameters
        ----------
        vals : dict
            ``(block, port) -> array``: reads the inputs from outside, stores
            every member output.
        N : int
            Chunk length at the diagram rate.
        call : callable
            ``call(block, fn, *args, **inputs)``, the diagram's checked call.
        compile : bool, default False
            Compile with numba if possible.
        """
        L = int(N * self.ratio)
        ext = {(b, k): vals[key] for b in self.members for k, key, inside in
               self._plan[b] if not inside}  # fmt: skip
        compile = compile and HAVE_NUMBA and self._why_python is None
        if compile and self._warm and self._compiled is None:
            self._try_compile(ext)
        outs = None
        if compile and self._warm and self._compiled is not None:
            try:
                outs = self._run_compiled(ext, L)
                self.mode = "numba"
            except Exception as err:  # numba types the call at the first run
                self._give_up(err, 5)
        if outs is None:
            outs = self._run_python(ext, L, call)
            self.mode, self._warm = "python", True
            if self._dtypes is None:
                self._dtypes = {key: v.dtype for key, v in outs.items()}
        vals.update(outs)

    def _run_python(self, ext, L, call):
        rec = {key: [] for key in self.ext_out}
        for i in range(L):
            local = {}

            def inputs(b, i=i, local=local):
                return {
                    k: local[key] if inside else ext[(b, k)][i : i + 1]
                    for k, key, inside in self._plan[b]
                }

            for b in self.state:
                res = b._normalise(call(b, b.output, 1), 1)
                local.update({(b, k): v for k, v in res.items()})
            for b in self.order:
                res = b._normalise(call(b, b.step, **inputs(b)), 1)
                local.update({(b, k): v for k, v in res.items()})
            for b in self.state:
                call(b, b.update, 1, **inputs(b))
            for key in self.ext_out:
                rec[key].append(local[key])
        return {key: np.concatenate(v) if v else np.empty(0) for key, v in rec.items()}

    # compiled
    def _give_up(self, err, depth):
        # depth: frames from here to the user's run() call
        self._compiled = None
        self._why_python = str(err).splitlines()[0] or type(err).__name__
        warnings.warn(
            f"{self.name} runs in Python (slow): {self._why_python}",
            RuntimeWarning,
            stacklevel=depth,
        )

    def _try_compile(self, ext):
        from ._kernels import NotCompilable, kernel

        try:
            if any(np.ndim(v) > 1 for v in ext.values()):
                raise NotCompilable("vector signals do not compile")
            self._kernels = {b: kernel(b) for b in self.members}
            self._compiled = self._generate()
        except NotCompilable as err:
            self._give_up(err, 6)

    def _generate(self):
        import numba

        from ._kernels import JIT

        idx = {b: i for i, b in enumerate(self.members)}
        E = {key: f"E{i}" for i, key in enumerate(self.ext_in)}
        O = {key: f"O{i}" for i, key in enumerate(self.ext_out)}  # noqa: E741

        def ref(b, port):
            return f"a{idx[b]}[{b.outputs.index(port)}]"

        def args(b):
            out = []
            for k in b.inputs:
                p = b._sources[k]
                inside = p.block in idx
                out.append(ref(p.block, p.name) if inside else f"{E[(b, k)]}[i]")
            return "".join(", " + a for a in out)

        sp = "".join(f", S{idx[b]}, P{idx[b]}" for b in self.members)
        lines = [
            f"def loop(n{''.join(', ' + v for v in [*E.values(), *O.values()])}{sp}):",
            "    for i in range(n):",
        ]
        for b in self.state:
            j = idx[b]
            lines.append(f"        a{j} = out{j}(S{j}, P{j})")
        for b in self.order:
            j = idx[b]
            lines.append(f"        a{j}, S{j} = step{j}(S{j}, P{j}{args(b)})")
        for b in self.state:
            j = idx[b]
            lines.append(f"        _u{j}, S{j} = step{j}(S{j}, P{j}{args(b)})")
        for (b, port), o in O.items():
            lines.append(f"        {o}[i] = {ref(b, port)}")
        states = "".join(f"S{idx[b]}, " for b in self.members)
        lines.append(f"    return ({states})")
        self.source = "\n".join(lines)
        ns = {}
        for b, j in idx.items():
            ns[f"step{j}"] = self._kernels[b].step
            ns[f"out{j}"] = self._kernels[b].out
        exec(self.source, ns)
        return numba.njit(**JIT)(ns["loop"])

    def _run_compiled(self, ext, L):
        args = [L, *(np.ascontiguousarray(ext[key]) for key in self.ext_in)]
        outs = {key: np.empty(L, self._dtypes[key]) for key in self.ext_out}
        args += [outs[key] for key in self.ext_out]
        for b in self.members:
            args += [self._kernels[b].get_state(), self._kernels[b].params()]
        states = self._compiled(*args)
        for b, st in zip(self.members, states, strict=True):
            self._kernels[b].set_state(st)
        return outs

    def __repr__(self):
        """Return ``<LoopCluster loop[...] mode>``."""
        return f"<LoopCluster {self.name} {self.mode}>"
