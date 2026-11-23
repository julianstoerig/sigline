# Architecture

How sigline works inside. The README is the user guide.

## Modules

| module | contents |
| --- | --- |
| `sigline.core` | `Block`, `StateBlock`, `Port`, `>>`, standalone calls |
| `sigline.diagram` | `Diagram`: discovery, rates, scheduling, execution |
| `sigline.loops` | loop clusters: feedback loops run sample by sample |
| `sigline._checks` | signal checks for `Diagram._step` |
| `sigline.analog` | `mixer`, AM, FM, PM, SSB: subsystems of library blocks |
| `sigline._format` | `si()`: SI-prefixed labels |
| `sigline.blocks` | math and general blocks |
| `sigline.filters` | `FIR`, `IIR` |
| `sigline.scope` | `Scope` |
| `sigline.sources` | `Source` base and sources |
| `sigline.streams` | `SampleBuffer`, `FromQueue`, `ToQueue` |
| `sigline.live` | `watch`: live plots |
| `sigline.measure` | measurement sinks |
| `sigline.multirate` | rate-changing blocks; polyphase engine |
| `sigline.nonlinear` | `Nonlinear`, `state`: per-sample functions |
| `sigline.continuous` | `ODE`, `ContinuousLTI`: continuous-time blocks |
| `sigline._kernels` | numba kernels of blocks for compiled loops |
| `sigline._numba` | optional `njit`, `HAVE_NUMBA` |
| `sigline.digital` | bits, constellations, mapping, pulse shaping, FSK |
| `sigline.devices` | `RtlSdrSource`, `AudioIn`, `AudioOut` (optional drivers) |
| `sigline.drawing` | graph layout and renderings |
| `sigline.subsystem` | `Subsystem`: wired blocks as one block |
| `sigline.testing` | `assert_chunk_invariant` |

## Blocks

- Signals: one NumPy array per line and chunk; axis 0 is time, extra axes are
  vector signals. `Sum` and `Product` repeat one-value-per-sample signals
  over the channels (`expands_scalars`); plain NumPy broadcasting would
  align them with the channels whenever a chunk is as long as the width.
- Blocks own their state. Contract: results never depend on the chunk size.
- `Block.__init__` sets the default name (class name + per-class counter),
  `_id` (creation order, for stable orderings), then calls `reset()`. Hence
  subclasses store parameters before `super().__init__`.
- `_normalise` turns an array, tuple or dict from `step` into
  `{port: array}`, broadcasts scalars, checks lengths. `__call__` uses it
  too, so standalone and diagram use behave alike.
- `StateBlock`: `output(L)` must not change state; `update(L, **inputs)`
  advances it. Its `step` calls both, so it also works standalone and in the
  order when `feedthrough(L)` is true (a delay shorter than the chunk).

## Ports and lines

- `Port`: frozen dataclass, identity equality (usable as dict key), made on
  demand by `block["name"]`, `inp()`, `out()`.
- `a >> b` records the line at both ends: `b._sources[port]` is the driving
  output, `a._sinks` the driven inputs.
- `<<`, tuples and functions reuse `_connect`. `_Arithmetic` (base of `Block`
  and `Port`) builds `Sum`, `Product`, `Gain` and `Const` blocks for
  operators; `__array_ufunc__ = None` makes NumPy defer to them. `a + b + c`
  nests two `Sum`s: merging would change a `Sum` the user may still hold.

## Subsystems

- A `Subsystem` maps port names to inner blocks' real ports; no blocks are
  added, so diagrams discover and run the inner blocks like any others.
- Membership: blocks connected to the given ports at creation. Each gets
  `_parent` (the subsystem, or for nested ones the inner subsystem's
  `_parent`), and its name is prefixed with the subsystem's path.
- `_connect` returns `_shown_as(port)`: the outermost subsystem exposing
  the port, so `src >> sub >> sink` chains through it.
- An input may expose several inner ports; `inp()` then returns a
  `PortGroup`, and `_connect` draws one line to each of them.
- `replace` moves the old block's lines to the new one (`core._disconnect`
  removes a line) and updates the exposed ports of every subsystem up the
  chain.
- Library parts (`mixer`, `am_mod`, ...) are such functions: they wire
  named primitives and pass `fs` to each, so a diagram at another rate
  refuses them instead of running with wrong scales.
- Called alone, a subsystem connects a temporary `FromArray` per input and
  `Scope` per output, runs a `Diagram` for one chunk, removes those lines
  and restores the blocks' `fs`, so it can join a diagram at another rate
  later. Connected subsystems refuse: their diagram would include outside
  blocks.
- Drawing: `view` shows the outermost subsystem below the drawn root as
  one node; lines inside it are left out, lines to its exposed ports land
  on its named ports.

## Diagram

1. Discovery: walk `_sources` and `_sinks` from the given blocks; sort by
   `_id`. Unconnected inputs are an error. `core._EDITS` counts lines drawn
   or moved; `_refresh`, called by `run`, `start` and the renderings,
   repeats discovery and rates when it changed, except during a background
   run, and drops probes on blocks no longer there.
2. Rates: `_assign_rates` gives every block input and output rates as
   `Fraction`s of `fs`. Sources: own `fs` or 1. Others: propagate along lines,
   times `resampling_ratio`. Sourceless loops: 1. Errors for mismatched
   inputs, a contradicting `fs`, a non-`Fraction` `resampling_ratio`. `granularity` = lcm of the
   denominators; `_len(b, N)` gives a block's lengths for `N` diagram
   samples.
3. Scheduling: `schedule(N)` splits blocks by `feedthrough(L_in)`, sorts the
   feedthrough ones topologically (Kahn) over lines between them, caches
   `(state, order)` per `N`. A remaining cycle becomes loop clusters
   (below), or raises `AlgebraicLoopError` with `compile_loops=False`.
4. Execution: `_step(N)` runs `output` of state blocks, `step` of the order,
   then `update` of state blocks, each with its own lengths.
5. Probes: `_probes` maps `(block, port)` to Scopes outside `blocks`; after
   each chunk `_step` feeds them the stored values. Their `fs` is the block's
   output rate.
6. Stop requests: `request_stop` sets `_stop_request`; `run` checks after
   each chunk, stores the first reason in `stopped`, clears all, breaks.
7. Background runs: `start` runs `_loop` in a daemon thread. Real-time
   pacing waits on `_halt` until `t0 + n / fs`; a chunk finishing later
   counts as an overrun. Errors are kept in `_error` and re-raised by
   `wait`/`stop`. `run`, `reset` and `start` refuse while running.
   `set`/`call` queue changes in `_pending`, drained before each chunk.
   `open()` runs once before the first chunk; `_loop` closes at its end.
   `stop` calls every block's `wake()` so a block waiting for data returns.
   `realtime=None` paces by the wall clock unless a block has
   `external_clock`.
8. Checks: every block call goes through `call(b, fn, /, ...)`
   (positional-only: ports may be named `b` or `fn`). When checking, it
   applies opt-in input checks, raises `ComplexWarning` as `TypeError` naming
   the block, and checks outputs for NaN/Inf. Inputs were checked as their
   drivers' outputs, so the first failure is the culprit. Exceptions get one
   `add_note` naming the innermost block.

## Loop clusters

`schedule(N)` finds the cycles among the feedthrough blocks (Tarjan's
strongly connected components) and makes each a `LoopCluster`, one node of
the topological order. Clusters are kept per member set in `_clusters`
across chunk lengths and dropped on rebuild.

A cluster schedules its members at chunk length 1: blocks without
feedthrough at 1 (`state`) give `output(1)` first, the others `step` in
topological order, then `state` blocks `update(1)`. No such order means no
block delays even one sample: `AlgebraicLoopError`. Inputs from outside are
whole arrays already, indexed per sample. Every member output goes to
`vals`, so lines out of the loop and probes find it. Members must share one
sample rate. Calls go through the diagram's `call`, so checks and error
notes work per sample.

Compiled clusters: `_kernels.kernel(b)` gives a numba `step(S, P, *u) ->
(outputs, S)` per block (plus `out(S, P)` for state blocks) and how to move
state and parameters between block and kernel; kernels are registered per
type with `kernel_for`. The first chunk after creation or `reset` runs in
Python: it makes lazily created state (`Delay.buf`, `FIR.hist`, `IIR.zi`)
and records output dtypes. From the second chunk, `_generate` writes the
per-sample schedule as one function (`source`) and compiles it. Around
every chunk, state is copied out of the blocks and written back, so a
failing call leaves blocks untouched, and `reset`, inspection and parameter
changes keep working. Without a kernel, with vector signals, or when numba
cannot type a call, the cluster warns once and stays in Python. Clusters
survive rebuilds that leave their blocks and lines unchanged (`same_as`), so
lines drawn elsewhere do not force a recompile.

## Nonlinear blocks

`Nonlinear` reads its ports and parameters from `f`'s signature and
generates a loop over the chunk (`_loop_source`) that calls `f` per sample
and threads the state. With numba, `f` and the loop are compiled
(`_build`, cached per function and shape); a numba error on the first call
switches to the same loop in Python, with a warning unless `jit=True`.
Output dtypes come from `dtype=` or from one call of `f` on a copy of the
state; ints become floats. `_copy_state` deep-copies state (keeping
namedtuples) so neither that call nor `reset()` changes the initial state.
Its loop kernel calls the compiled `f` directly.

## Continuous blocks

`ODE` is a `StateBlock` that integrates per sample: `_y(u)` samples the
output at `t_k = k / fs`, `_advance(u)` integrates to `t_k + 1/fs` with the
input held (`solve_ivp`, or fixed-step RK4). `feedthrough(L)` is False only
at `L == 1` and only if `g` ignores the input, so a loop through it becomes
a loop cluster with the ODE as a state block. `output(L)` is valid only for
`L == 1`; longer chunks go through `step`. The RK4 loop kernel inlines `f`
and `g` compiled with numba; `solve_ivp` methods have no kernel.

`ContinuousLTI` is an `ODE` with `f`, `g` from `A, B, C, D` (so any
`solve_ivp` method works). With `method="zoh"` it replaces integration by
the exact discretisation `Ad, Bd = expm([[A, B], [0, 0]] / fs)`, made once
per `fs` (`_discretise`), and runs whole chunks through `_zoh`: explicit
loops compiled with numba, or matrix products per sample without it. Its
loop kernel covers single-input single-output systems.

## Drawing

`drawing.view(d)` turns blocks into `Node`s and lines into `Edge`s.
`_layout` marks feedback lines (back edges of a depth-first search from the
sources), assigns columns by the longest path over the other lines, moves
sources next to their first consumer, and gives a line that skips columns
a `_Lane` in each of them, so it gets a slot of its own and never runs
through a box. Six barycenter sweeps order each column, once with lanes
first and once last on ties; the order with the fewest crossings wins.
Every rendering uses this one layout. `Diagram._repr_svg_`
draws with matplotlib, so Jupyter shows diagrams as graphs.
