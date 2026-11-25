# Architecture

How sigline works inside. The README is the user guide.

## Design in brief

- A line carries one NumPy array per chunk: axis 0 is time, extra axes
  make a vector signal. Blocks keep their state between chunks, so the
  chunk length changes speed, never results.
- Time is implicit. `Diagram.n` counts samples at the diagram rate, each
  block counts its own (`Source.k`, `ODE.k`); `step` gets no time.
- Every block runs at a fixed rate, a `Fraction` of the diagram's `fs`,
  assigned at build from the sources and every `resampling_ratio`. It fixes
  each block's output length per chunk; `_normalise` rejects any other.
- One static schedule per chunk length, cached (`Diagram.schedule`): blocks
  without direct feedthrough give `output`, the others `step` in
  topological order, then the first group `update`s.
- Feedthrough is declared per chunk length by `feedthrough(L)`, not by
  class: never for a `StateBlock`, for `Delay(n)` only when `n < L`, for an
  `ODE` only when `L > 1` or its output map reads the input.
- A loop through a block without feedthrough at the chunk length runs
  vectorised. Other loops become loop clusters, run sample by sample inside
  the chunk, compiled with numba when every member has a kernel. A loop no
  block delays by even one sample raises `AlgebraicLoopError`.
- numba is optional (`HAVE_NUMBA`; `SIGLINE_NO_NUMBA=1` turns it off). It
  compiles loop clusters, `FuncSingle` and the polyphase, ZOH and Markov
  kernels; the NumPy paths give the same results.
- Continuous blocks are sampled: each `ODE` integrates from sample to
  sample with its inputs held. There is no global solver.
- Subsystems exist only while building: they map port names to inner
  ports, and a diagram runs the inner blocks like any others.
- A diagram is every block reachable over lines from the blocks given.
  Lines drawn or moved between runs rebuild it at the next run.
- `run` steps chunks in the caller's thread; `start` steps them in one
  background thread, paced to the wall clock or by a device that waits for
  data. Pacing is soft: late chunks are counted, not prevented. Changes
  from other threads land between chunks (`set`, `call`).

## Modules

| module | contents |
| --- | --- |
| `sigline` | public names, `__version__` |
| `sigline.core` | `Block`, `StateBlock`, `Port`, `PortGroup`, `>>`, operators, standalone calls |
| `sigline.diagram` | `Diagram`: discovery, rates, scheduling, execution, probes, background runs |
| `sigline.loops` | loop clusters: feedback loops run sample by sample; `AlgebraicLoopError` |
| `sigline._checks` | signal checks for `Diagram._step` |
| `sigline.analog` | `mixer`, AM, FM, PM, SSB: subsystems of library blocks |
| `sigline.digital` | bits, constellations, `WordsToBits`, `BitsToWords`, `FSKDemod`; mapper, pulse shaping, FSK as subsystems |
| `sigline.channels` | noise and channel models as subsystems |
| `sigline._format` | `si()`: SI-prefixed labels |
| `sigline.blocks` | `Gain`, `Sum`, `Product`, `FuncChunk`, `Delay`, `Unwrap` |
| `sigline.filters` | `FIR`, `IIR` |
| `sigline.scope` | `Scope`: recording, plots, Welch PSD |
| `sigline.sources` | `Source` base and sources |
| `sigline.streams` | `SampleBuffer` (thread-safe FIFO), `FromQueue`, `ToQueue` |
| `sigline.live` | `watch`: live plots of a background run |
| `sigline.measure` | measurement sinks; each takes its lock for updates and readings (`_Sink`), spectra share their framing (`_Frames`) |
| `sigline.multirate` | rate-changing blocks; polyphase engine |
| `sigline.func_single` | `FuncSingle`, `state`: per-sample functions |
| `sigline.continuous` | `ODE`, `ContinuousLTI`: continuous-time blocks |
| `sigline._kernels` | numba kernels of blocks for compiled loops |
| `sigline._numba` | optional `njit`, `HAVE_NUMBA`, `callable_in_numba` |
| `sigline.devices` | `RtlSdrSource`, `AudioIn`, `AudioOut` (optional drivers) |
| `sigline.drawing` | graph layout and renderings |
| `sigline.subsystem` | `Subsystem`: wired blocks as one block |
| `sigline.testing` | `assert_chunk_invariant` |

## Blocks

- Signals: one NumPy array per line and chunk; axis 0 is time, extra axes are
  vector signals. `Sum` and `Product` repeat one-value-per-sample signals
  over the channels (`blocks._expand`; `expands_scalars` lets the width
  check pass them); plain NumPy broadcasting would align them with the
  channels whenever a chunk is as long as the width.
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
- Any block may return False from `feedthrough(L)` and implement `output`
  and `update`; the diagram then treats it as a state block for chunks of
  that length.
- Hooks besides `step`: `reset`, `info` (drawings), `open` and `close`
  (devices), `wake` (end a wait), `request_stop`, and the attribute
  `external_clock` (`Diagram.start`).
- `Source` is a `StateBlock` without inputs: `generate(n)` maps absolute
  sample indices to samples, so a counter `k` is its only state. Random
  sources (`Noise`, `RandomBits`, `RandomPulses`, `MarkovChain`) draw one
  chunk per `update` from one generator and keep it in `_next`, so
  `output` stays repeatable; complex noise comes as (I, Q) pairs, so the
  stream does not depend on chunking.
- `FuncChunk` and `FuncSingle` take inputs and parameters from a function's
  signature (`core._arguments`, `core._parameters`).

## Ports and lines

- `Port`: frozen dataclass, identity equality (usable as dict key), made on
  demand by `block["name"]`, `inp()`, `out()`.
- `a >> b` records the line at both ends: `b._sources[port]` is the driving
  output, `a._sinks` the driven inputs.
- `<<`, tuples and functions reuse `_connect`. `_Arithmetic` (base of
  `Block`, `Port` and `Subsystem`) builds `Sum`, `Product`, `Gain` and
  `Const` blocks for operators; `__array_ufunc__ = None` makes NumPy defer
  to them. `a + b + c` nests two `Sum`s: merging would change a `Sum` the
  user may still hold.
- An input has one driver: `_connect` raises for a second. Every line drawn
  or removed counts in `core._EDITS`.

## Subsystems

- A `Subsystem` maps port names to inner blocks' real ports; no blocks are
  added, so diagrams discover and run the inner blocks like any others.
- Membership: blocks connected to the given ports at creation, not counting
  lines that leave through an exposed output (`_inside`). Each gets
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
  named primitives and pass `fs` to each. Given, it pins them, so a diagram
  at another rate refuses them instead of running with wrong scales. It is
  required where a coefficient depends on it (`fm_mod`, `fm_demod`,
  `phase_noise`, `fsk_mod`, `modulated_interferer`), optional elsewhere.
- Called alone, a subsystem connects a temporary `FromArray` per input and
  `Scope` per output, runs a `Diagram` for one chunk and removes those
  lines. The rate it ran at does not pin its blocks, so it can join a
  diagram at another rate later. Connected subsystems refuse: their
  diagram would include outside blocks.
- Drawing: `view` shows the outermost subsystem below the drawn root as
  one node; lines inside it are left out, lines to its exposed ports land
  on its named ports. With `expand=True` every block is a node and
  `Node.groups` lists the subsystems holding it.

## Diagram

1. Discovery: walk `_sources` and `_sinks` from the given blocks (a
   subsystem stands for its blocks); sort by `_id`. Unconnected inputs are
   an error. `core._EDITS` counts lines drawn or moved; `_refresh`, called
   by `run`, `start`, `probe`, the inspection methods and the renderings,
   repeats discovery and rates when it changed, except during a background
   run, and drops probes on blocks no longer there. The counter is global,
   so an edit anywhere rebuilds every diagram at its next refresh.
2. Rates: `_assign_rates` gives every block input and output rates as
   `Fraction`s of `fs`. Sources: own `fs` or 1. Others: propagate along lines,
   times `resampling_ratio`, and back against them (`spread`), so a source
   inside a subsystem gets its rate from the signal it meets: such sources
   get the fallback 1 only after the top-level ones. Sourceless loops: 1. Errors for mismatched
   inputs, a contradicting `fs`, a non-`Fraction` `resampling_ratio`. `granularity` = lcm of the
   denominators; `_len(b, N)` gives a block's lengths for `N` diagram
   samples. A block keeps two rates: `_fs_given` from the user (pins it,
   checked here) and `_fs_run`, set here; `fs` reads the first, else the
   second. So a block moves between diagrams at different rates unless
   pinned. Setting `blk.fs` counts as an edit and triggers a rebuild.
   `_ratio` turns a source's own `fs` into a `Fraction` with a denominator
   of at most 10^6.
3. Scheduling: `schedule(N)` splits blocks by `feedthrough(L_in)`, sorts the
   feedthrough ones topologically (Kahn) over lines between them, caches
   `(state, order)` per `N`, until a rebuild or a change of
   `compile_loops`. A remaining cycle becomes loop clusters (below), or
   raises `AlgebraicLoopError` with `compile_loops=False`.
4. Execution: `_step(N)` runs `output` of state blocks, `step` of the order,
   then `update` of state blocks, each with its own lengths.
5. Probes: `_probes` maps `(block, port)` to Scopes outside `blocks`; after
   each chunk `_step` feeds them the stored values. Their `fs` is the block's
   output rate.
6. Stop requests: `request_stop` sets `_stop_request`; `_chunks`, the loop
   of `run` and `start`, checks blocks and probes after each chunk, stores
   the first reason in `stopped`, clears all, breaks.
7. Background runs: `start` runs `_loop` in a daemon thread. Real-time
   pacing waits on `_halt` until `t0 + done / (fs * speed)`, `done` being
   the samples run so far; a chunk finishing later counts as an overrun.
   Paced runs default to chunks of `fs * speed / 50` samples, about 20 ms
   of wall time. Errors are kept in `_error` and re-raised by
   `wait`/`stop`. `run`, `reset` and `start` refuse while running.
   `set`/`call` queue changes in `_pending`, drained before each chunk and
   once at the end. Lines drawn meanwhile wait for the next `start`.
   `open()` runs once before the first chunk; `_loop` closes at its end,
   while after `run` blocks stay open until `close()` or the end of a
   `with` block. `stop` calls every block's `wake()` so a block waiting
   for data returns. `realtime=None` paces by the wall clock unless a
   block has `external_clock`. `watch` starts the run and hands it to
   `live.watch`.
8. Checks: every block call goes through `call(b, fn, /, ...)`
   (positional-only: ports may be named `b` or `fn`). It checks on the
   first chunk after a build or `reset` (`check="first"`), on every chunk
   (`True`) or never (`False`). When checking, it applies opt-in input
   checks, raises `ComplexWarning` as `TypeError` naming the block, and
   checks outputs for NaN/Inf. Inputs were checked as their
   drivers' outputs, so the first failure is the culprit. Exceptions get one
   `add_note` naming the innermost block.

## Loop clusters

`schedule(N)` finds the cycles among the feedthrough blocks (Tarjan's
strongly connected components) and makes each a `LoopCluster`, one node of
the topological order. Clusters are kept per member set in `_clusters`,
across chunk lengths and across rebuilds that leave them unchanged
(below).

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
type with `kernel_for` and looked up by exact type, so a subclass has none.
Kernels exist for `Gain` (scalar gain), `Sum`, `Product`, `FuncChunk`,
`Delay`, `FIR`, `IIR`, `FuncSingle` (compiled), `ODE` (RK4, one output)
and `ContinuousLTI` (ZOH, single input and output). The first chunk after
creation or `reset` runs in Python: it makes lazily created state
(`Delay.buf`, `FIR.hist`, `IIR.zi`) and records output dtypes. From the
second chunk, `_generate` writes the per-sample schedule as one function
(`source`) and compiles it, once per process: generated code is not in
numba's cache. Around every chunk, state is copied out of the blocks and
written back, so a
failing call leaves blocks untouched, and `reset`, inspection and parameter
changes keep working. Without a kernel, with vector signals, or when numba
cannot type a call, the cluster warns once and stays in Python. Clusters
survive rebuilds that leave their blocks and lines unchanged (`same_as`), so
lines drawn elsewhere do not force a recompile.

## Per-sample functions: FuncSingle

`FuncSingle` reads its ports and parameters from `f`'s signature and
generates a loop over the chunk (`_loop_source`) that calls `f` per sample
and threads the state. With numba, `f` and the loop are compiled
(`_build`, cached per function and shape); a numba error on the first call
switches to the same loop in Python with a warning, or raises with
`jit=True`. Output dtypes come from `dtype=` or from one call of `f` per
chunk on a copy of the state; ints become floats. `_copy_state`
deep-copies state (keeping namedtuples) so neither that call nor `reset()`
changes the initial state. Its loop kernel calls the compiled `f`
directly. It always has direct feedthrough: its state never breaks a
loop, a `Delay` must.

## Continuous blocks

`ODE` is a `StateBlock` that integrates per sample: `_y(u)` samples the
output at `t_k = k / fs`, `_advance(u)` integrates to `t_k + 1/fs` with the
input held (`solve_ivp`, or fixed-step RK4). `feedthrough(L)` is False only
at `L == 1` and only if `g` ignores the input, so a loop through it becomes
a loop cluster with the ODE as a state block. `output(L)` is valid only for
`L == 1`; longer chunks go through `step`. The RK4 loop kernel inlines `f`
and `g` compiled with numba; `solve_ivp` methods have no kernel. Outside
compiled clusters both integrate in Python, sample by sample. `solve_ivp`
starts afresh every sample, so its step size does not carry over; `t`
comes from the block's own count `k`.

`ContinuousLTI` is an `ODE` with `f`, `g` from `A, B, C, D` (so any
`solve_ivp` method works). With `method="zoh"` it replaces integration by
the exact discretisation `Ad, Bd = expm([[A, B], [0, 0]] / fs)`, made once
per `fs` (`_discretise`), and runs whole chunks through `_zoh`: explicit
loops compiled with numba, or matrix products per sample without it. Its
loop kernel covers single-input single-output systems. Exact ZOH is
linear, so it takes complex inputs; with integration `real_inputs`
refuses them. `bode(f)` gives the continuous response from `A, B, C, D`.

## Multirate

Rate changers set `resampling_ratio`; the diagram gives each its input and
output lengths. `Upsample` and `Repeat` keep no state; `Downsample` counts
input samples, so the kept phase survives chunk edges. `Interpolate`,
`Decimate`, `Resample` and `DDC` share `_Polyphase`: it keeps the last
inputs and the global counts `n_in`, `m_out`, and `_poly_kernel` (numba)
computes only the kept outputs of zero-stuffing by `up`, filtering and
keeping every `down`-th. Without numba, or for vector signals,
`_step_reference` does exactly that with `scipy.signal.convolve` over the
kept history. History, not filter state, is kept, so changed taps apply
without a transient. `DDC` moves the mixer behind the filter (band-pass
taps `h[k] e^(jwk)`, then a rotation of the kept outputs) and on a change
of `f` shifts the LO phase so the output stays continuous at the sample it
is centred on.

## Streams and devices

`SampleBuffer` is a FIFO guarded by a `threading.Condition`. `push` appends
and drops the oldest samples beyond `capacity` (`overflows`); `pull(n,
timeout)` returns exactly `n` samples and fills what does not arrive in
time with zeros (`underruns`). `wake` ends waits until `rearm`.
`FromQueue` pulls in `step_source`; with `timeout=None` it waits, so the
data paces the diagram (`external_clock`). `ToQueue` pushes in `step`; with
`block=True` it waits for its reader. `Diagram.stop` calls `wake`, `open`
calls `rearm`.

`devices` builds on these. `RtlSdrSource` and `AudioIn` are `FromQueue`s
fed by the driver's thread or callback; `AudioOut` is a `ToQueue` emptied by
the sound card's callback, silent until `prefill` samples are buffered,
played out on `close`. `open` imports the driver (`_driver`), so the
package imports without it. `center_freq` and `gain` apply to an open
receiver at once.

## Measurement and live plots

`Scope` and the sinks of `measure` (`_Sink`) take a lock in `step` and in
their readings, so another thread never reads half an update;
`Scope.snapshot` returns time and data consistent with each other. Spectra
(`SpectrumAnalyzer`, `Spectrogram`) frame the stream with `_Frames`, scaled
as `scipy.signal.welch(..., detrend=False)`. Decisions use absolute sample
indices (`skip`, `offset`, frame starts), so chunking changes no reading.
`BERCounter` finds the delay of the received bits on windows of `sync`
bits and can `request_stop`.

`live.watch` draws in the caller's thread while the diagram runs in its
own. Each frame calls `draw(fig)`; once `draw` returns an `update`
function, frames call only that, and in a window blit the artists it
returns. `_mode` chooses between updating a Jupyter display and a GUI
window; `window=True` opens a Tk window. Interrupting stops the run; at the
end `Diagram.wait` re-raises its error.

## Testing helper

`testing.assert_chunk_invariant(make, inputs)` builds a fresh block per run,
fed by `FromArray` and read by `Scope`s, and compares one run in a single
chunk with a run after `reset`, runs in each of `chunks` and runs in random
chunk sizes (multiples of `granularity`). It names the first differing
sample.

## Drawing

`drawing.view(d)` turns blocks into `Node`s and lines into `Edge`s.
`_layout` marks feedback lines (back edges of a depth-first search from the
sources), assigns columns by the longest path over the other lines, moves
sources next to their first consumer, and gives a line that skips columns
a `_Lane` in each of them, so it gets a slot of its own and never runs
through a box. Six barycenter sweeps order each column, once with lanes
first and once last on ties; the order with the fewest crossings wins.
Opened subsystems (`expand=True`) add rules: `_columns` starts a
subsystem after the blocks feeding it and ends it before the blocks it
feeds, so it owns its columns (if lines leave and re-enter it, the plain
rules apply); a lane belongs to the frames of its ends that span its
column; `_together` keeps each subsystem's blocks and lanes adjacent in
every column. `draw` frames them with a rectangle, or an outline stepping
between columns where another block shares one; DOT uses clusters,
Mermaid subgraphs. `draw` and `to_text` use this layout; `to_dot` and
`to_mermaid` call it only to mark feedback lines and leave placement to
Graphviz and Mermaid. `Diagram._repr_svg_` draws with matplotlib, so
Jupyter shows diagrams as graphs. In `draw`,
`_symbol` picks a compact shape for `Sum`, `Product` (circles), `Gain`
(triangle) and `Const`; their ports sit on the shape's outline, their name
shows only when it is not the default type-plus-counter. Boxes share one
width, from the longest name. Feedback lines run dashed below the blocks;
in a multirate diagram a label gives the new rate after each block that
changes it; probes are tags on their lines.

## Strengths and limits

Measured figures are single runs with numba; they show orders of
magnitude, not benchmarks.

### Strengths

- The chunk contract holds across the library and is tested per block
  (`testing.assert_chunk_invariant`; sinks, which it cannot read, over
  several chunk sizes): random sources keep one stream per
  block (`Noise.output`), rate changers keep global counts (`_Polyphase`,
  `Downsample`), sinks use absolute indices. The chunk length is a speed
  setting only.
- Whole chunks go through NumPy and SciPy (`FIR.step` switches to
  `oaconvolve` from 128 taps, `IIR.step` carries `zi` through `sosfilt` or
  `lfilter`). A chain of `Noise`, a 31-tap `FIR`, `Gain` and `Scope` took
  about 0.12 µs per sample at chunk 4096.
- Feedback costs per-sample execution only inside the loop
  (`Diagram._with_clusters`); the rest stays vectorised. The generated
  loop (`LoopCluster._generate`) ran `Sum`, `Delay(1)`, `Gain` at about
  0.08 µs per sample, against 40 µs in Python (both including the chain
  around the loop). State is copied in and out, so blocks stay inspectable
  and a failed call leaves them untouched.
- Blocks are plain objects: parameters are attributes, blocks and
  subsystems work as functions (`Block.__call__`, `Subsystem.__call__`),
  operators build blocks (`core._Arithmetic`), functions become blocks
  (`FuncChunk`, `FuncSingle`).
- Subsystems cost nothing at run time and stay open to change
  (`Subsystem.replace`, members by name); library parts are functions
  returning them, not duplicated blocks.
- Rates are exact fractions, propagated along lines and back to sources
  inside subsystems (`Diagram._assign_rates`); polyphase resampling
  computes only the kept samples (`multirate._poly_kernel`).
- `ContinuousLTI` with ZOH is exact at the samples for held inputs and
  cheap: about 0.04 µs per sample (`_discretise`, `_zoh`).
- Failures name the block at fault: NaN/Inf and dropped imaginary parts on
  the first chunk, opt-in input checks, a note on every exception
  (`Diagram._step`, `_checks`).
- Live runs come built in: a background thread with pacing, overrun count
  and queued changes (`Diagram.start`, `set`, `call`), locked sinks,
  thread-safe streams (`SampleBuffer`) and devices that pace the run
  (`external_clock`).

### Limits

- No global continuous solver. Each `ODE` integrates alone, one sample at
  a time, with its inputs held (`ODE._advance`), so coupled continuous
  blocks see each other through a one-sample hold, and `fs`, not `rtol`,
  bounds the accuracy of the coupled system.
- `ODE.step` and `ODE.update` run in Python per sample: about 280 µs per
  sample with RK45 and 230 µs with RK4 (four substeps). Only RK4 inside a
  compiled loop cluster runs compiled (`_kernels._ode`).
- No events: no zero-crossing detection or event location (`solve_ivp` gets
  no `events`), switching happens on sample instants (`Step.generate`), no
  differential-algebraic systems; an undelayed loop is an error, not
  solved (`AlgebraicLoopError`).
- No linearisation, trimming or frequency response of a diagram; only per
  block `FIR.freqz`, `IIR.freqz` and `ContinuousLTI.bode`. Nothing
  symbolic.
- Rates are static and output lengths fixed: `_normalise` demands
  `L * resampling_ratio` samples per chunk, so data-dependent output
  counts (timing recovery that slips a symbol, packet extraction) need a
  fixed-rate encoding. Inputs of one block must share a rate (no implicit
  rate transition), run and chunk lengths are multiples of `granularity`,
  and a loop cannot span rates unless a delay of a whole chunk breaks it
  (`LoopCluster.__init__`).
- Compiled loops cover a fixed set of types, found by exact type
  (`_kernels.kernel`) and registered in a private module
  (`_kernels.kernel_for`), for scalar signals only. Rate changers,
  `Unwrap`, `FSKDemod`, user blocks, `ODE` with `solve_ivp`, `ContinuousLTI`
  with several inputs or outputs and `FuncSingle` in Python keep a cluster
  in Python. Generated code compiles anew in each process
  (`LoopCluster._generate`, `func_single._build`).
- Execution is single-threaded: one thread steps every block
  (`Diagram._chunks`); only the fixed kernels release the GIL
  (`_numba.njit`). Real time is soft: late chunks are counted
  (`overruns`), and live drawing in the main thread competes for the GIL.
- Each block costs about 14 µs per chunk (a chain of `Gain`s at chunk
  64); a profile puts about 60 % of `_step` in `Diagram._len`, which
  multiplies `Fraction`s every chunk. Short chunks for low latency are
  expensive.
- Global state: `core._EDITS` is one counter for all diagrams, so a line
  drawn anywhere rebuilds every diagram at its next run and re-arms its
  first-chunk check; `Block._counter` makes default names depend on what
  the process built before, and names need not be unique
  (`Diagram.__getitem__` returns the first); `func_single._LOOPS` keeps
  every wrapped function alive.
- Thread safety covers readings and queued changes, not the structure:
  `probe` and `remove_probe` change `_probes` while the run thread iterates
  it in `_step`, and attribute writes outside `set` land mid-chunk.
- Errors are not transactional outside compiled clusters: an exception in
  `_step` leaves the blocks before it advanced while state blocks and `n`
  are not (a `FIR` before a failing block kept its new history), so
  continuing after an error needs `reset()`.
- Central modules know concrete types: `_kernels` imports every block with
  a kernel, `core` imports `sources` and `blocks` for operators and
  `FuncChunk`, `drawing._symbol` tests for `Sum`, `Product`, `Gain`,
  `Const`, `measure` uses `digital.nearest`. A new block type that should
  compile or draw as a symbol means editing them.
- `SampleBuffer.pull` concatenates the whole backlog on every pull.
  Devices cover reception (`RtlSdrSource`) and sound cards only.
