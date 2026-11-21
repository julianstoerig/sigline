# Architecture

How sigline works inside. The README is the user guide.

## Modules

| module | contents |
| --- | --- |
| `sigline.core` | `Block`, `StateBlock`, `Port`, `>>`, standalone calls |
| `sigline.diagram` | `Diagram`: discovery, rates, scheduling, execution |
| `sigline._checks` | signal checks for `Diagram._step` |
| `sigline._format` | `si()`: SI-prefixed labels |
| `sigline.blocks` | math and general blocks |
| `sigline.filters` | `FIR`, `IIR` |
| `sigline.scope` | `Scope` |
| `sigline.sources` | `Source` base and sources |
| `sigline.streams` | `SampleBuffer`, `FromQueue`, `ToQueue` |
| `sigline.live` | `watch`: live plots |
| `sigline.devices` | `RtlSdrSource`, `AudioIn`, `AudioOut` (optional drivers) |
| `sigline.testing` | `assert_chunk_invariant` |

## Blocks

- Signals: one NumPy array per line and chunk; axis 0 is time, extra axes are
  vector signals.
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

## Diagram

1. Discovery: walk `_sources` and `_sinks` from the given blocks; sort by
   `_id`. Unconnected inputs are an error.
2. Rates: `_assign_rates` gives every block input and output rates as
   `Fraction`s of `fs`. Sources: own `fs` or 1. Others: propagate along lines,
   times `resampling_ratio`. Sourceless loops: 1. Errors for mismatched
   inputs, a contradicting `fs`, a non-`Fraction` `resampling_ratio`. `granularity` = lcm of the
   denominators; `_len(b, N)` gives a block's lengths for `N` diagram
   samples.
3. Scheduling: `compile(N)` splits blocks by `feedthrough(L_in)`, sorts the
   feedthrough ones topologically (Kahn) over lines between them, caches
   `(state, order)` per `N`. A remaining cycle raises `AlgebraicLoopError`.
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
