# Architecture

How sigline works inside. The README is the user guide.

## Modules

| module | contents |
| --- | --- |
| `sigline.core` | `Block`, `StateBlock`, `Port`, `>>`, standalone calls |
| `sigline.diagram` | `Diagram`: discovery, rates, scheduling, execution |
| `sigline._checks` | signal checks for `Diagram._step` |
| `sigline.scope` | `Scope` |

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
   times `rate`. Sourceless loops: 1. Errors for mismatched inputs, a
   contradicting `fs`, a non-`Fraction` `rate`. `granularity` = lcm of the
   denominators; `_len(b, N)` gives a block's lengths for `N` diagram
   samples.
3. Scheduling: `compile(N)` splits blocks by `feedthrough(L_in)`, sorts the
   feedthrough ones topologically (Kahn) over lines between them, caches
   `(state, order)` per `N`. A remaining cycle raises `AlgebraicLoopError`.
4. Execution: `_step(N)` runs `output` of state blocks, `step` of the order,
   then `update` of state blocks, each with its own lengths.
5. Stop requests: `request_stop` sets `_stop_request`; `run` checks after
   each chunk, stores the first reason in `stopped`, clears all, breaks.
6. Checks: every block call goes through `call(b, fn, /, ...)`
   (positional-only: ports may be named `b` or `fn`). When checking, it
   applies opt-in input checks, raises `ComplexWarning` as `TypeError` naming
   the block, and checks outputs for NaN/Inf. Inputs were checked as their
   drivers' outputs, so the first failure is the culprit. Exceptions get one
   `add_note` naming the innermost block.
