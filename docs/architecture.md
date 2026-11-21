# Architecture

How sigline works inside. The README is the user guide; this file explains the
machinery for anyone changing it.

## Modules

| module | contents |
| --- | --- |
| `sigline.core` | `Block`, `StateBlock`, `Port`, `>>` wiring, standalone calls |
| `sigline.diagram` | `Diagram`: discovery, scheduling, execution |

## Signals and blocks

- Every signal on a wire is a NumPy array per chunk; the first axis is time,
  extra axes give vector signals.
- Blocks own all their state. The contract: **results must not depend on the
  chunk size**. That is what makes the vectorised fast path correct.
- `Block.__init__` assigns a default name (class name plus a per-class
  counter), records the creation order in `_id` (used for stable orderings)
  and calls `reset()`. Subclasses therefore store their parameters *before*
  calling `super().__init__`.
- `step(**inputs)` returns an array, a tuple (in `outputs` order) or a dict.
  `_normalise` turns any of these into `{port: array}`, broadcasts scalars over
  the chunk and checks the length.
- `Block.__call__` runs one chunk outside a diagram, through the same
  `_normalise`, so standalone use and diagram use behave alike.

## Ports and wiring

- `Port` is a frozen dataclass with identity equality, so ports can be dict
  keys; `block["name"]`, `inp()` and `out()` create them on demand.
- `a >> b` records the connection on both ends: `b._sources[port]` is the
  driving output port, `a._sinks` lists the driven input ports. A block with
  several inputs (or outputs) must be addressed by port name.

## Diagram

1. **Discovery.** `Diagram(*blocks)` walks `_sources` and `_sinks` from the
   given blocks and keeps every block it reaches, sorted by creation order.
   Unconnected inputs are an error.
2. **Scheduling.** `compile(L)` splits the blocks by `feedthrough(L)`. The
   feedthrough blocks are sorted topologically over the wires between them
   (Kahn's algorithm); wires out of non-feedthrough blocks impose no order.
   The schedule `(state blocks, order)` is cached per chunk length. A cycle
   left among feedthrough blocks raises `AlgebraicLoopError`.
3. **Execution.** `run` splits the requested samples into chunks. `_step(L)`
   runs three phases: `output(L)` of every state block, then `step` of the
   feedthrough blocks in order, each with the arrays its drivers produced for
   this chunk, then `update(L, **inputs)` of every state block. `n` counts the
   simulated samples; `reset()` resets every block.

`StateBlock` splits a block into `output(L)` (must not change state) and
`update(L, **inputs)`. Its `step` and `step_source` call both in turn, so a
state block also works standalone and inside the order when it returns
`feedthrough(L) is True` (e.g. a delay shorter than the chunk).
