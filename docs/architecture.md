# Architecture

How sigline works inside. The README is the user guide; this file explains the
machinery for anyone changing it.

## Modules

| module | contents |
| --- | --- |
| `sigline.core` | `Block`, `Port`, `>>` wiring, standalone calls |
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
2. **Scheduling.** `compile(L)` sorts the blocks topologically over the wires
   (Kahn's algorithm) and caches the order per chunk length. A cycle raises
   `AlgebraicLoopError` naming the blocks involved.
3. **Execution.** `run` splits the requested samples into chunks; `_step(L)`
   calls each block in order with the arrays its drivers produced for this
   chunk. `n` counts the simulated samples; `reset()` resets every block.
