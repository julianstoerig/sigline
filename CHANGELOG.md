# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Project skeleton: packaging, tests and continuous integration.
- `Block` and `Port`: blocks with named ports, `>>` wiring and standalone use
  as plain functions.
- `Diagram`: runs connected blocks chunk by chunk in topological order;
  runs continue, `reset()` restarts; `AlgebraicLoopError` for loops.
- `StateBlock` (`output` / `update`) and `Block.feedthrough`: feedback loops
  through blocks without direct feedthrough.
- Sample rates: `Block.rate`, sources with their own `fs`, `Diagram.rates()`
  and `Diagram.granularity`.
- Stop requests: `Block.request_stop(reason)` ends a run after the current
  chunk; `Diagram.stopped` gives the reason.
- Signal checks (`Diagram(check="first" | True | False)`): NaN/Inf and
  dropped imaginary parts name the failing block; opt-in input checks;
  exceptions get a note naming the block.
- `Diagram.summary()`: execution order and wires as text.
