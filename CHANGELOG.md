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
- Sample rates: `Block.resampling_ratio`, sources with their own `fs`, `Diagram.rates()`
  and `Diagram.granularity`.
- Stop requests: `Block.request_stop(reason)` ends a run after the current
  chunk; `Diagram.stopped` gives the reason.
- Signal checks (`Diagram(check="first" | True | False)`): NaN/Inf and
  dropped imaginary parts name the failing block; opt-in input checks;
  exceptions get a note naming the block.
- `Diagram.summary()`: execution order and lines as text.
- Notebook `01_blocks_and_diagrams`: the core from start to end.
- `Scope`: records its input; `keep` (ring buffer), `every` (decimation).
- `Source` (sample counter, `generate(n)`) and `Const`.
- `Step`.
- `Signal`: any vectorised function of time.
- `FromArray`: plays back an array, padded with zeros or looped.
- `sigline.testing.assert_chunk_invariant`: same output for any chunking.
- `Sine`: phase-continuous oscillator, real or complex.
- `Noise`: white Gaussian noise, real or circular complex, seeded.
- `Gain`.
- `Sum`: signed sum of inputs `x0, x1, ...`.
- `Product`.
- `Func`: any vectorised function as a block; arguments become inputs.
- `Delay`: breaks feedback loops for chunks of at most `n` samples.
- `FIR`: persistent state, real/complex/vector signals, FFT path for long filters.
- `IIR`: from second-order sections or `b, a`, persistent state.
- `Mixer`: multiplies by a phase-continuous LO (`cos`, `sin`, `exp`, `exp-`).
- `Scope.plot`, `Scope.welch`, `Scope.psd`; matplotlib becomes a dependency.
- `Diagram.probe`, `remove_probe`, `scopes`, `plot`.
- Notebook `02_standard_blocks`.
- `Sum` rejects signs other than `+` and `-`.
- `Scope.snapshot`; scopes can be read while a diagram runs.
- `Diagram.start`, `stop`, `wait`, `running`, `overruns`: background runs paced
  to real time or free-running.
- `Diagram.set`, `Diagram.call`: parameter changes between chunks of a running
  diagram.
