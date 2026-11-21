<img src="docs/logo.svg" alt="" width="96">

# sigline

Block diagrams for DSP, communications and mixed continuous/discrete systems,
on NumPy and SciPy.

Blocks are connected by lines with `>>`; everything else is plain Python.
Parameters are attributes, sweeps are `for` loops. Signals are NumPy arrays
in chunks, and blocks own their state, so chunking never changes results.

> Status: early development, not yet on PyPI.

## Installation

```bash
pip install sigline            # core
pip install "sigline[numba]"   # compiled per-sample code
```

numba is optional; without it everything runs, only slower.

## Blocks

Subclass `Block`, store parameters, call `super().__init__(**kw)` (it calls
`reset()`), implement `step`. Argument names of `step` are the input ports.

```python
import numpy as np
import sigline as sl


class Accumulate(sl.Block):
    def reset(self):
        self.acc = 0.0

    def step(self, x):
        y = self.acc + np.cumsum(x)
        self.acc = y[-1]
        return y
```

- `inputs` / `outputs` name the ports; default one input `x`, one output `y`.
- `step` returns an array, a tuple in `outputs` order, or `{port: array}`.
- A source has `inputs = ()` and implements `step_source(L)`.
- Blocks are functions too: `acc(x)` processes one chunk, `src(L)` makes `L`
  samples. State carries over; `reset()` restarts.

## Lines

```python
a >> b  # only output of a to only input of b
a["re"] >> b["x"]  # ports by name
blk.inp("y"), blk.out("y")  # when a name is both input and output
a >> b >> c["x"]  # >> returns the destination
```

An output can drive many inputs; each input has one driver.

## Diagrams

```python
d = sl.Diagram(sink, fs=48_000)  # any one block of the graph
d.run(1.0, chunk=1024)  # seconds; or d.run(n=48_000)
gain.k = 0.5  # change parameters between runs
d.run(0.5)  # continues
d.reset()
```

Each chunk, every block runs once in topological order. The chunk size
changes speed, never results. `d.n` counts samples, `d.t` is their time,
`d["name"]` finds a block, `print(d.summary())` shows the order and the
lines.

### Feedback

A `StateBlock` has no direct feedthrough: `output(L)` reads the state,
`update(L, **inputs)` advances it. The diagram takes all such outputs first
and updates last, so a loop through one, e.g. a delay of at least one chunk,
runs vectorised. A loop without one raises `AlgebraicLoopError`. Blocks can
decide per chunk length with `feedthrough(L)`.

### Sample rates

`rate` (reserved) is a block's output rate / input rate as a `Fraction`.
Sources run at their own `fs` or the diagram's; rates propagate along the
lines. `d.rates()` lists them; chunk and run lengths are multiples of
`d.granularity`.

### Stopping early

`self.request_stop("reason")` inside a block ends the run after the chunk;
`d.stopped` gives the reason. The next run continues.

### Signal checks

On the first chunk after build or `reset()`, NaN/Inf and dropped imaginary
parts raise an error naming the first block at fault. Opt-in input checks:
`same_shape_inputs = True`, `real_inputs = True`. `check=True` checks every
chunk, `check=False` never. Exceptions inside blocks get a note naming the
block.

## Blocks in the library

### Sources

Subclass `Source` and implement `generate(n)` for integer sample indices `n`
(time is `n / self.fs`).

| block | output |
| --- | --- |
| `Const(value)` | constant |
| `Step(T, after=1, before=0)` | `before`, from `T` / s on `after` |
| `Signal(fn)` | `fn(t)`, any vectorised function of time |
| `FromArray(data, loop=False)` | `data`, then zeros (or repeats) |

### Scope

`Scope()` records its input: `.data`, `.t` (absolute time / s). `keep=N`
keeps the newest `N` samples, `every=M` records every `M`-th.

## Testing blocks

```python
from sigline.testing import assert_chunk_invariant

assert_chunk_invariant(lambda: MyBlock(0.3), x)  # x, {port: x} or n for sources
```

Runs the block in one chunk, after `reset()`, in fixed and in random chunk
sizes, and fails at the first differing sample.

## Examples

- [`01_blocks_and_diagrams`](notebooks/01_blocks_and_diagrams.ipynb): blocks,
  lines, feedback, sample rates, sweeps, stopping, checks.

## Development

```bash
git clone https://github.com/julianstoerig/sigline.git
cd sigline
pip install -e ".[numba]" --group dev
pytest                          # includes the notebooks
ruff check . && ruff format --check .
python tools/run_notebooks.py   # refresh notebook outputs
python tools/make_logo.py       # regenerate docs/logo.svg
```

## License

MIT, see [LICENSE](LICENSE).
