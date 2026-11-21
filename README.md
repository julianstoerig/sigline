# sigline

Block-diagram simulation of DSP, communications and mixed continuous/discrete
systems, built on the scientific Python stack.

Blocks are wired with `>>`, but the code stays flat Python: plain scripts,
plain functions, parameters as attributes and parameter sweeps as `for` loops.
Signals are NumPy arrays processed in chunks; every block owns its state, so
any chunking gives the same result.

> **Status:** early development. The API is being built up step by step and is
> not yet released on PyPI.

## Installation

```bash
pip install sigline            # core
pip install "sigline[numba]"   # with numba for compiled per-sample code
```

numba is optional: everything runs without it, only slower.

## Blocks

A block owns its parameters (plain attributes) and its state, and processes
one chunk of samples per call. Signals are NumPy arrays whose first axis is
time. Write a block by subclassing `Block`: store the parameters, call
`super().__init__(**kw)` (which calls `reset()`), and implement `step`, whose
argument names are the input ports:

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

The class attributes `inputs` and `outputs` name the ports (default: one
input `x`, one output `y`). `step` returns one array, a tuple in `outputs`
order or a dict `{port: array}`. A block with `inputs = ()` is a source and
implements `step_source(L)`, which produces `L` samples.

**Blocks are plain functions too.** `acc(x)` processes one chunk outside any
diagram; state carries over between calls, and `acc.reset()` starts again.
`src(L)` asks a source for `L` samples.

**Wiring.** `a >> b` connects the only output of `a` to the only input of `b`;
`a["re"] >> b["x"]` picks ports by name, and `blk.inp(name)` / `blk.out(name)`
are the explicit forms, needed when a name is both an input and an output.
`>>` returns the destination, so chains read left to right: `a >> b >> c["x"]`.
An output can drive many inputs; each input has exactly one driver.

## Diagrams

`Diagram(*blocks, fs=...)` collects every block connected to the given ones
(one is enough) and runs them at the sample rate `fs`, which each block can
read as `self.fs`:

```python
d = sl.Diagram(sink, fs=48_000)
d.run(1.0, chunk=1024)  # seconds; or d.run(n=48_000)
gain.k = 0.5  # parameters are attributes: change them between runs
d.run(0.5)  # continues where the last run stopped
d.reset()  # back to the initial state
```

Time advances in chunks; per chunk every block runs once, in topological
order. Because blocks own their state, **the chunk size never changes the
result**, only the speed. `d.n` counts the samples simulated so far and `d.t`
is their time axis; `d["name"]` finds a block by name. `print(d.summary())`
shows the execution order and every wire. 
**Signal checks.** On the first chunk after building or `reset()`, the
diagram checks every output for NaN/Inf and names the first block that
produced one, and turns a dropped imaginary part (NumPy's `ComplexWarning`)
into an error at the block that dropped it. Blocks can opt in to input checks
with the class attributes `same_shape_inputs = True` (all inputs have the same
vector width) and `real_inputs = True` (complex inputs are rejected).
`Diagram(..., check=True)` checks every chunk, `check=False` never. Any
exception raised inside a block gets a note naming the block and its inputs.

**Stopping early.** A block can call `self.request_stop("reason")` from its
`step`; the diagram finishes the chunk and ends the run, and `d.stopped`
reads `"<block>: reason"` (it is `None` after a complete run). The next
`d.run(...)` continues normally. This lets a measurement end a simulation as
soon as it has enough data.

**Sample rates.** Sources run at their own `fs` if given, otherwise at the
diagram's. Every other block takes the rate of its inputs (all must agree)
and multiplies it by its `rate` attribute, the ratio output rate / input rate
as a `Fraction` (an upsampler by 4 has `rate = Fraction(4)`). The attribute
name `rate` is reserved for this. `d.rates()` lists `{name: (fs in, fs out)}`
and each block's `fs` is set to its input rate. Run lengths and chunk sizes
are counted at the diagram rate and kept at multiples of `d.granularity`, so
every block gets a whole number of samples per chunk.

**Feedback.** A `StateBlock` has no direct feedthrough: its output depends
only on its state. It implements `output(L)` (which must not change the
state) and `update(L, **inputs)` (which advances it by one chunk). Per chunk
the diagram first takes the outputs of all state blocks, then runs the other
blocks in topological order, then updates the state blocks. A feedback loop
therefore works whenever it passes through a block without feedthrough at
the chunk length used, such as a delay of at least one chunk. A loop in which
every block needs its own input of the same chunk cannot be scheduled and
raises `AlgebraicLoopError`, naming the blocks involved. A block can decide
per chunk length by overriding `feedthrough(L)`.

## Examples

Notebooks in [`notebooks/`](notebooks) walk through the library from start
to end, with every result checked against theory:

- [`01_blocks_and_diagrams`](notebooks/01_blocks_and_diagrams.ipynb): write
  blocks, wire and run diagrams, feedback loops, sample rates, parameter
  sweeps, stopping early and the signal checks.

## Development

```bash
git clone https://github.com/julianstoerig/sigline.git
cd sigline
pip install -e ".[numba]" --group dev
pytest                          # includes the notebooks
ruff check . && ruff format --check .
python tools/run_notebooks.py   # refresh the stored notebook outputs
```

## License

MIT, see [LICENSE](LICENSE).
