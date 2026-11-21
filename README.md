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

## Development

```bash
git clone https://github.com/julianstoerig/sigline.git
cd sigline
pip install -e ".[numba]" --group dev
pytest
ruff check . && ruff format --check .
```

## License

MIT, see [LICENSE](LICENSE).
