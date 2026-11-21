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
pip install "sigline[rtlsdr]"  # RTL-SDR receiver
pip install "sigline[audio]"   # sound cards
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

`p = d.probe("lp")` records any output without drawing a line to a Scope
(targets: a block, a port, `"name"`, `"name:port"`); `keep`, `every` as for
`Scope`. `d.plot()` plots every Scope.

### Running live

```python
d.start(chunk=1024)  # background thread, paced to real time
d.running, d.overruns  # overruns: chunks that finished late
d.stop()  # after the current chunk
d.start(10.0, realtime=False)  # as fast as possible
d.wait()  # until it ends; re-raises its errors
```

Live plots: `sl.watch(d, draw)` redraws a figure while `d` runs, in Jupyter
or a matplotlib window; `draw(fig)` is any plotting code, e.g.

```python
def draw(fig):
    ax1, ax2 = fig.subplots(2, 1)
    rx.plot(ax=ax1)
    rx.psd(ax=ax2)


d.start()
sl.watch(d, draw)  # Ctrl+C / stop button stops d
```

Blocks holding devices implement `open()` and `close()`: a run opens them, a
background run closes them at its end, `d.close()` or `with d:` otherwise.
Read scopes meanwhile with `sc.snapshot()`. Change parameters with
`d.set(block, f=2e3)` or `d.call(fn)`: applied between chunks, never in the
middle of one.

### Streams

`FromQueue` is a source fed from another thread (`src.push(x)`), `ToQueue` a
sink read from one (`sink.pull(n)`); both count `underruns` (zero-filled
samples) and `overflows` (dropped samples). A `FromQueue` that waits for data,
or a `ToQueue(block=True)` that waits for its reader, paces a background run;
`d.start()` then skips wall-clock pacing.

### Devices

```python
sdr = sl.RtlSdrSource(145.8e6, gain=30)  # complex IQ at the diagram's fs
mic, spk = sl.AudioIn(), sl.AudioOut()  # sound card in and out
```

They open when the diagram runs and pace a background run with their own
clock. `d.set(sdr, center_freq=...)` retunes while running. `AudioOut` waits
for `prefill` samples before playing; keep it above one chunk plus one card
block. Drivers are optional extras, imported on first use. For the RTL-SDR
Blog V4, pyrtlsdr needs the V4-capable librtlsdr from rtl-sdr-blog.

### Feedback

A `StateBlock` has no direct feedthrough: `output(L)` reads the state,
`update(L, **inputs)` advances it. The diagram takes all such outputs first
and updates last, so a loop through one, e.g. a delay of at least one chunk,
runs vectorised. A loop without one raises `AlgebraicLoopError`. Blocks can
decide per chunk length with `feedthrough(L)`.

### Sample rates

`resampling_ratio` is a block's output rate / input rate as a `Fraction`
(default 1). Sources run at their own `fs` or the diagram's; rates propagate
along the lines. `d.rates()` lists them; chunk and run lengths are multiples of
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
| `Sine(f, amp=1, phase=0, complex=False)` | phase-continuous `cos`, or `exp(j·)` |
| `Noise(std=1, seed=0, complex=False)` | white Gaussian, reproducible |

### Math

| block | output |
| --- | --- |
| `Gain(g)` | `g · x` |
| `Sum("+-")` | `x0 - x1`; one sign per input `x0, x1, ...` |
| `Product(n=2)` | `x0 · x1 · ...` |
| `Func(fn, outputs=("y",))` | `fn(...)`; its arguments are the input ports |
| `Delay(n=1, init=0)` | `x` delayed by `n`; breaks loops for chunks `<= n` |
| `Mixer(f, lo="cos")` | `x` times a phase-continuous LO: `cos`, `sin`, `exp`, `exp-` |

### Filters

| block | output |
| --- | --- |
| `FIR(taps)` | FIR filtered `x`; `.freqz()` |
| `IIR(sos=)`, `IIR(b=, a=)` | IIR filtered `x`; `.freqz()` |

### Scope

`Scope()` records its input: `.data`, `.t` (absolute time / s). `keep=N`
keeps the newest `N` samples, `every=M` records every `M`-th. `.plot()` draws
it over time (I and Q if complex), `.welch()` returns its PSD (no
detrending: DC can be a carrier), `.psd()` plots that in dB. `.snapshot()`
returns matching `(t, data)`, also while a diagram runs in the background.

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
- [`02_standard_blocks`](notebooks/02_standard_blocks.ipynb): sources, a DSP
  chain with spectra, filters, I/Q down-conversion, a control loop, probes.
- [`03_live`](notebooks/03_live.ipynb): background runs in real time, changes
  while running, live plots, samples from other threads and devices.

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
