<img src="docs/logo.svg" alt="" width="96">

# sigline

Computational graphs for DSP, communications and more.
Supports mixed continuous/discrete systems, mixed sample rates and custom blocks.
It is built on the Python scientific stack for almost all of its functionality.

Simple diagrams are written with operators, like this FM broadcast receiver
(tuned 100 kHz below the station, which the mixer moves to 0 Hz):
`sl.RtlSdrSource(103.7e6, fs=fs) >> sl.mixer(100e3, "exp-") >> sl.fm_demod(75e3, channel_width=200e3, fs=fs)`.
More complicated diagrams are composed from these simple connections:

```python
import sigline as sl

s = sl.Sum()
g = sl.Gain(-0.1)
z = sl.Delay(tau=0.1)
x = sl.Impulse()

# connect the impulse signal to one input of the summing junction
x >> s["x0"]
# a feedback loop: a delayed and scaled line from the junction's output
# back to its other input
s["y"] >> z >> g >> s["x1"]

d = sl.Diagram(s, fs=1000)
# show the diagram in a Jupyter notebook, or give a path to save it to disk
d.draw()
# run 1 s; chunks up to the 0.1 s delay run the loop vectorised, longer
# ones sample by sample
d.run(1.0, chunk=100)
```

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

numba is optional; without it everything runs, only slower. `sl.HAVE_NUMBA`
tells which; `SIGLINE_NO_NUMBA=1` turns it off.

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
- `info()` returns a one-line parameter summary for drawings.
- Blocks are functions too: `acc(x)` processes one chunk, `src(L)` makes `L`
  samples. State carries over; `reset()` restarts.

## Lines

```python
a >> b  # only output of a to only input of b
a["re"] >> b["x"]  # ports by name
blk.inp("y"), blk.out("y")  # when a name is both input and output
a >> b >> c["x"]  # >> returns the destination
c << b << a  # the same lines, right to left
(a, b) >> add  # a to add["x0"], b to add["x1"]
src >> np.abs >> sc  # a plain function becomes a FuncChunk block
```

Arithmetic on signals adds blocks: `a + b`, `a - b` give a `Sum`, `a * b`,
`a / b` a `Product`, `2 * a`, `a / 2`, `-a` a `Gain`, and a number next to a
signal a `Const`. It acts on signals, so `a + b >> sc` is `(a + b) >> sc`.

An output can drive many inputs; each input has one driver.

## Subsystems

A function that wires blocks can return them as one block: a `Subsystem`
names which inner ports are its inputs and outputs.

```python
import scipy.signal as sig
import sigline as sl

fs = 48_000


def upconverter(fc, cutoff):
    mix = sl.Product(name="mix")
    sl.Sine(fc, name="lo") >> mix["x1"]
    hpf = mix >> sl.FIR(sig.firwin(101, cutoff, pass_zero=False, fs=fs), name="hpf")
    return sl.Subsystem(inputs={"x": mix["x0"]}, outputs={"y": hpf})


up = upconverter(10e3, 10e3)
sl.Sine(3e3) >> up >> sl.Scope("rf")  # used as one block
up.lo.f = 12e3  # inner blocks by name
hp = sig.butter(6, 12e3, "highpass", fs=fs, output="sos")
up.replace(up.hpf, sl.IIR(sos=hp))  # swap one; its lines stay
up.draw()  # the inside, ports marked; diagrams show up as one box
```

Members are the blocks behind the given ports when the subsystem is made:
connected to them, but not through lines that already leave an output
port, which lead outside. They are renamed by path (`"upconverter1/hpf"`). The ports are the inner blocks' own, so a diagram runs exactly the
blocks inside, at no extra cost. An input given as a list feeds all its
ports, e.g. `inputs={"x": [mix_i["x0"], mix_q["x0"]]}`. Subsystems nest.

Like a block, an unconnected subsystem works as a function: `up(x, fs=fs)`
runs one chunk and keeps the state. It runs as a small diagram, so it needs
a sample rate: `fs=`, or the one its blocks have. `info=` takes a
string or a function returning one, so drawings follow parameter changes.

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
lines, `print(d)` the signal flow and every block. Lines drawn or moved
after building, e.g. by `Subsystem.replace`, take effect at the next run.

### Drawing

In Jupyter a diagram shows itself as a graph: end a cell with `d`.
`d.draw()` draws it with matplotlib, `d.draw("rx.pdf")` also writes a file
(any matplotlib format). Blocks show their name and type; `Sum` and
`Product` are circles with their signs, `Gain` a triangle with its gain,
`Const` a small box with its value, named only if you named them.
`info=True` adds each block's parameters. Feedback lines are dashed;
probes are tagged. `expand=True` opens subsystems, nested ones too: their
blocks sit in a dashed frame titled with name and type, in columns of their
own. `to_dot`, `to_mermaid` and `to_text` take `expand=True` as well
(clusters, subgraphs, block paths such as `fm/scale`).
`d.to_dot()` gives Graphviz DOT (`dot -Tpdf`, or Typst's `diagraph`),
`d.to_mermaid()` a Mermaid flowchart, `d.to_text()` what `print(d)` shows.
Conversions to another representation are all named `to_...`.

`p = d.probe("lp")` records any output without drawing a line to a Scope
(targets: a block, a port, `"name"`, `"name:port"`); `keep`, `every` as for
`Scope`. `d.probe("lp", sink=sl.SpectrumAnalyzer())` attaches any one-input
sink instead. `d.plot()` plots every Scope.

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
    rx.plot_psd(ax=ax2)


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
runs vectorised. Blocks can decide per chunk length with `feedthrough(L)`.

Other loops, e.g. through `Delay(1)` at chunk 1024, run sample by sample
inside the chunk as a loop cluster; the rest of the diagram stays
vectorised. With numba, a loop whose blocks all have compiled kernels runs
as one compiled function (`Gain`, `Sum`, `Product`, `FuncChunk` if numba can
compile it, `Delay`, `FIR`, `IIR`, `FuncSingle`, `ODE` with RK4,
`ContinuousLTI` with exact ZOH, and subsystems made of these, like
`mixer`), else in Python with a warning.
`d.loops()` lists the clusters and their `mode`. Results do not depend on
the chunk. A loop that no block delays raises `AlgebraicLoopError`.
`compile_loops="python"` never compiles; `compile_loops=False` raises for
every cluster.

### Sample rates

`resampling_ratio` is a block's output rate / input rate as a `Fraction`
(default 1). Sources run at their own `fs` or the diagram's; rates propagate
along the lines. A source inside a subsystem without its own `fs`, such as
the noise of `awgn` or the oscillator of `mixer`, takes the rate of the
signal it meets, so these work after any rate change. `d.rates()` lists them; chunk and run lengths are multiples of
`d.granularity`.

A block given `fs` is pinned to it: a diagram that feeds it at another rate
raises. Without one, `blk.fs` reads the rate of the last diagram it ran in,
and the next diagram may run it at another. Functions that compute
coefficients from the rate, such as `fm_mod(75e3, fs=fs)`, pin their blocks.

### Stopping early

`self.request_stop("reason")` inside a block ends the run after the chunk;
`d.stopped` gives the reason. The next run continues.

### Signal checks

On the first chunk after build or `reset()`, NaN/Inf and dropped imaginary
parts raise an error naming the first block at fault. Opt-in input checks:
`same_shape_inputs = True` (with `expands_scalars = True`, signals with one
value per sample may meet vector signals), `real_inputs = True`,
`same_kind_inputs = True` (all real or all complex; set on an instance to
check just that one). `check=True` checks every
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
| `Impulse(T=0, amp=1)` | one sample of `amp` at `T` / s |
| `Signal(fn)` | `fn(t)`, any vectorised function of time |
| `FromArray(data, loop=False)` | `data`, then zeros (or repeats) |
| `Sine(f, amp=1, phase=0, complex=False)` | phase-continuous `cos`, or `exp(j·)` |
| `Chirp(f0, rate=0, period=None)` | linear sweep, phase-continuous; with `period` a sawtooth sweep |
| `Noise(std=1, seed=0, complex=False, width=None)` | white Gaussian, reproducible; `width` channels, independent |
| `Pulses(width, period, delay=0)` | 1 for `width` every `period`, else 0 |
| `RandomBits(seed=0)` | random bits, 0 or 1: test data for links |
| `RandomPulses(rate, duration=0)` or `(p=)` | 1 during pulses starting at random (Bernoulli), else 0 |
| `MarkovChain(P, start=0)` | states of a Markov chain, one step per sample |

### Math

| block | output |
| --- | --- |
| `Gain(g)` | `g · x` |
| `Sum("+-")` | `x0 - x1`; one sign per input `x0, x1, ...`; one value per sample adds to every channel |
| `Product("*/")` | `x0 / x1`; one `*` or `/` per input, or a count; one value per sample scales every channel |
| `FuncChunk(fn, outputs=("y",))` | `fn(...)` on whole chunks; required arguments are inputs, defaulted ones parameters |
| `Delay(n=1, init=0, tau=None)` | `x` delayed by `n` samples or `tau` / s; loops through it run vectorised for chunks `<= n` |
| `Unwrap(skip=0)` | phase `x` without its 2 pi jumps, as `numpy.unwrap` of the whole signal; the first `skip` samples stay wrapped |

### Filters

| block | output |
| --- | --- |
| `FIR(taps)` | FIR filtered `x`; `.freqz()` |
| `IIR(sos=)`, `IIR(b=, a=)` | IIR filtered `x`; `.freqz()` |

### Functions as blocks

`FuncChunk(fn)` calls `fn` once per chunk on whole arrays: vectorised NumPy, the
fastest for what NumPy can do. `FuncSingle(f)` calls `f` once per sample on
scalars, written in plain Python with `if`/`else` and state, and compiles
the loop over the chunk with numba. Both read their ports and parameters
from the signature the same way: required arguments are inputs (or
`inputs=`), arguments with defaults are parameters (`params=`,
`blk.params`, changed between chunks), `outputs=` names several outputs.

```python
scale = sl.FuncChunk(lambda x, k=2.0: k * x)  # k is a parameter
scale.params["k"] = 3.0
```

A `FuncSingle` example:

```python
import math

import sigline as sl


def schmitt(x, s, hi=0.5, lo=-0.5):  # stateful: returns (y, new state)
    if x > hi:
        s = 1.0
    elif x < lo:
        s = -1.0
    return s, s


trigger = sl.FuncSingle(schmitt, state=-1.0)
trigger.params["hi"] = 0.7  # takes effect at the next chunk
clip = sl.FuncSingle(math.tanh)
```

With `state=`, the last required argument gets the state. The state is a
number, a tuple, an array or named: `sl.state(phase=0,
freq=0)`, read as `s.phase`, returned as `type(s)(...)`. Several outputs:
`outputs=("a", "b")`, return a tuple. `dtype=` declares output types; else
`f` runs once per chunk on a copy of the state to find them.
`jit="auto"` falls back to Python with a warning if numba cannot compile
`f`; `jit=False` runs Python for breakpoints, with the same results.

| function | block |
| --- | --- |
| vectorises with NumPy (`np.tanh`, `np.clip`) | `FuncChunk`: NumPy's kernels are fastest |
| scalar logic: `if`/`else`, piecewise, lookup | `FuncSingle` |
| memory: hysteresis, PLL, AGC, slew limit | `FuncSingle(state=...)` |

### Continuous

Continuous-time blocks hold each input sample over the sample interval
(zero-order hold) and integrate to the next sample. Their output at sample
`k` depends on inputs up to `k - 1`, so a loop through one needs no extra
delay: it runs as a loop cluster.

| block | output |
| --- | --- |
| `ODE(f, x0, g=None, method="RK45")` | `y = g(t, x[, *u])` with `dx/dt = f(t, x, *u)` |
| `ContinuousLTI(num, den)`, `ContinuousLTI(A=, B=, C=, D=)` | LTI system, exact for held inputs; `.bode(f)` |

`method` is any `solve_ivp` method, or `"RK4"`: fixed steps (`substeps`
per sample), compiled inside loops when numba compiles `f` and `g`.
`inputs=()` makes an autonomous system. `ContinuousLTI` discretises exactly
(matrix exponential, compiled with numba) and takes complex inputs; any
`solve_ivp` method instead integrates numerically.

```python
import numpy as np

import sigline as sl


def spring(t, x, u):  # mass on a spring, force u
    return [x[1], -4 * x[1] - 100 * x[0] + 100 * u]


mass = sl.ODE(spring, [0.0, 0.0], g=lambda t, x: x[0], method="RK4")
error = sl.Sum("+-")
sl.Step(0.1) >> error["x0"]
error >> sl.Gain(2.0) >> mass >> sl.Scope("position")
mass >> error["x1"]  # unit feedback, no Delay needed
```

### Multirate

| block | does |
| --- | --- |
| `Upsample(L)` | insert `L - 1` zeros per sample |
| `Repeat(L)` | hold each sample `L` times |
| `Downsample(M, phase=0)` | keep every `M`-th sample |
| `Interpolate(L, taps=None)` | upsample and low-pass, polyphase |
| `Decimate(M, taps=None)` | low-pass and keep every `M`-th, polyphase |
| `Resample(up, down)` | rational resampling, polyphase |
| `DDC(f, M)` | mix `f` to 0 Hz, low-pass, decimate by `M` |

### Mixing and analog modulation

These are functions returning subsystems of library blocks, with input
`x` and output `y`. Inner blocks have names, so parameters stay attributes:
the mixer is a `Sine` named `lo` and a `Product` named `mix`, and
`mix.lo.f = 2e3` retunes it. Modulators give complex baseband; demodulators
take it, optionally through a channel filter (`channel_width` / Hz,
two-sided). Functions that set up a filter or a scale from the sample rate
take `fs`.

| function | does | made of |
| --- | --- | --- |
| `mixer(f, lo="cos", phase=0)` | `x` times a phase-continuous LO: `cos`, `sin`, `exp`, `exp-` (an LO at `-f`) | `Sine`, `Product` |
| `am_mod(index)`, `am_demod(index, method)` | AM with or without carrier; envelope or coherent | `Gain`, `Const`, `Sum`; `FuncChunk`, `FIR` |
| `fm_mod(deviation, fs=)`, `fm_demod(deviation, fs=)` | FM; quadrature discriminator | `Gain`, `IIR` accumulator, `FuncChunk`; `Delay`, `FuncChunk`, `FIR` |
| `pm_mod(index)`, `pm_demod(index)` | PM; phase unwrapped across chunks | `Gain`, `FuncChunk`; `FuncChunk`, `Unwrap`, `FIR` |
| `ssb_mod(side)`, `ssb_demod(side)` | single sideband; FIR Hilbert, one-sided channel filter | `Delay`, `FIR`, `Gain`, `Sum`; `FIR`, `FuncChunk` |

### Digital modulation

Lower-case names are functions returning subsystems of library blocks.

| block or function | does | made of |
| --- | --- | --- |
| `WordsToBits(bits)`, `BitsToWords(bits)` | integer words and bits, MSB first | |
| `pam(M)`, `psk(M)`, `qam(M)` | Gray-coded constellations, unit average energy | |
| `mapper(points)`, `demapper(points)` | bits to symbols and back (nearest point) | `BitsToWords`, `FuncChunk`; `FuncChunk`, `WordsToBits` |
| `pulse_shaper(sps)`, `matched_filter(sps)` | root-raised-cosine shaping (`rrc_taps`) and its matched filter | `Interpolate`; `FIR` |
| `fsk_mod(M, sps, h, fs=)`, `FSKDemod(M, sps, h)` | continuous-phase FSK (MSK at `h=0.5`); noncoherent detection | `FuncChunk`, `Repeat`, `fm_mod`; block |

### Channels

Subsystems that add noise or interference to a signal, input `x`, output
`y`. Power is
`mean(|x|^2)`. A source cannot see the signal, so give `complex=True` for
complex baseband; the inner `Sum` raises on the first chunk if signal and
noise differ in kind. `width=` gives each channel of a vector signal its own
noise. Every random part takes a `seed`.

| function | does | made of |
| --- | --- | --- |
| `awgn(snr_db)`, `awgn(power=)`, `awgn(temperature=, fs=)` | white Gaussian noise; SNR optionally in a `bandwidth`; thermal `k T B` | `Noise`, `Sum` |
| `colored_noise(power, color)` | pink (1/f), brown (1/f²), blue, violet noise | `Noise`, `IIR`, `Sum` |
| `phase_noise(linewidth, fs=)` or `(dbc_hz=, offset=, fs=)` | oscillator phase noise (random walk, Lorentzian line); outputs `y`, `phi` | `Noise`, `IIR`, `FuncChunk`, `Product` |
| `impulsive_noise(power, rate, duration=)` | Gaussian noise in bursts starting at random | `Noise`, `RandomPulses`, `Product`, `Sum` |
| `burst_noise(p_gb, p_bg, power_bad)` | Gilbert-Elliott: noise power switching between a good and a bad state; outputs `y`, `bad` | `MarkovChain`, `FuncChunk`, `Noise`, `Product`, `Sum` |
| `tone_interferer(f, power, drift=0)` | a carrier, optionally drifting | `Chirp`, `Sum` |
| `chirp_interferer(f0, f1, period, power)` | a sweep from `f0` to `f1`, repeating | `Chirp`, `Sum` |
| `pulsed_interferer(f, power, width, period)` | a pulsed carrier (radar); `power` while on | `Sine`, `Pulses`, `Product`, `Sum` |
| `modulated_interferer(f, power, sps, points, fs=)` | another transmitter: random bits, mapped, RRC-shaped, at `f` | `RandomBits`, `mapper`, `pulse_shaper`, `mixer`, `Gain`, `Sum` |

### Scope

`Scope(name)` records its input: `.data`, `.t` (absolute time / s).
`keep=N` keeps the newest `N` samples, `every=M` records every `M`-th.
`.plot()` draws it over time (I and Q if complex), `.psd()` returns its
Welch PSD as arrays (no detrending: DC can be a carrier), `.plot_psd()`
plots that in dB. `.snapshot()` returns matching `(t, data)`, also while a
diagram runs in the background.

### Measurements

Sinks with bounded state, as blocks or as probes
(`d.probe("rx", sink=...)`). Their readings are safe to take while a diagram
runs in the background: each sees whole updates.

| block | measures |
| --- | --- |
| `SpectrumAnalyzer(nfft)` | running Welch PSD, max-hold; `.plot()` |
| `Spectrogram(nfft, keep)` | short-time spectra over time; `.plot()` |
| `PowerMeter(interval, unit)` | mean power in dB, dBm, dBFS; per interval |
| `ConstellationScope(reference)` | newest symbols, MER, EVM; `.plot()` |
| `EyeDiagram(sps, span=2)` | histogram of traces, newest traces; `.plot()` |
| `SNREstimator(method)` | SNR: blind `m2m4` or decision-directed `dd` (the MER of `ConstellationScope`) |
| `BERCounter()` | bit errors after automatic alignment; confidence interval |

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
- [`04_drawing`](notebooks/04_drawing.ipynb): diagrams as graphs inline, as
  text, in files, as Graphviz DOT and Mermaid, subsystems opened.
- [`05_measurements`](notebooks/05_measurements.ipynb): a QPSK link measured
  with probes against theory, and a live dashboard.
- [`06_analog_modulation`](notebooks/06_analog_modulation.ipynb): one message
  through AM, FM, PM and SSB, with noise and a neighbouring signal.
- [`07_digital_modulation`](notebooks/07_digital_modulation.ipynb):
  constellations, a shaped QPSK link, BER of five schemes against theory.
- [`08_subsystems`](notebooks/08_subsystems.ipynb): an upconverter as one
  block, retuned and refitted inside, an I/Q mixer, a nested receiver.
- [`09_func_single`](notebooks/09_func_single.ipynb): an amplifier curve, a
  Schmitt trigger, a PLL in one block, an AGC inside a compiled loop.
- [`10_continuous`](notebooks/10_continuous.ipynb): an ODE against
  `solve_ivp`, exact LTI systems, a PI controller on a nonlinear tank,
  compiled loops through continuous plants.
- [`11_pid_control`](notebooks/11_pid_control.ipynb): a DC motor under PI
  speed control: exact discrete theory, windup and anti-windup, a load
  torque, phase margins.
- [`12_state_space_control`](notebooks/12_state_space_control.ipynb): a
  pendulum on a cart balanced by LQR: linearisation, moving the cart, force
  limits, sample rate against discrete stability.
- [`13_cafe_queue`](notebooks/13_cafe_queue.ipynb): customers counted one
  by one in a café: a day with rushes, staffing, and the exact Markov-chain
  distribution.

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
