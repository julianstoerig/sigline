# Design

Proposal for the second version of sigline, written from scratch. It
replaces the design in [architecture.md](architecture.md), whose limits and
the comparison in [comparison.md](comparison.md) are its starting point.

Status: the goals are fixed and the decisions at the end are taken. The
rest of "Proposed model" is a recommendation that follows from them.

## Goals

1. Python first. A model is a plain script. Signals are NumPy arrays,
   parameters are attributes, sweeps are `for` loops. Results go straight
   into SciPy, pandas, matplotlib, python-control.
2. General purpose. The core knows nothing about radio, control or any
   other domain. Domains are libraries on top of it.
3. Solvers are a choice, not a property of a block: fixed-step (Euler,
   RK2, RK4, RK5), adaptive (RK23, RK45, DOP853), stiff (Radau, BDF,
   LSODA). Changing the solver changes no block.
4. Block diagrams in and out. Code that builds a model reads like the
   diagram; every model can be drawn.
5. Operators. `>>` draws lines, arithmetic on blocks and ports makes
   sums, gains and products.
6. Simple things are quick. A step response of a first-order lag takes
   about five lines and no setup.
7. Complex things are accurate. Coupled continuous dynamics, mixed
   continuous and discrete parts, multirate, events.
8. Little redundancy. One block per concept. No second adder, no second
   noise source.
9. Live. Samples arrive from a device or leave for one while the diagram
   runs and decodes.

Not goals: code generation for embedded targets, acausal physical
modelling (Modelica, Simscape), a control-analysis toolbox of its own, hard
real time. A graphical editor is not planned; the model stays inspectable
data so that one remains possible.

## Where the goals pull apart

- Goals 3 and 7 need an engine that owns time and integrates all
  continuous states together. The first version had neither.
- Goal 9 needs bulk processing. An RTL-SDR gives 2.4 MS/s; Python cannot
  call even one block per sample at that rate. Samples must move in
  chunks, or in compiled code.
- Goal 6 forbids ceremony: no mandatory solver setup, rate declarations
  or compile calls for a model that does not need them.
- Goal 8 limits the number of execution paths as well as blocks. Each
  block that has a per-sample, a chunked and a compiled form is three
  blocks to keep equal.
- Goal 2 and goal 9 compete for attention. A general core that is slow on
  sample streams will not be used for the SDR; a fast stream engine with
  continuous blocks bolted on is the first version again.

The model below settles these by defining behaviour per time instant and
treating chunks as a faster way to compute the same numbers.

## Proposed model

### Time

- The diagram owns time. Blocks never count their own.
- Discrete time is counted in integer ticks of exact rational rates.
  `t = n / fs` is computed, never accumulated.
- Continuous time exists only between ticks, inside the solver.
- A model without discrete blocks has no `fs`. The solver chooses its
  steps and outputs are recorded at solver steps or at requested times.

### Signals

- A discrete signal has a rate and a value at each of its ticks. Between
  ticks it holds.
- A continuous signal has a value at every time. Blocks without state
  pass on the kind of their inputs.
- Going from continuous to discrete is sampling; from discrete to
  continuous is a hold. Both are blocks: `Sample` and `Hold`. A line
  between a continuous output and a discrete input, or the reverse, is
  an error before the run that names the block to insert.
- `Sample` states the rate. `Hold` is zero-order by default; other holds
  are a parameter, not more blocks.
- A third kind of line carries messages: timestamped values that are not
  sampled signals (packets, decoded frames). The port model reserves
  them from the start; they are built with the live stage.
- Values are NumPy arrays: real or complex, scalar or with channel axes.
  In chunked execution axis 0 is time.

### Blocks

A block declares equations; the engine decides when to evaluate them.

- `initial()`: the state at the start.
- `output(t, x, u)`: outputs from state and inputs. No side effects.
- `update(t, x, u)`: returns the next discrete state, once per tick.
- `derivative(t, x, u)`: time derivative of the continuous state.
- Optional: `zero_crossing(t, x, u)` for events, `open` and `close` for
  devices.
- Direct feedthrough is declared per output and input, statically.

Consequences:

- All state belongs to the engine, discrete as well as continuous. A
  block object holds parameters only; its methods are functions of
  `(t, x, u)`.
- For continuous state this is forced: no general solver could otherwise
  step it, reject a step or integrate several blocks together. For
  discrete state it buys snapshots, restarts, loops solved by repeated
  calls and compiled regions without per-block code.
- A base class or decorator hides the state for simple blocks, so a
  one-function block stays one function.
- `output` and `derivative` run several times per step (solver stages,
  rejected steps). Recording, device output and state changes happen
  only in accepted steps.
- A block with only `output` is memoryless. Most user blocks are one
  function.
- Discrete blocks receive arrays with a time axis. A single tick is a
  chunk of length one, not another code path.

### Initial conditions

Initial conditions belong to state, never to transfer functions.

- Every block with state takes its initial value as `x_start`, default
  zero: integrators, state space, delays, filters, ODE
  blocks. No variant blocks with initial states or initial outputs.
- The value is given in the coordinates the block exposes. An
  integrator or a first-order lag takes its output, which is its state.
  A state-space block takes the state vector. `Delay(n)` takes the `n`
  values it outputs during its first `n` ticks.
- A transfer function has no states of its own; any initial state would
  refer to the realisation the block picks inside. It therefore allows
  zero, an initial output when it is first order, and the steady state
  for the input at `t = 0`. Anything else is written as state space or
  as an ODE, where the states have names.
- Initial outputs with their derivatives are not offered: with direct
  feedthrough the output jumps at `t = 0` and the values are ambiguous.
  States are continuous, so initial states never are.

Initial values come from three places, each overriding the one before:

1. The block's parameter. It is saved in the diagram file, and `reset()`
   returns to it.
2. An argument of one run, per block name. Sweeps over starting points
   need no rebuild.
3. A snapshot: the state at the end of an earlier run.

Starting at an operating point ("the motor already runs at 6 V") is the
diagram's job, not one block's. The engine solves `x' = 0` for
continuous state and `x[n + 1] = x[n]` for discrete state, for the whole
diagram at the inputs at `t = 0`. Several equilibria are possible; the
fixed-guess rule of algebraic loops applies, and a failure names the
blocks.

Conventions:

- `t = 0` is after a step: a source that steps at 0 has its new value at
  `t = 0`.
- Initial values are states at `t = 0`. Outputs at `t = 0` follow from
  them, so they are always consistent; loops at `t = 0` are solved like
  at any other time.

### Rates

- Each discrete block has a rate, a `Fraction` of the diagram's base
  rate. Rates propagate along lines; a contradiction is an error before
  the run.
- Rate-changing blocks declare their ratio. Inputs of one block share a
  rate; anything else needs an explicit rate transition.
- Lengths per chunk are computed once per chunk length, not per chunk.

### Execution

Before the run the engine sorts the diagram and splits it into regions:

- A discrete block that is in no loop runs once per chunk on whole
  arrays.
- Connected continuous blocks, with the memoryless blocks between them,
  form one region with one state vector and one solver. Solver steps end
  at every tick of a discrete signal entering or leaving the region,
  because held inputs jump there.
- A loop through discrete blocks runs tick by tick, compiled with numba
  where every member allows it.
- A loop with no delay and no state is an algebraic loop. The engine
  solves it; see below.

The rule that keeps this honest: any chunk length gives the same result,
for every block and every solver. For adaptive solvers this holds because
steps end on ticks and the step size is carried across them.

### Algebraic loops

Algebraic loops are solved, not refused.

- The sort finds them as cycles over direct-feedthrough paths. Declaring
  feedthrough per output and input keeps loops that are not real from
  being found.
- The engine cuts each loop at as few signals as possible. The unknowns
  are the values `z` on the cut lines; one pass through the loop's blocks
  gives `g(z)`, and the solution satisfies `z = g(z)`.
- Method: Newton on `z - g(z)` with a numerical Jacobian. It solves a
  linear loop in one step, whatever the loop gain. Plain fixed-point
  iteration diverges at loop gains above one, which is the normal case in
  control, so it is not the default.
- Every solve starts from a fixed guess: zero, or the loop's `guess=`
  setting. The solution is then a function of the inputs at that instant
  alone. The loop keeps no hidden state, a snapshot needs no saved guess,
  and a rejected solver step leaves no trace.
- In a discrete region the ticks of a chunk are therefore independent
  and the iteration runs on arrays. A tick that has converged is frozen
  while the others iterate, so each tick gets the iterations it would
  get alone and chunk invariance stays exact.
- In a continuous region the loop is solved at every solver stage, from
  the same fixed guess.
- Option per loop: a warm start from the previous solution, solved tick
  by tick. It follows a moving solution where the fixed guess fails and
  is still exactly invariant, but it is slow and makes the last solution
  part of the state.
- Not offered: a warm start from the previous chunk with the iteration
  on arrays. It is fast, but invariant only to the loop tolerance.
- A loop with memory (hysteresis) has state and is modelled with state,
  not by relying on where the iteration started.
- This works only because `output` has no side effects: the solver calls
  it as often as it needs.
- Errors, each naming the blocks of the loop: no convergence within the
  iteration limit, a singular Jacobian (no solution or many), and loops
  over signals that are not real or complex numbers (bits, integers,
  messages).

### Solvers

- One interface: advance the state from `t` to at most `t_stop`, report
  the step taken and an error estimate, offer dense output.
- Fixed-step explicit methods come from one Runge-Kutta routine and a
  table of coefficients. Euler, RK2, RK4 and a fifth-order method are
  four tables, not four solvers.
- Adaptive and stiff methods wrap SciPy's stepper classes (`RK23`,
  `RK45`, `DOP853`, `Radau`, `BDF`, `LSODA`) at first. Own versions only
  where compiled speed needs them.
- DOP853 is an explicit eighth-order method for smooth problems and tight
  tolerances. It is not a stiff solver. Stiff problems need `Radau`,
  `BDF` or `LSODA`, which need a Jacobian: numerical by default,
  user-supplied as an option.
- The default is chosen by the model: no solver without continuous state,
  RK45 otherwise.

### Events

- Planned in the interface from the start, implemented after the basic
  solvers work.
- An event is a sign change of a block's `zero_crossing`. The engine
  locates it within the step from the dense output, ends the step there
  and lets the block change state.
- Discrete blocks see an event at their next tick.

### Live runs

- A device block is a discrete source or sink with a buffer between the
  device's thread and the engine.
- One clock paces a diagram: a device, or the wall clock. A second device
  with its own clock needs a resampling buffer between them.
- Live runs use fixed-step solvers or none, so the work per chunk is
  bounded. Late chunks are counted and reported, not hidden.
- The chunk length sets latency against throughput and is the user's
  choice.
- Decoders produce data whose amount depends on the signal: packets,
  symbols after timing recovery. They travel on message lines.

### Building diagrams

- `a >> b` draws a line and returns `b`, so chains read left to right.
- Named ports by indexing: `mixer["lo"]`.
- Arithmetic builds blocks: `ref - plant` is a sum, `2 * x` a gain.
- Feedback needs no special syntax: blocks exist before their lines, so
  `err = ref - plant` may precede `err >> controller >> plant`.
- Every block is callable as a function on arrays, outside any diagram.
- A subsystem is a function that returns connected blocks under one name.
- `==` and hashing stay untouched: blocks must work as dictionary keys.

### Naming

Variables follow the state-space letters of control engineering.

| name | meaning |
| --- | --- |
| `t` | time in seconds |
| `n` | tick index, an integer; `t = n / fs` |
| `fs` | sample rate in hertz |
| `dt` | solver step in seconds |
| `x` | state, continuous or discrete |
| `u` | input of a block |
| `y` | output of a block |
| `z` | unknowns of an algebraic loop |
| `dx` | time derivative of the state, returned by `derivative` |

- Values at the start and the end of a run take the suffixes `_start`
  and `_stop`: `t_start`, `t_stop`, `n_start`, `n_stop`, `x_start`.
  `x_start` is the initial-condition parameter of every block with
  state.
- A run covers `t_start <= t < t_stop`, as Python's `range` does. A tick
  at `t_stop` belongs to the next run, so consecutive runs join without
  repeating a sample. The state at `t_stop` ends one run and starts the
  next.
- A block with one input calls it `u`, with one output `y`. Several
  ports are named by meaning (`mixer["lo"]`) or numbered from 0, like
  Python indexing: `u0, u1, ...` and `y0, y1, ...`.
- SciPy calls the state `y` (`solve_ivp(fun, t_span, y0)`). The solver
  wrappers translate and say so where they do.
- `n` is only the tick index; a filter's order is `order`.
- `s` and `z` as transform variables appear in formulas only, never in
  code.
- The library claims no other letters. `k`, `r`, `e` and `d` stay free
  for gains, references, errors and disturbances in user code.

### Layers and redundancy

- Core: engine, solvers, lines, subsystems, drawing, and the primitives
  that cannot be built from others (sources, sum, gain, product,
  function, delay, integrator, linear system, filter, rate changers,
  sample, hold, scope).
- Libraries: control, communications, channels, measurement. They use
  only the public interface of the core. If a library needs a private
  hook, the core is missing something.
- One block per concept, with variants as parameters. One block serves
  real and complex, scalar and vector signals.
- What can be connected from existing blocks is a function returning a
  subsystem, never a new block.
- Design is SciPy's job. Blocks take coefficients, `scipy.signal`
  systems or python-control systems; sigline designs no filters and
  tunes no controllers.
- Analysis is exported: a diagram linearises to a state-space object
  that SciPy and python-control accept.
- A new block must name the existing blocks it could not be built from.

### Serialisation

The script is the source of a model. A file is an export of what the
script built, as plain text that can be compared and edited.

- A diagram file holds blocks and lines. A block is its type, as an
  import path, and its constructor arguments. A line is a pair of ports.
  A format number comes first.
- This is complete because blocks hold parameters only. The base class
  records the constructor arguments, so a block author writes nothing
  for it.
- Loading imports each type, calls it with the stored arguments and
  draws the lines again.
- Blocks made by operators are saved like any other. Automatic names
  come from the block's place in its diagram, not from a counter shared
  by the process, so the same script always writes the same file.
- State is not in the diagram file. A snapshot of a run is the engine's
  state per block name and the time, in a file of its own. A run
  continues exactly from it.
- Functions are stored by name, never by content: `numpy.clip`,
  `aircraft.aero`, with their keyword arguments. The file is complete
  only together with the code it names, and records a hash of each user
  module so that loading can warn when the code has changed.
- Data is stored as data. Short arrays go into the text. Long ones
  (filter taps, lookup tables, recordings) go into an `.npz` beside the
  file, which refers to it. A model whose unknown function is a table is
  the one case where the files hold everything.
- A lambda, a closure or an object that exists only in memory has no
  name to store. Such a block runs and draws, but saving the diagram
  raises an error that names the block. No incomplete file is written.
  The remedy is to give the function a name in a module, or to turn it
  into data.
- A subsystem is saved as a named group: its blocks and the port names
  it exposes. Loading rebuilds it, nested ones included, so a loaded
  diagram draws and connects like the original.
- The call that made a subsystem is not saved. The file says what the
  subsystem consists of, not `PI(kp=2, ki=15)`: there is one description
  of every block and nothing that could disagree with it.
- Loading runs the imports the file names and nothing else: no pickle,
  no evaluation of text from the file.

Two parts affect the first commits: how blocks record their constructor
arguments, and how automatic names are made. The format, snapshots and
loading wait until the model is stable.

## Things to be careful about

Engine:

- Purity of `output` and `derivative`. One hidden side effect breaks
  adaptive solvers, events and linearisation at once. Test every block by
  calling them twice.
- Float time. `t += dt` drifts; ticks are integers.
- Held inputs are discontinuities. A solver that steps across one loses
  its order silently.
- Chunk invariance with adaptive steps. It is easy to lose and hard to
  notice; it needs a test from the first solver on.
- Algebraic loops make results depend on a tolerance. It must sit well
  below the solver's.
- The fixed guess costs iterations on nonlinear loops, at every solver
  stage in a continuous region, and Newton may fail once the solution
  has moved far from it. The error must name the loop and point to
  `guess=` and the warm start.
- A loop may have several solutions (a comparator with feedback). The
  fixed guess picks the one it is nearest to, and the output can jump
  between them as the inputs move. Document it, do not hide it.
- Exact invariance needs the per-tick stopping rule as well as the fixed
  guess. Test loops with `assert_chunk_invariant` for equality, not to
  a tolerance.
- A loop solved at every solver stage multiplies the cost of a
  continuous region. Report the loops found, so a user can break one
  with a delay or a small lag on purpose.
- Overhead per block and chunk. The first version spent most of a step
  on bookkeeping with `Fraction`s.

Structure:

- No global state: no shared edit counters, no name counters that depend
  on what ran before.
- A diagram is checked and fixed at an explicit point. Edits during a run
  are queued or refused.
- The engine never tests for concrete block types. Compiled kernels,
  drawing symbols and checks are found through the block's interface.
- An error inside a step must leave the diagram in a state that can
  continue or says that it cannot.

Operators:

- Precedence: `>>` binds weaker than `+` and `*`, stronger than
  comparisons. `a >> b + c` connects `a` to the sum.
- Blocks made by operators need names and a place in drawings.
- NumPy must defer to blocks (`__array_ufunc__ = None`), or `array * block`
  becomes an array of blocks.
- Blocks with several outputs or inputs need one rule for what `>>`
  connects, and an error for everything ambiguous.

Live:

- Threads: device callbacks, the engine, plots. Shared data crosses in
  buffers only.
- Buffers overflow and run empty. Both are counted and visible.
- Transmit timing is stricter than receive timing: a late chunk is a gap
  on the air.
- Continuous blocks at megahertz tick rates cost one solver step per tick
  at least. Live signal paths should be discrete.

Scope:

- The general core comes first and stays small. The SDR path is its
  hardest client, not its design centre.
- Speed work (numba regions, own adaptive solvers) starts only when a
  measured example needs it.
- Each stage ends with examples that run, so the library is usable long
  before it is complete.

## Decisions

1. State: engine-owned for discrete and continuous state alike.
   `update` returns the next state; blocks hold parameters only.
2. Chunks: block methods take arrays with a time axis from the first
   commit. The first engine is simple and may use chunks of one. Chunked
   execution comes as a second scheduler, tested against the first.
3. Sampling and hold: explicit `Sample` and `Hold` blocks. A direct line
   between a continuous and a discrete block is an error.
4. Data that is not a sampled signal travels on message lines, reserved
   in the port model now and built later.
5. Algebraic loops are solved, from a fixed guess, with a warm start
   tick by tick as an option per loop. Method: `scipy.optimize.root` on
   the cut signals first, an own Newton iteration on arrays once chunked
   execution exists; accelerated fixed-point iteration as an option for
   loops without a usable Jacobian.
6. Solvers: one own fixed-step Runge-Kutta routine with coefficient
   tables; SciPy's steppers for adaptive and stiff methods. Own versions
   only when a measurement asks for them.
7. The name stays `sigline`.
8. Models are built in code; no graphical editor is planned. A diagram
   stays serialisable data: blocks, ports, lines, parameters.
9. Initial conditions belong to state, given per block, per run or from
   a snapshot; transfer functions allow only zero, a first-order initial
   output and the steady state. The parameter is `x_start`.
10. Naming: the state-space letters `t, n, fs, dt, x, u, y, z, dx`;
    the suffixes `_start` and `_stop` for the ends of a run and for start
    values; `t_stop` is excluded from a run. Numbered ports count from
    0.

What they cost:

- Explicit `Sample` and `Hold` add blocks to every mixed model. The
  error for a missing one must make the fix obvious, or goal 6 suffers.
- Serialisable diagrams and blocks taking arbitrary functions pull apart.
  A block built from a lambda can be drawn and run but not saved; see
  "Serialisation".
- Engine-owned state makes a block's attributes parameters by
  definition. A block that caches something in `self` during `update`
  breaks snapshots silently; test it by running from a restored state.
- Two schedulers must agree. The simple one is the reference and stays
  in the code base as the test oracle.

## Order of work

Each stage is defined by scripts that must run, written before the code.

1. Lines, operators, memoryless blocks, one fixed-step solver, scope.
   Example: step response of a first-order lag in five lines, against
   the closed form.
2. Joint integration and solver choice. Examples: an oscillator split
   over two blocks, with energy conserved to tolerance; a stiff problem
   (Robertson, Van der Pol) with `Radau`.
3. Algebraic loops. Examples: `y = r - k y` for `k` below and above one,
   against `r / (1 + k)`; a loop through a nonlinearity, against
   `scipy.optimize.brentq`; the same loop inside a continuous region.
4. Discrete blocks, rates, hold and sampling. Example: a continuous
   plant with a discrete PID, against a python-control reference.
5. Chunked execution and the invariance test. Example: a filter chain on
   a recorded IQ file, equal for every chunk length.
6. Loops tick by tick, then compiled. Example: a phase-locked loop.
7. Multirate. Example: a decimating receiver front end.
8. Live devices. Example: receive, demodulate and play FM from an
   RTL-SDR; then decode a digital signal.
9. Events. Example: a bouncing ball; a relay with hysteresis.
10. Subsystems, drawing, linearisation, the libraries.

Drawing can move earlier if it helps debugging; events can move earlier if
control work needs them before the SDR does.
