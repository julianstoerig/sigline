# Comparison with Simulink, PathSim and python-control

How sigline's design compares with three other tools for simulating
interconnected systems. sigline here means its first version, described in
[architecture.md](architecture.md). The analyses of the other three follow
the comparison, each on the same fourteen points.

Versions: PathSim 0.32.1 and python-control 0.10.2, both read from source,
not run. Simulink from the MathWorks documentation. No tool but sigline was
timed, so speed is compared by mechanism, not by measurement.

## The four designs in one sentence each

- sigline: a static schedule runs blocks on whole chunks of samples at
  exact rational rates; only feedback loops run sample by sample.
- Simulink: an engine owns time, states and solvers, and calls every
  block's outputs, update and derivatives in sorted order at each step.
- PathSim: one global timestep loop in Python, each block with its own
  integrator, loops closed by fixed-point iteration, events in the step
  control.
- python-control: systems are algebraic objects; an interconnection is one
  more system, simulated by one call to an ODE solver.

## Side by side

| aspect | sigline | Simulink | PathSim | python-control |
| --- | --- | --- | --- | --- |
| time | implicit: sample counts at fixed rates | explicit: major and minor steps | explicit: global `dt`, fixed or adaptive | explicit: one solver call per run |
| signal | NumPy array per chunk, real or complex, vector axes | value per step with type, size, complexity, rate; frames per block | float64 scalar per channel | real arrays inside an interconnection |
| block author writes | `step`, or `output` and `update`; `feedthrough(L)` | outputs, update, derivatives, zero crossings; feedthrough per port | `update`, `step`, `solve`; feedthrough per block (`__len__`) | `updfcn`, `outfcn`; feedthrough undeclared |
| who integrates | each `ODE` block alone, inputs held over a sample | the engine, all continuous states together | each block's solver, kept in step by the simulation | `solve_ivp` over the whole state vector |
| ordering | topological sort per chunk length, cached | sorted lists per method and per rate | depth order of algebraic blocks | none: repeated sweeps until inputs settle |
| algebraic loops | error; loops need a delay of one sample | solved each step (trust region, line search) | solved by Anderson-accelerated fixed point | `RuntimeError` at run time; linear ones closed by algebra |
| feedback with delay | vectorised if the delay spans the chunk, else a per-sample cluster, compiled with numba | per step, as everything | per step, as everything | part of the state equations |
| rates | exact `Fraction`s, propagated both ways, checked | sample times inferred and checked; rate transitions | a period per discrete block, unchecked | one timebase per interconnection |
| multirate blocks | polyphase interpolate, decimate, resample, DDC | yes (DSP System Toolbox) | none | none |
| events, zero crossings | none | located by the solver | located inside adaptive steps | none |
| stiff systems | `solve_ivp` methods per block | implicit variable-step solvers | ESDIRK, BDF, GEAR | `solve_ivp` methods |
| speed mechanism | vectorised chunks, numba for loops | accelerator, generated C, frames | none in the package | closed-form LTI responses, SLICOT |
| hierarchy | subsystems flat at run time | virtual and atomic subsystems, model references, buses | subsystems with their own graph and loop solver | nested interconnections |
| linearisation | none | yes (Simulink Control Design) | `linearize`, `to_statespace`, steady state, periodic steady state | `linearize`, operating points, full linear toolset |
| chunk or stream operation | core contract, tested per block | frames; state kept by the engine | per step; streaming generators | none: batch only |
| hardware, real time | RTL-SDR, sound card; soft pacing | code generation, HIL targets, SDR packages | real-time pacing, no devices | none |
| model as text | Python script | zipped XML | Python script | Python script |
| licence | MIT | proprietary, paid | MIT | BSD-3-Clause |

## Upsides and downsides

### Simulink

Upsides:

- Block equations are separate from scheduling and integration. One block
  works with any solver, in accelerated runs and in generated C.
- One semantics covers continuous, discrete, multirate and event-driven
  parts of one model.
- Types, sizes, complexity and rates are inferred and checked before the
  run.
- Algebraic loops are solved, zero crossings located, rate transitions
  deterministic.
- The same model goes to embedded code and hardware-in-the-loop tests.
- The block libraries, verification tools and documentation are the
  largest of the four by far.

Downsides:

- Price: about EUR 6,050 per seat for MATLAB and Simulink at German list
  prices, toolboxes extra.
- Closed source. Engine behaviour is known from documentation only.
- Models are zipped XML: no useful `git diff`, merges need a graphical
  tool.
- Scripting is indirect (`add_block`, `set_param` with strings); Python
  only through the MATLAB Engine.
- Every run compiles the model. Acceleration pays off for runs of minutes.
- Sample-based DSP is slow; frames fix that at the price of latency, set
  block by block.
- Desktop real time ends near 20 kHz, far below SDR sample rates.

### PathSim

Upsides:

- A small, readable engine in pure Python on numpy, scipy and matplotlib.
- The widest solver choice of the open tools: 31 integrators, with stiff
  and variable-step methods.
- Events are part of step control: zero crossings, conditions and
  schedules.
- Algebraic loops are solved without user action.
- Linearisation, steady state and periodic steady state come built in.
- Blocks, connections and parameters can change during a run.

Downsides:

- Python cost per sample, for every block and connection at every solver
  stage. A chain at 1 MS/s needs 10^6 global steps per simulated second.
- No sample-rate bookkeeping and no rate-changing blocks.
- Signals are real float64 scalars: no complex baseband, no bit types.
- No vectorisation over time, JIT or code generation in the package.
- Feedthrough per block, not per port, so false loops cost iterations.
- Implicit solves use block-local Jacobians; stiffness spread over many
  blocks may converge slowly.
- The documentation still advertises automatic differentiation that the
  code no longer has.

### python-control

Upsides:

- Systems are values. A closed loop is a model that every analysis and
  design function accepts.
- The linear toolset is the broadest of the open tools: margins, Nyquist,
  root locus, LQR, H2 and H-infinity, model reduction.
- LTI time responses are exact (matrix exponential), not integrated.
- The path from nonlinear model to operating point, linearisation, design
  and nonlinear simulation is complete.
- Connection by signal name keeps control loops short.
- Optimal control, MPC, flatness and describing functions share the
  object model.

Downsides:

- No scheduler and no declared feedthrough: each evaluation sweeps all
  subsystems up to n+1 times in Python.
- Algebraic loops surface as an error during the run, depending on signal
  values.
- One timebase per interconnection: no hybrid, no multirate, no
  sample-and-hold.
- Batch only: no state between calls, no chunks, no streams, no real time.
- No events, and transport delay only as a Padé approximation.
- Real-valued internals: complex baseband cannot pass an interconnection.
- No DSP or communications blocks.

### sigline, for reference

Upsides:

- Whole chunks through NumPy and SciPy; per-sample cost only inside
  feedback loops, and compiled there.
- Results do not depend on the chunk size, tested per block.
- Exact rational rates with polyphase resampling.
- Complex and vector signals, communications blocks, channel models,
  measurement sinks.
- Background runs paced by the clock or by a device.

Downsides:

- No global continuous solver, no events, no algebraic-loop solver.
- No linearisation or frequency response of a diagram.
- Static rates and fixed output lengths per chunk.
- Compiled loops cover a fixed set of block types.
- Single-threaded, soft real time.

## What the comparison shows

- The four split along one axis: who owns time. Simulink and PathSim step
  a global clock and pay per step; sigline has no clock, only sample
  counts, and pays per chunk; python-control hands the whole run to a
  solver. Speed on sample streams and accuracy on coupled continuous
  dynamics pull in opposite directions along it.
- For DSP and SDR work only sigline and Simulink with frames process
  samples in bulk, and only they track rates. PathSim and python-control
  are not built for that workload.
- For continuous and hybrid dynamics the order reverses: Simulink, then
  PathSim, then python-control, then sigline, which couples continuous
  blocks through a one-sample hold.
- For analysis python-control leads among the open tools, PathSim covers
  the basics, sigline has per-block responses only.
- Feedthrough information decides what can be known before the run:
  per port (Simulink), per chunk length (sigline), per block (PathSim),
  none (python-control).

Ideas the other tools suggest for sigline, none decided:

- Linearisation of a diagram to a `scipy.signal.StateSpace`, or export of
  LTI parts to python-control, instead of an analysis toolset of its own.
- An algebraic-loop solver by fixed-point iteration inside loop clusters,
  as in PathSim.
- Event location in `ODE` through the `events` argument of `solve_ivp`.
- A joint integrator for directly connected continuous blocks.
- Variable output lengths per chunk for timing recovery and packet work.

## Simulink

Facts come from MathWorks documentation (sources at the end). Items marked (gk) come from general knowledge and were not checked against a source.

### Purpose and scope

- Graphical environment for modelling, simulating and generating code for dynamic systems: continuous, discrete, multirate and hybrid.
- Runs inside MATLAB. Models are block diagrams edited in the Simulink Editor and saved as `.slx`.
- Domains come from add-on products: DSP System Toolbox, Communications Toolbox, Stateflow (state machines), Simscape (acausal physical networks, gk), Simulink Control Design.
- Aimed at model-based design: one model goes from simulation to production C code (Embedded Coder) and hardware-in-the-loop tests (Simulink Real-Time).

### Model of computation

- Time-driven. The engine advances simulated time in steps; at each step it computes block outputs, then states.
- Each block is a hybrid system with inputs, outputs, continuous states (integrated by the engine) and discrete states (updated at sample hits).
- A signal is the time-varying value carried by a line, not a buffer. Discrete outputs hold between sample hits (gk).
- Every signal has compile-time attributes: data type, dimensions (scalar, vector, matrix, N-D), complexity, sample time. Variable-size signals may change size but not number of dimensions.
- Sample time says when a block computes outputs and updates state. Encoded as `[Ts, To]`: continuous `[0 0]`, fixed in minor step `[0 1]`, discrete `[Ts To]`, variable `[-2 Tvo]`, inherited `[-1 0]`, constant `[inf 0]`; also asynchronous, triggered, union and dataflow.
- Sample-based by default: one value per channel per hit. Frame-based: DSP and comms blocks read an M-by-N matrix as N channels of M samples ("Columns as channels"); frame period is M times the sample period.
- Frame status used to be a signal attribute; since about 2012 it is a per-block "Input processing" parameter.

### Block interface

- Built-in blocks are part of the closed engine. User blocks are C/C++ MEX S-functions, Level-2 MATLAB S-functions, MATLAB System blocks (System objects) or MATLAB Function blocks.
- S-function callbacks: `mdlInitializeSizes` (ports, state counts, work vectors, feedthrough), `mdlInitializeSampleTimes`, optional `mdlSetInputPortWidth`-style hooks for size propagation, `mdlStart`, `mdlInitializeConditions`, `mdlOutputs`, `mdlUpdate`, `mdlDerivatives`, `mdlZeroCrossings`, `mdlGetTimeOfNextVarHit`, `mdlTerminate`, `mdlRTW`.
- `mdlOutputs` runs at major and minor steps; `mdlUpdate` once per major step for discrete states; `mdlDerivatives` returns `dx/dt` and the engine integrates; `mdlZeroCrossings` supplies signals the engine brackets.
- The engine calls one method on every block in sorted order before moving to the next method.
- The engine owns state memory. A block declares counts of continuous and discrete states and work vectors; the engine allocates them in the link phase.
- Direct feedthrough is declared per input port (`ssSetInputPortDirectFeedThrough`, default 0). It drives sorting and algebraic-loop detection; a wrong flag gives wrong results or failed convergence.
- Ports may be dynamically sized; sizes are fixed at compile time for the whole run.
- System objects implement `setupImpl`, `stepImpl`, `resetImpl`; or `outputImpl` plus `updateImpl` with `isInputDirectFeedthroughImpl` returning false to break loops. State lives in `DiscreteState` properties; parameters are tunable or `Nontunable`.

### Connections and hierarchy

- Blocks have numbered input and output ports. A line connects one output to one or more inputs (branches); one input takes exactly one line (gk).
- Virtual subsystems are graphical only; they are flattened at compile and their blocks sorted in the parent.
- Nonvirtual (atomic) subsystems execute as one unit with their own sorted order. All conditional subsystems are atomic.
- Conditional subsystems: enabled (hold or reset states and outputs), triggered, enabled-and-triggered, function-call, if/switch-case action, resettable. Iterators: for, while, for-each.
- Model reference: a Model block includes another model file. Each referenced model compiles independently, runs in normal or accelerator mode, takes instance parameters, and builds incrementally. Protected models hide IP.
- Libraries hold reusable blocks; copies are links to the library. Masks add parameter dialogs and icons to any block or subsystem.
- Virtual buses group lines without memory and may mix sample rates. Nonvirtual buses are structs in memory and C code, need a `Simulink.Bus` object and one sample time.

### Compilation phase

- Runs at every simulation start ("update diagram"): evaluate parameter expressions in MATLAB workspaces, propagate signal attributes, resolve sample times, reduce blocks, flatten virtual subsystems, sort.
- Attribute propagation fills unspecified data types and dimensions from sources downstream and checks each block accepts its inputs.
- Sample-time propagation repeats forward pass, backward pass and heuristics until all rates are known. An inherited block with several rates takes the fastest if all are integer multiples, else the GCD (fixed-step) or fixed-in-minor-step (variable-step). Sources set to inherited may take the rate of the block they drive.
- Sorting is task-based: blocks are grouped by sample time into tasks, and each task is sorted by direct-feedthrough dependencies. Block priorities are honoured only if consistent with data dependencies.
- A cycle of direct-feedthrough blocks is an algebraic loop; the diagnostic can be none, warning or error. Atomic subsystems create artificial loops because all their inports count as direct feedthrough.
- Link phase allocates signal, state and work memory, then builds method execution lists from the sort.

### Execution

- Each major step: Outputs on all blocks in order; Update for discrete states; Derivatives for continuous states, with minor steps that call Outputs and Derivatives again; optional zero-crossing location; compute next step time.
- Variable-step solvers adapt step size to error tolerance and events: ode45, ode23, ode113 (nonstiff), ode15s, ode23s, ode23t, ode23tb (stiff/DAE), daessc (gk for parts of this list). An auto mode picks one.
- Fixed-step solvers (ode1 to ode8, ode14x, ode1be; gk for list) are the only option for generated code.
- Discrete-only models use a discrete solver and skip integration; time jumps from hit to hit.
- Algebraic loops are solved every step by a gradient-based dogleg trust-region or line-search method. It needs smooth, double-valued signals; models with algebraic loops cannot generate code.
- Multirate models run on one time line. Single-tasking runs all rates as one task; multitasking makes one task per rate, fastest rate highest priority. Simulation assumes zero execution time, so preemption only matters in generated code.
- Rate Transition blocks guarantee data integrity and optionally deterministic transfer between rates. Simulink can insert them automatically.
- Zero-crossing detection: registered signals are checked for sign change after each step, then bracketed to locate the event. An adaptive algorithm limits chattering. Fixed-step zero-crossing correction exists since R2022a.
- Conditional subsystems run only on enable, trigger or function-call events. Stateflow charts wake on time steps or events and optionally take several transitions per step (super-step).

### Performance techniques

- Normal mode interprets the block methods. MATLAB Function and Stateflow blocks are compiled even here.
- Accelerator mode builds an in-memory JIT engine by default (no C compiler needed), or a MEX file in classic mode. It runs in the MATLAB process.
- Rapid accelerator builds a separate executable holding solver and model, linked to Simulink by inter-process communication. Some features are restricted (scopes for some subsystems; algebraic loops and pause in older releases).
- Acceleration pays off only for runs of minutes. MathWorks' fuel-system example: normal 130 s, rapid accelerator 77 s with rebuild, 43 s with an up-to-date target.
- Fast restart compiles once and reuses an initialised snapshot. Only tunable parameters, inputs and workspace values may change; structure, sample times, sizes and types may not.
- `parsim` runs `Simulink.SimulationInput` sweeps in parallel (needs Parallel Computing Toolbox, else serial).
- Frame-based processing amortises per-call overhead over M samples at the cost of M samples latency. Referenced models build in parallel and share cache files.

### Analysis features

- Simulink Control Design linearises block by block: each block has an exact analytic linearisation, results are combined, structurally nonminimal states are removed. Numerical perturbation (forward or central differences) is the alternative.
- Linear analysis points select the part to linearise. Multirate models get a choice of rate-conversion method.
- Operating points come from trimming (`findop` with `operspec`, optimisers such as gradient descent), simulation snapshots or triggered events. Steady State Manager is the GUI. Base Simulink also has `linmod` and `trim` (gk).
- `frestimate` estimates frequency response by simulation with sinestream, chirp or PRBS injection. It serves models that do not linearise, such as PWM or triggered subsystems.
- Verification: Model Verification blocks (Assertion and checks) report per step; Simulink Test adds harnesses and a Test Manager; Simulink Design Verifier adds formal property proving and test generation (gk for SLDV detail).

### Extensibility

- S-functions in C, C++, Fortran or MATLAB. Legacy Code Tool and S-Function Builder wrap existing C code.
- A TLC file inlines an S-function into generated code; MATLAB and Fortran S-functions must be inlined to generate code.
- MATLAB Function block: a MATLAB function restricted to the code-generation subset. Extrinsic calls run in MATLAB during simulation and are dropped from generated code.
- MATLAB System block hosts System objects, shared with MATLAB scripts.
- Masks, libraries with lockable links, and subsystem references package reusable blocks.
- Models can be built and changed by script: `new_system`, `add_block`, `add_line`, `set_param`; run with `sim`, `parsim`, `batchsim`.
- Python reaches Simulink only through the MATLAB Engine API or packages built with Simulink Compiler; stepping a model from Python is not well supported.
- Simulink Compiler exports simulations as standalone executables, web apps or FMUs.

### Hardware and real-time I/O

- Simulink Coder generates C/C++ from models, Stateflow charts and MATLAB functions, for real-time and non-real-time targets. External mode tunes parameters and shows signals of running code (XCP).
- Embedded Coder adds production code: MISRA C, AUTOSAR, ASAP2, SIL/PIL back-to-back tests, traceability for DO-178, IEC 61508 and ISO 26262.
- Simulink Real-Time runs models on Speedgoat target computers for rapid control prototyping and HIL, with CAN, EtherCAT, Modbus, ARINC and FPGA I/O; HDL Coder targets the FPGA for higher rates.
- Simulink Desktop Real-Time adds a real-time kernel on Windows, Linux and macOS. Connected IO mode runs the model in MATLAB and only drivers in the kernel, up to about 1 kHz without guarantee. Kernel mode runs generated code, up to about 20 kHz.
- SDR support packages provide device blocks, e.g. RTL-SDR Receiver: frame-based complex output, 1024 samples per frame by default, burst mode, lost-sample and latency ports. USRP and ADALM-Pluto have similar packages (gk).

### Visualisation and tooling

- Graphical editor with overlays for sample-time colours and execution order.
- Scopes and DSP/comms viewers (spectrum analyser, constellation, eye diagram; gk). Simulation Data Inspector stores, plots and compares logged runs.
- Simulink debugger, Solver Profiler and Simulink Profiler help with debugging and speed (gk).
- `.slx` is an Open Packaging Conventions zip of XML parts; legacy `.mdl` is a single text file. The Comparison Tool (`visdiff`) diffs and merges models graphically.
- Model Advisor checks modelling standards; requirements traceability via Requirements Toolbox (gk).

### Licence, dependencies, maturity

- Proprietary and closed source. Requires MATLAB; most domain features need further paid toolboxes. Runs on Windows, Linux and macOS (gk).
- German list prices, excluding tax (fetched November 2026), perpetual / annual: MATLAB EUR 2,410 / 964; Simulink 3,640 / 1,455; Simulink Coder 3,640 / 1,455; Stateflow 3,440 / 1,375; DSP System Toolbox 1,475 / 590; Communications Toolbox 2,065 / 826; Simulink Control Design 1,770 / 708; Desktop Real-Time 2,950 / 1,180.
- Standard, startup, academic, student and home licence types exist; student and home prices were not verified.
- Released in 1990. Two releases per year, R20xxa and R20xxb (gk). De facto standard in automotive and aerospace control engineering (gk).

### Strengths

- One hybrid semantics covers continuous, discrete, multirate and event-driven parts in the same diagram.
- Blocks only declare equations (outputs, update, derivatives, zero crossings); the engine owns states, integration and scheduling. The same block works with any solver and in generated code.
- The compile phase checks types, sizes and rates before the run and infers most of them, so models need few explicit annotations.
- Feedthrough flags give a static execution order and early algebraic-loop detection; loops are solved instead of rejected.
- Rate transitions and tasking are explicit and deterministic, and map directly to real-time task code.
- Zero-crossing location gives accurate event timing with variable-step solvers.
- One model serves simulation, accelerated simulation, embedded C, FPGA code and HIL tests.
- As a product: large block libraries, qualified code generation, linearisation and trim tools, test and verification tooling, long-term vendor support and documentation.

### Weaknesses

- Cost: MATLAB plus Simulink is about EUR 6,050 per perpetual seat before any toolbox; a DSP/comms/real-time setup multiplies that.
- Closed source: engine behaviour is known only from documentation. Solvers, sorting and block internals cannot be read, patched or debugged.
- Poor text diffability: `.slx` is zipped XML, so plain `git diff` is useless and merges need the graphical Comparison Tool.
- Scripting friction: models built by `add_block`/`add_line` with string paths and `set_param` string values; Python access only through the MATLAB Engine.
- Compile overhead on every run. Acceleration pays off only for runs of minutes; fast restart forbids structural, size and rate changes. MATLAB startup itself takes seconds (gk).
- Normal mode has per-block, per-step interpretive overhead. DSP models need frame-based processing for speed, which adds latency and is set block by block.
- The algebraic-loop solver needs smooth double signals and blocks code generation.
- Many interacting modes and options (normal, accelerator, rapid accelerator, fast restart, tasking, solver settings) with feature restrictions per mode.
- Real-time needs extra products and often dedicated hardware; Desktop Real-Time stops near 20 kHz, far below SDR sample rates, so SDR work relies on frames and buffering.
- Vendor lock-in: models are usable outside MATLAB only as generated code or FMUs.

### Sources

- https://www.mathworks.com/help/simulink/ug/simulating-dynamic-systems.html
- https://www.mathworks.com/help/simulink/sfg/how-the-simulink-engine-interacts-with-c-s-functions.html
- https://www.mathworks.com/help/simulink/sfg/s-function-callback-methods.html
- https://www.mathworks.com/help/simulink/sfg/s-function-concepts.html
- https://www.mathworks.com/help/simulink/sfg/sssetinputportdirectfeedthrough.html
- https://www.mathworks.com/help/simulink/ug/what-is-sample-time.html
- https://www.mathworks.com/help/simulink/ug/types-of-sample-time.html
- https://www.mathworks.com/help/simulink/ug/how-propagation-affects-inherited-sample-times.html
- https://www.mathworks.com/help/simulink/ug/controlling-and-displaying-the-sorted-order.html
- https://www.mathworks.com/help/simulink/ug/algebraic-loops.html
- https://www.mathworks.com/help/simulink/ug/types-of-solvers.html
- https://www.mathworks.com/help/simulink/ug/zero-crossing-detection.html
- https://www.mathworks.com/help/simulink/ug/zero-crossing-detection-with-fixed-step-simulation.html
- https://www.mathworks.com/help/rtw/ug/tasking-modes-and-execution-order.html
- https://www.mathworks.com/help/rtw/ug/rate-transitions.html
- https://www.mathworks.com/help/simulink/ug/signal-dimensions.html
- https://www.mathworks.com/help/simulink/ug/variable-size-signal-basics.html
- https://www.mathworks.com/help/dsp/ug/sample-and-frame-based-concepts.html
- https://blogs.mathworks.com/simulink/2012/01/04/frames-from-signals-to-blocks/
- https://www.mathworks.com/help/simulink/ug/system-design-in-simulink-using-system-objects.html
- https://www.mathworks.com/help/matlab/ref/matlab.system.isinputdirectfeedthroughimpl.html
- https://www.mathworks.com/help/simulink/ug/types-of-subsystems.html
- https://www.mathworks.com/help/simulink/slref/simulink-subsystem-semantics.html
- https://www.mathworks.com/help/simulink/ug/overview-of-model-referencing-1.html
- https://www.mathworks.com/help/simulink/ug/create-nonvirtual-buses.html
- https://www.mathworks.com/help/simulink/ug/how-the-acceleration-modes-work.html
- https://www.mathworks.com/help/simulink/ug/comparing-performance.html
- https://www.mathworks.com/help/slcompiler/ug/rapid-accelerator-dependencies.html
- https://www.mathworks.com/help/simulink/ug/fast-restart-workflow.html
- https://www.mathworks.com/help/simulink/slref/parsim.html
- https://www.mathworks.com/help/slcontrol/ug/linearizing-nonlinear-models.html
- https://www.mathworks.com/help/slcontrol/ug/findop.html
- https://www.mathworks.com/help/slcontrol/ug/frequency-response-estimation-using-simulation-based-techniques.html
- https://www.mathworks.com/help/sltest/ug/examine-model-verification-results-by-using-simulation-data-inspector.html
- https://www.mathworks.com/help/rtw/ug/s-functions-and-code-generation.html
- https://uk.mathworks.com/help/rtw/tlc/inlining-s-functions.html
- https://au.mathworks.com/help/simulink/slref/matlabfunction.html
- https://www.mathworks.com/help/simulink/block-masks.html
- https://www.mathworks.com/help/simulink/libraries.html
- https://www.mathworks.com/help/simulink/slref/add_block.html
- https://github.com/mathworks/Call-simulink-from-Python
- https://www.mathworks.com/products/simulink-coder.html
- https://www.mathworks.com/products/embedded-coder.html
- https://www.mathworks.com/products/simulink-real-time.html
- https://www.mathworks.com/products/simulink-desktop-real-time.html
- https://www.mathworks.com/help/sldrt/low-sample-rate-simulation.html
- https://www.mathworks.com/help/comm/rtlsdrradio/ug/rtlsdrreceiver.html
- https://www.mathworks.com/help/stateflow/ug/chart-during-actions.html
- https://ch.mathworks.com/help/stateflow/chart-execution-semantics.html
- https://www.mathworks.com/help/simulink/ug/compare-simulink-model-files.html
- https://ch.mathworks.com/matlabcentral/answers/353070-what-is-the-difference-between-mdl-and-slx-files-and-full-form-of-mdl-and-slx
- https://www.mathworks.com/store/products/standard/perpetual/
- https://www.mathworks.com/store/products/standard/annual/ML
- https://www.mathworks.com/pricing-licensing.html
- https://www.mathworks.com/company/technical-articles/the-growth-of-matlab-and-the-mathworks-over-two-decades.html

## PathSim

Analysed: PathSim 0.32.1, the current PyPI release (2026-11-18). The source was read from the sdist and not installed or run.

### Purpose and scope
- PathSim is a general-purpose, script-based block-diagram simulator for continuous, discrete and hybrid dynamical systems. The JOSS paper presents it as an open alternative to Simulink.
- The examples cover control (PID, DC motor, ABS), chemical reactors, mixed-signal circuits (SAR ADC, delta-sigma ADC), impact mechanics, FMCW radar and RF one-ports.
- The core holds the engine, 31 integrators and about 100 blocks. Domain blocks live in separate toolboxes: pathsim-rf, -fmi, -chem, -batt, -vehicle, -flight and -fusion.
- PathView (view.pathsim.org) is a web block-diagram editor that exports PathSim Python.

### Model of computation
- Continuous time comes first. `Simulation.time` is a float that advances by a global step `dt`, either fixed or adaptive.
- A signal is one float64 scalar per port channel at the current instant. Each block keeps `inputs` and `outputs` as a `utils.register.Register`, which wraps a flat `np.float64` array.
- Execution is sample by sample: each evaluation covers one time point. There are no frames or chunks. Time series exist only inside recorders (`Scope.recording_data`).
- Complex values cannot travel on connections because `Register` is float64. Internal states may be complex (`Spectrum`).
- One system evaluation, `Simulation._update(t)`, sweeps the algebraic graph once and then iterates any algebraic loops to a fixed point.
- Discrete time is emulated. Discrete blocks act inside `Schedule` event callbacks at `tau + k*T` and hold their outputs between ticks.

### Block interface
- A block author subclasses `blocks._block.Block` and declares ports in `input_port_labels` and `output_port_labels`, which map names to an index or a `Carrier`.
- For algebraic behaviour the author sets `op_alg = Operator(func)` (`optim.operator`) or overrides `update(t)`, which reads `inputs` and writes `outputs`.
- Defining the attribute `initial_value` marks a block as dynamic. `Block.set_solver` then gives it its own engine (`Solver.create`).
- A dynamic block implements `step(t, dt)` for an explicit stage, returning `(success, err_norm, scale)`. It also implements `solve(t, dt)` for one implicit iteration, returning the residual. Both usually go through `op_dyn = DynamicOperator(f(x, u, t), jac_x, jac_u)`.
- Discrete state is held in plain attributes that `Schedule` callbacks in `self.events` change (`blocks.discrete.FIR`, `DiscreteStateSpace`).
- `sample(t, dt)` runs after each accepted step. Recorders, noise draws and delay buffers use it.
- Feedthrough is declared per block, not per port pair. `Block.__len__` returns 1 for an algebraic path and 0 for a block that cuts the path.
- `StateSpace.__len__` tests `np.any(D)`. `DynamicalSystem.__len__` tests a numerical `jac_u` of the output function at the current state.
- The `@mutable` decorator (`utils.mutable`) re-runs `__init__` when a parameter changes and keeps the engine. It also bumps `Block._structure_version`, which makes the graph rebuild.
- Optional hooks are `to_statespace(t)`, `linearizable`, `derivative(t)`, `to_checkpoint`/`load_checkpoint` and `plot`.

### Connections and hierarchy
- `Connection(source, *targets)` links one output `PortReference` to any number of inputs. Ports are selected with `block[i]`, `block["name"]`, slices such as `block[0:3]`, or lists.
- Each input channel accepts only one driver; `Graph._validate_connections` raises an error on a conflict. `Connection.update` copies values through cached integer indices (`_transfers`).
- Vector signals are bundles of scalar channels. Port slices, `utils.carrier.Carrier` (named channels) and `Vector` (unnamed channels) group them. `Connection._validate_dimensions` checks widths and channel names.
- `Subsystem` is a `Block` with its own blocks, connections, `Graph` and loop solver. `Interface` is its boundary block.
- Subsystems are not flattened. The parent sees a single block whose `__len__` is 1 if any path from interface to interface is algebraic.
- Blocks, connections and events can be added or removed during a run (`Simulation.add_block`, `remove_connection`). The graph is rebuilt lazily.

### Compilation phase
- There is no separate compile step. `Simulation.__init__` calls `set_solver` on every block and `Connection.resolve_ports` on every connection, then builds a `utils.graph.Graph`.
- `Graph._compute_depths_iterative` (an iterative depth-first search) gives each algebraic block its algebraic depth. Blocks with `len == 0` sit at depth 0 and cut paths.
- Blocks on a cycle get no depth. `_process_loops` finds their strongly connected components with an iterative Tarjan search and orders each component by breadth-first search. Back edges become loop-closing connections.
- Each loop-closing connection is wrapped in an `optim.booster.ConnectionBooster`, which owns an `Anderson` accelerator (history depth 4).
- Nothing propagates rates, types, units or dimensions. Port widths grow lazily (`Register.resize`), and sample times are constructor arguments of each block.

### Execution
- `Simulation._timestep` buffers engines and events, then evaluates the system and calls `step` on the dynamic blocks for each `engine.stages(t, dt)`. It then evaluates at `t + dt`, detects and resolves events, and calls `sample` on all blocks.
- `_update(t)` runs `_dag(t)` and then `_loops(t)`. `_dag(t)` makes one pass by depth, updating blocks and then connections. `_loops(t)` sweeps the loop levels and steps the boosters until the largest residual is at most `tolerance_fpi` (1e-10). It raises `RuntimeError` after 200 iterations.
- Every block owns a solver instance. A dummy `Simulation.engine` is their `parent` and keeps the stage counter. One solver class applies to all blocks.
- Fixed-step explicit solvers are EUF, SSPRK22 (the default), SSPRK33, SSPRK34 and RK4. Adaptive embedded explicit solvers (`ExplicitRungeKutta`) are RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78 and RKDP87.
- Fixed-step implicit solvers are EUB, DIRK2, DIRK3, ESDIRK4 and BDF2 to BDF6. Adaptive implicit solvers are ESDIRK32/43/54/85 and the variable-step BDF methods GEAR21/32/43/54/52A. BDF starts up with DIRK3 and GEAR with ESDIRK32.
- In an implicit stage, `Simulation._solve` alternates a full `_update` with `block.solve`. Each engine takes a `NewtonAnderson` step on its own state using only its block-local Jacobian. There is no global Newton iteration and no global Jacobian.
- Adaptive control uses a scaled max-norm error per block (`tolerance_lte_abs` 1e-8, `tolerance_lte_rel` 1e-4). `StepTracker` keeps the worst error and the smallest step scale. A rejected step calls `_revert` and retries.
- The `events` package provides `ZeroCrossing` (also `Up` and `Down` variants, located by secant), `Condition` (located by bisection) and `Schedule`/`ScheduleList` (time-based).
- In adaptive runs a step that crosses an event is rejected and rescaled until it is close (|g| ≤ 1e-4). `Event.estimate` also shortens `dt` to the next scheduled tick.
- Fixed-step runs resolve an event at the interpolated time but use end-of-step values, and handle at most one tick per event per step.
- Multirate works only by giving each discrete block its own `Schedule`. There is no base rate, no rate-transition check and no up- or downsampling block.
- `run`, `run_streaming` and `run_realtime` share `_run_loop`. `run_streaming` is a generator that calls a callback at a wall-clock tick rate. `run_realtime` paces simulated time to the wall clock with `time.sleep(1e-3)`. `stop()` or raising `StopSimulation` ends a run. Everything runs in one thread.

### Performance techniques
- The package is pure Python and NumPy, with no numba, Cython, JIT or code generation.
- The optimisations are all at the interpreter level: cached port indices (`Connection._transfers`), `__slots__`, iterative graph algorithms, a cached LU in `NewtonAnderson`, and reuse of the end-of-step evaluation for the next first stage (`Simulation._evaluated`).
- An s-stage explicit step costs about s full sweeps. Each sweep makes one Python `update` call per block and per connection, plus `inputs.to_array()` copies, followed by event checks and `sample` on every block.
- Discrete signal processing runs in Python once per tick. `FIR` sums `c*b` over a deque inside its `Schedule` callback, and each sampled block adds an event that is checked every step.
- The `Subsystem` docstring itself advises a monolithic `ODE` block for speed.
- The docs list FastSim, a separate Rust reimplementation with JIT and C code generation. It is not part of this package and was not examined.

### Analysis features
- `Simulation.linearize()` replaces each block's `Operator`/`DynamicOperator` with a first-order Taylor surrogate; `delinearize()` restores the originals.
- `Simulation.to_statespace(inputs, outputs)` assembles global A, B, C, D matrices from the block models and returns them as a `StateSpace` block (`optim.linearization.assemble_from_ports`). The docstring shows handing the result to python-control.
- `Subsystem.to_statespace` composes models hierarchically.
- Jacobians come from user callables or from central differences (`optim.numerical.num_jac`, relative step 1e-3).
- Version 0.32.1 has no automatic differentiation. The forward-mode `Value` class described in the JOSS paper ships in wheels up to 0.11.0 and is missing from 0.12.0 on. Docstrings in `operator.py` still describe it, and `test_table.py` skips its test.
- `steadystate()` swaps in the `SteadyState` solver, which sets dx/dt = 0. `pss(period)` finds a periodic steady state by Anderson-accelerated shooting.
- `estimate_initial_step` implements the Hairer–Nørsett–Wanner rule for the initial step size.
- There is no Bode plot or frequency sweep. `Spectrum` integrates a running Fourier transform (with optional exponential forgetting) as ODE states during the run.

### Extensibility
- Generic blocks are `Function` (algebraic), `DynamicalFunction`, `ODE` and `DynamicalSystem` (dynamic), `Wrapper` (calls arbitrary Python every `T`) and `LUT`/`LUT1D` (lookup tables).
- DAE and implicit blocks are `SemiExplicitDAE`, `MassMatrixDAE`, `FullyImplicitDAE`, `SparseLinearDAE`, `AlgebraicConstraint` and `BVP1D` (built on `scipy.integrate.solve_bvp`).
- A new solver subclasses `ExplicitRungeKutta` or `DiagonallyImplicitRungeKutta` and supplies a Butcher table.
- A new event type subclasses `Event` and implements `detect`, `resolve` and `estimate`.
- `Block.info()` and `Block.icon()` expose parameters, ports and SVG icons to GUIs. Toolboxes ship their own icons.

### Hardware and real-time I/O
- Neither the core nor the listed toolboxes include device drivers or audio/SDR I/O.
- `run_realtime` keeps pace with a sleep loop. It has no deadline handling and does not report overruns.
- Hardware would be coupled through `Wrapper` or a custom block.
- FMU support moved to pathsim-fmi (FMPy, FMI 2.0 and 3.0). The deprecated `blocks.fmu` and `utils.fmuwrapper.FMUWrapper` remain in the core.
- `RFNetwork`, which fits a state-space model to Touchstone data with scikit-rf, moved to pathsim-rf.

### Visualisation and tooling
- Plotting uses matplotlib: `Scope.plot`, `plot2D`, `plot3D`, `Spectrum.plot` and `Simulation.plot`. `Scope.save` and `Spectrum.save` write CSV.
- `utils.realtimeplotter.RealtimePlotter` updates a plot live.
- Logging and diagnostics come from `utils.logger.LoggerManager` (a singleton, stdout or file), `ProgressTracker`, per-step `Diagnostics` snapshots (full history with `diagnostics="history"`) and `utils.analysis.Timer`/`Profiler`.
- `save_checkpoint` and `load_checkpoint` store states as JSON plus NPZ, keyed by block type and insertion order. The model structure itself is not serialised.
- The core cannot render diagrams (no text or DOT output); PathView is the graphical front end.

### Licence, dependencies, maturity
- Licence is MIT. Runtime dependencies are numpy ≥ 1.15, scipy ≥ 1.2 and matplotlib ≥ 3.1, on Python ≥ 3.8. The `rf` and `fmi` extras are deprecated.
- The source has about 27,300 lines, much of it docstrings. Tests: 119 modules with about 1,350 test functions and 29,000 lines. CI is a single Ubuntu job with Codecov.
- PyPI lists 99 releases, from 0.1.0 (2024-08-12) to 0.32.1 (2026-11-18), several per week in November 2026. PathSim is also on conda-forge.
- It is still at 0.x. Many APIs are deprecated for removal in 1.0.0, among them `Simulation.step`, `collect`, `RealtimeScope` and the FMU and RF blocks.
- There is one principal author. The JOSS paper was published 2025-05-13.

### Strengths
- The engine is small and readable, with minimal dependencies. Its behaviour can be traced directly in `Simulation._timestep`.
- The solver suite is broad and tested, including stiff ESDIRK and variable-step BDF methods. Solvers can be swapped mid-run (`_set_solver`).
- Hybrid simulation is first-class. Locating events is part of adaptive step control, and scheduled ticks shape `dt`.
- Algebraic loops are solved automatically with Anderson-accelerated fixed-point iteration instead of being rejected. DAE and constraint blocks build on this.
- Models can change live: blocks and connections can be added or removed and parameters are mutable. Streaming and real-time generators are built in.
- Linearisation to a global state-space model, steady state and periodic steady state are included.
- Checkpoints, diagnostics, a GUI editor and domain toolboxes round it out.

### Weaknesses
- Python overhead is paid per sample. Each stage sweeps every block and connection, so a DSP chain at 1 MS/s needs at least 10^6 global steps per simulated second.
- There is no sample-rate bookkeeping. Each discrete block has its own `T`, nothing checks rates between connected blocks, and there are no decimate, interpolate or resample blocks.
- Signals are real float64 scalars only. There are no complex baseband, integer or bit types; `ADC` emits bits as floats on separate ports.
- There is no vectorisation over time, no JIT and no code generation in this package.
- Feedthrough is declared per block, not per port. A subsystem with one feedthrough path counts as fully algebraic, so a false loop costs fixed-point iterations instead of a static schedule.
- `DynamicalSystem` infers feedthrough from a numerical Jacobian at the current state, so the answer can depend on the operating point.
- Implicit solves use only block-local Jacobians inside a global fixed-point loop. Stiff dynamics spread across many blocks may converge slowly.
- In fixed-step runs, events use end-of-step values, and ticks fall behind when `T < dt`.
- The advertised automatic differentiation is gone, but the paper, the docstrings and the `pyproject` description ("differentiable") still claim it.
- Releases are frequent and deprecations common. Checkpoints match blocks only by type and insertion order.

### Sources
- PathSim 0.32.1 sdist (PyPI). Main files read: `simulation.py`, `subsystem.py`, `connection.py`, `blocks/_block.py`, `blocks/discrete.py`, `blocks/spectrum.py`, `blocks/dynsys.py`, `blocks/delay.py`, `blocks/noise.py`, `blocks/rf.py`, `blocks/fmu.py`, `events/*.py`, `solvers/_solver.py`, `solvers/_rungekutta.py`, `solvers/bdf.py`, `solvers/gear.py`, `optim/anderson.py`, `optim/booster.py`, `optim/operator.py`, `optim/numerical.py`, `optim/linearization.py`, `utils/graph.py`, `utils/register.py`, `utils/carrier.py`, `pyproject.toml` and `tests/`.
- Wheels 0.8.0 to 0.24.0 from PyPI. Only the file lists were checked, for the presence of `optim/value.py`.
- PyPI release history: https://pypi.org/pypi/pathsim/json
- GitHub release notes: https://github.com/pathsim/pathsim/releases
- M. Rother, "PathSim – A System Simulation Framework", JOSS 10(109), 8158 (2025): https://doi.org/10.21105/joss.08158
- Documentation and toolbox list: https://docs.pathsim.org and https://docs.pathsim.org/fastsim

## python-control

Analysed version: `control` 0.10.2 (released 2025-07-05, still the latest release on 2026-11-25). Read from the PyPI wheel source; every claim names the module, class or function behind it.

### Purpose and scope

- Library for control-system analysis and design, modelled on the MATLAB Control System Toolbox; `control.matlab` adds MATLAB-named wrappers (`step`, `lsim`, `bode`, `rlocus`).
- Its core objects are system models: `StateSpace`, `TransferFunction` and `FrequencyResponseData` (all `LTI`), plus `NonlinearIOSystem` for general ODEs and difference equations.
- Simulation is one consumer of these models among many. Most of the ~35k library lines cover linear analysis, synthesis, optimisation and plotting.
- It is not a block-diagram simulator: no editor, no scheduler, no sample-stream engine, no signal-processing block library.

### Model of computation

- A system is a state-space map: `updfcn(t, x, u, params)` returns dx/dt or x[k+1], and `outfcn(t, x, u, params)` returns y (`NonlinearIOSystem`, `nlsys.py`).
- Every `InputOutputSystem` has a timebase `dt`: 0 is continuous, a positive number is discrete with that period, True is discrete with an unspecified period, None is unspecified. The default comes from `config.defaults['control.default_dt']` and is 0.
- Static LTI models (`StateSpace` and `TransferFunction` without dynamics) default to `dt=None` (`_process_dt_keyword(static=True)`), so gains combine with any timebase. A static `nlsys` still defaults to 0.
- `common_timebase(dt1, dt2)`: None takes the other value, True takes a positive dt, equal values pass (`np.isclose`), and anything else raises `ValueError("Systems have incompatible timebases")`.
- Signals exist only in results: 2-D arrays indexed (signal, time), or (signal, trace, time) for multi-trace responses. Time is the last axis (`TimeResponseData`). `NamedSignal`, an ndarray subclass, allows `resp.outputs['y']`.
- A simulation is a single call over a whole time vector: `input_output_response(sys, timepts, inputs, initial_state, params)`, or `forced_response`, `step_response`, `impulse_response` and `initial_response` for LTI models.
- Continuous-time inputs are linearly interpolated between grid points (`ufun` in `input_output_response`; a first-order-hold matrix exponential in `forced_response`).
- An interconnection is itself a system, so `linearize`, `poles` and the margin functions apply to the whole diagram. The algebraic model gives you analysis of the whole loop. It costs you a run-time engine.

### Block interface

- A model is either two plain functions passed to `nlsys(updfcn, outfcn, inputs=, outputs=, states=, params=, dt=)`, or a subclass that overrides `_rhs(t, x, u)` and `_out(t, x, u)`.
- State is a flat float vector that the caller or solver passes in. The system object only describes the model and keeps no signal state between calls.
- Parameters are a dict. Defaults live in `sys.params` and can be overridden per call (`params=`). `InterconnectedSystem._update_params` merges them down the hierarchy and stores them on each object as `_current_params`.
- Direct feedthrough is never declared. Any `outfcn` may read `u`, and the framework only learns this at run time.
- If `outfcn` is missing, y = x. If `updfcn` is missing, the system is a static map that can be called as `sys(u)` (`NonlinearIOSystem.__call__`).
- `t` is passed through, so time-varying models work. There are no hooks for events, resets or discrete updates.
- LTI classes use the same interface: `StateSpace.__init__` installs lambdas for `A x + B u` and `C x + D u`. A `TransferFunction` is converted to `StateSpace` before interconnection (`_convert_to_statespace`); for MIMO this needs slycot `td04ad`.

### Connections and hierarchy

- There are two ways to connect systems. Algebra returns new model objects (operators and `bdalg`). Name-based connection uses `interconnect()`, which returns an `InterconnectedSystem` or a `LinearICSystem`.
- The operators are `*` for series (`series(a, b)` is `b * a`, in signal-flow order), `+` for parallel and `-` for negation. The functions are `feedback(sys1, sys2, sign=-1)`, `append` (block diagonal, LTI only) and `connect` (index-based, deprecated with a `FutureWarning`).
- `StateSpace` algebra builds block matrices (`StateSpace.__mul__`, `.feedback`).
- `TransferFunction` algebra multiplies polynomials and never cancels common factors: `feedback` returns num1·den2 / (den1·den2 − sign·num1·num2). MIMO `TransferFunction.feedback` raises `ControlMIMONotImplemented`.
- `FrequencyResponseData` algebra works point by point on one frequency grid. Mismatched grids raise `NotImplementedError` (`_convert_to_frd`).
- With mixed LTI operands, the left operand's class wins (`SS * TF` gives SS, `TF * SS` gives TF). Any nonlinear operand makes the result an `InterconnectedSystem` (`NonlinearIOSystem.__mul__`, `__add__`, `feedback`).
- Signals are named scalar channels. Vectors are families of names like `u[0]`, `u[1]`, matched by base name or slice (`'P.y[0:2]'`, `_find_signals`).
- `interconnect(syslist, connections=None, ...)` with no connections connects each subsystem input to every subsystem output of the same name. Several outputs with one name are summed.
- Explicit connections are lists `[input_spec, output_spec, ...]` with optional gains (`('sys', 'sig', k)`, `'-sys.sig'`). `summing_junction(inputs=['r', '-y'], output='e')` builds a `StateSpace` with only a D matrix.
- `check_unused_signals` warns about unconnected signals, `add_unused=True` makes them external inputs and outputs, and `connection_table()` prints the routing.
- Nesting works because `InterconnectedSystem` is a `NonlinearIOSystem`. Inner signals are hidden, `sys.sig` names reach one level down, and states are flattened to labels like `P_x[0]`.
- Duplicate subsystem objects are deep-copied, renamed with a `$copy` suffix and reported with a warning (`InterconnectedSystem.__init__`).

### Compilation phase

- `InterconnectedSystem.__init__` only builds three dense matrices and offset lists into the stacked x, u and y vectors:
  - `connect_map`: subsystem inputs × subsystem outputs
  - `input_map`: subsystem inputs × external inputs
  - `output_map`: external outputs × subsystem outputs and inputs
- There is no dependency graph, topological sort, execution order, feedthrough analysis or rate propagation.
- The only semantic check is the timebase: `common_timebase` is folded over all subsystems, and mixing continuous and discrete raises an error.
- Algebraic loops in nonlinear interconnections are not checked at build time. They show up at run time (section 6).
- When every subsystem is LTI, `interconnect` returns a `LinearICSystem` and gets A, B, C and D from `NonlinearIOSystem.linearize(0, 0)`. That is forward differencing (eps = 1e-6) through the same fixed-point evaluator, so feedthrough loops fail there too.
- A `LinearICSystem` keeps `syslist` and the maps, so it is both a `StateSpace` and an `InterconnectedSystem`.
- There is no scheduling in the Simulink sense. Evaluation order comes from repeated sweeps.

### Execution

- `InterconnectedSystem._compute_static_io(t, x, u)` calls every subsystem's `_out` with the current input guesses and sets the internal inputs to `connect_map @ y + input_map @ u`. It repeats until the inputs stop changing exactly, for at most n+1 sweeps over n subsystems.
- If that limit is reached it raises `RuntimeError("algebraic loop detected")` (checked by `iosys_test.py::test_algebraic_loop`).
- There is no algebraic-loop solver, and detection depends on the values: an all-zero point passes.
- LTI loops through D terms can instead be closed algebraically: `StateSpace.feedback` solves with F = I − sign·D2·D1 and fails only if F is singular.
- `_rhs` and `_out` each repeat the full sweep.
- For continuous time, `input_output_response` calls `scipy.integrate.solve_ivp(ivp_rhs, (T0, Tf), X0, t_eval=T, method='RK45', vectorized=False)` and then evaluates `_out` again at each output time.
- Solver tolerances are SciPy's defaults unless given in `solve_ivp_kwargs`. The docstring warns about discontinuous inputs and suggests `max_step`, `rtol` or LSODA.
- For discrete time, a Python loop over `t_eval` calls `_out` and then `_rhs` once per sample. The step must equal `sys.dt`, otherwise `ValueError`.
- LTI `forced_response` handles continuous systems by exact discretisation with a first-order hold: one `scipy.linalg.expm` of an augmented matrix, then a Python loop per step. Discrete systems go to `scipy.signal.dlsim`.
- There is no mixed continuous/discrete or multirate simulation, since an interconnection has one timebase.
- `sample_system` and `StateSpace.sample` (ZOH, bilinear, Euler, matched; via `cont2discrete`) discretise LTI models only. `NonlinearIOSystem` has no `sample`.
- There is no event or zero-crossing API. `events` can be passed through `solve_ivp_kwargs`, but `TimeResponseData` drops `t_events`.
- Nothing streams: each call starts from `initial_state` and returns full arrays. To continue a run, pass the last state as the new X0.
- Pure delays exist only as Padé approximations (`pade`) or as explicit discrete states.

### Performance techniques

- Linear responses avoid ODE solvers:
  - matrix exponentials in `forced_response`
  - `dlsim` for discrete systems
  - frequency evaluation by `solve(sI − A, B)` per point, or slycot `tb05ad` (`StateSpace.horner`, `slycot_laub`)
- Slycot (SLICOT Fortran) provides Riccati and Lyapunov solvers (`sb02md`, `sb03md`, `sg02ad`), H2/H∞ synthesis (`sb10hd`, `sb10ad`), balanced reduction (`ab09*`), MIMO TF↔SS conversion (`td04ad`, `tb04ad`) and norms (`ab13bd`).
- `care`, `dare`, `lyap` and `system_norm` fall back to SciPy. H2/H∞ synthesis, balanced reduction and MIMO TF↔SS conversion fail without slycot (`ControlSlycot`, `ControlMIMONotImplemented`).
- The nonlinear path has no JIT, no compilation and no vectorisation over time.
- Each right-hand-side call runs 2 to n+1 Python sweeps over all subsystems, and RK45 makes about six such calls per step.
- Cost grows with subsystem count × sweeps × solver steps. That suits a handful of blocks with smooth dynamics, not long sample streams.
- `create_mpc_iosystem` runs a full `scipy.optimize.minimize` inside both `_update` and `_output`, so every extra sweep repeats an optimisation.

### Analysis features

- `find_operating_point` (alias `find_eqpt`) uses `scipy.optimize.root` with fixed or free subsets of states, inputs, outputs and derivatives, and returns an `OperatingPoint`.
- `linearize(sys, xeq, ueq)` uses forward differences on `_rhs`/`_out` and returns a `StateSpace` with signal names.
- `frequency_response` returns `FrequencyResponseData`, which also serves as a data model. Related tools: `bode_plot`, `nyquist_response` (counts encirclements, indents around imaginary-axis poles), `nichols_plot`, `singular_values_plot`, `gangof4_plot`.
- Margins: `stability_margins` (polynomial or FRD method), `margin`, `phase_crossover_frequencies` and `disk_margins` (MIMO needs slycot `ab13md`).
- Roots and stability: `poles`, `zeros`, `damp`, `pole_zero_map`, `root_locus_map`, `ispassive` (cvxopt LMI) and `system_norm`.
- Synthesis:
  - `place` (`scipy.signal.place_poles`), `place_acker`
  - `lqr`/`dlqr`, `lqe`/`dlqe`
  - `h2syn`, `hinfsyn`, `mixsyn` (slycot)
  - `create_statefbk_iosystem` (with gain scheduling), `create_estimator_iosystem`
- Model reduction and realisation: `minimal_realization`, `balanced_reduction`, `eigensys_realization`, `markov`, `canonical_form`.
- `control.optimal`: `OptimalControlProblem` uses shooting or collocation and passes the result to `scipy.optimize.minimize`. Also `solve_optimal_trajectory`, `create_mpc_iosystem` and moving-horizon estimation (`OptimalEstimationProblem`).
- `control.flatsys`: `FlatSystem` takes user `forward`/`reverse` maps and uses `PolyFamily`, `BezierFamily` or `BSplineFamily` bases, with `point_to_point` and `solve_flat_optimal`.
- Describing functions: `describing_function` (numerical), built-in saturation, relay-hysteresis and backlash nonlinearities, and `describing_function_plot` for finding limit cycles.
- `phase_plane_plot` combines `vectorfield`, `streamlines`, `equilpoints` and `separatrices` for 2-state systems.
- `white_noise(T, Q, dt)` uses the global `np.random`; there is no seed or generator argument.

### Extensibility

- New dynamics: pass functions to `nlsys`, or subclass `NonlinearIOSystem` and override `_rhs`, `_out` and optionally `linearize`.
- New nonlinearities for harmonic balance: subclass `DescribingFunctionNonlinearity` and implement `describing_function(A)`.
- New trajectory bases: subclass `flatsys.BasisFamily`.
- Global behaviour is set through `config.defaults` (timebase, name delimiters, plot settings) and `use_legacy_defaults`.
- There is no block registry, no port typing, and no solver hook beyond `solve_ivp` method names and keyword arguments.

### Hardware and real-time I/O

- None. The source has no device drivers, threads, clocked loops or streaming API.
- For real-time use you would extract a discrete model (`sample_system`, `ssdata`) or call `sys.dynamics()` and `sys.output()` per sample from your own loop, with full Python overhead per call.

### Visualisation and tooling

- Matplotlib plotting is split into `*_response` (returns a data object) and `*_plot` (draws it). Plots return a `ControlPlot` with lines, axes, figure and legend.
- `TimeResponseData.plot` calls `time_response_plot`. `TimeResponseData` and `FrequencyResponseData` both have `to_pandas()`.
- Interactive tools: `sisotool`, `rootlocus_pid_designer` and click-to-read gains on root-locus plots.
- In Jupyter, `StateSpace` and `TransferFunction` render as LaTeX. `InterconnectedSystem.__str__` lists each subsystem input as a sum of sources.
- There is no diagram rendering (no DOT or graph export) and no GUI editor.

### Licence, dependencies, maturity

- BSD-3-Clause. Copyright: Caltech 2009–2016, TU Delft 2012, python-control developers 2016–2024 (`LICENSE`).
- Requires Python ≥ 3.10, numpy ≥ 1.23, scipy ≥ 1.8 and matplotlib ≥ 3.6. Optional extras are `slycot` ≥ 0.4 and `cvxopt` ≥ 1.2; pandas is used when installed.
- Classified "Development Status :: 4 - Beta" and still 0.x. Recent releases: 0.10.0 (2024-03), 0.10.1 (2024-08), 0.10.2 (2025-07). About 2.1k GitHub stars.
- The wheel holds ~35k library lines and ~27k test lines (658 test functions, many parametrised). `docstrings_test.py` checks the numpydoc docstrings.

### Strengths

- Systems are values. Algebra (`*`, `+`, `feedback`) returns new models that every analysis function accepts.
- One interface covers LTI and nonlinear, continuous and discrete systems. A `LinearICSystem` keeps its signal names and is a `StateSpace` at the same time.
- `interconnect` connects by name and supports summing junctions, which keeps control-loop setup short.
- LTI time responses are exact, and the linear toolset is broad and well tested (margins, Nyquist, LQR/LQE, H2/H∞, reduction), backed by SLICOT.
- The nonlinear workflow is complete: `find_operating_point`, then `linearize`, then design, then `input_output_response` on the nonlinear plant.
- Optimal control, MPC, flatness and describing functions use the same object model.
- Models are plain functions with dict parameters, so scripts stay flat and sweeps just pass `params=` without rebuilding.

### Weaknesses

- There is no scheduler. Every evaluation repeats up to n+1 sweeps over all subsystems in pure Python, and the cost grows with diagram size.
- Algebraic loops through nonlinear feedthrough raise `RuntimeError` mid-simulation. There is no loop solver, no build-time diagnosis, and detection depends on the signal values.
- Feedthrough is undeclared, so the library cannot order blocks or warn early.
- Each interconnection has one timebase. There are no hybrid continuous/discrete models, no multirate, no rate transitions and no sample-and-hold blocks, and only LTI models can be discretised.
- Simulation is batch only: no persistent block state between calls, no chunked or streaming runs, no real-time I/O.
- There are no events, zero-crossings, resets or mode switches. The default solver is RK45 with SciPy tolerances, and inputs are always linearly interpolated.
- Transport delay is only available as a Padé approximation. Noise has no managed seeds.
- It is not a DSP or communications tool: no FIR/IIR or resampling blocks, no bit or frame handling.
- Internal buffers are real arrays (`np.zeros` in `_rhs` and `_compute_static_io`), so complex baseband signals cannot pass through an interconnection.
- Transfer-function algebra raises the order without cancelling factors (you need `minreal`). Several MIMO paths need slycot or are not implemented.
- `set_connect_map`, `set_input_map` and `set_output_map` create their shape-check `ValueError` but never raise it.

### Sources

1. PyPI wheel `control-0.10.2-py3-none-any.whl` (downloaded with `pip download control --no-deps`, read without importing). Files read: `iosys.py`, `nlsys.py`, `statesp.py`, `xferfcn.py`, `frdata.py`, `lti.py`, `bdalg.py`, `timeresp.py`, `dtime.py`, `optimal.py`, `flatsys/`, `descfcn.py`, `margins.py`, `statefbk.py`, `mateqn.py`, `robust.py`, `stochsys.py`, `ctrlplot.py`, `tests/iosys_test.py`, `METADATA`, `LICENSE`.
2. https://github.com/python-control/python-control (release list and repository metadata via the public GitHub REST API).
3. https://python-control.readthedocs.io/en/0.10.2/ (pages `intro`, `iosys`, `nlsys`, `linear`, `response`).
4. https://pypi.org/project/control/
