import cmath
import math
import warnings

import numpy as np
import pytest

import sigline as sl
from sigline import Diagram, FromArray, Nonlinear, Scope, state
from sigline._numba import HAVE_NUMBA
from sigline.testing import assert_chunk_invariant

needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba")
X = np.sin(np.linspace(0, 20, 5000)) + 0.3 * np.random.default_rng(0).standard_normal(
    5000
)


def halfwave(x, knee=0.2):
    if x > knee:
        return x - knee
    return 0.0


def schmitt(x, s, hi=0.5, lo=-0.5):
    if x > hi:
        s = 1.0
    elif x < lo:
        s = -1.0
    return s, s


def polar(i, q):
    return math.hypot(i, q), math.atan2(q, i)


def pll(x, s, alpha=0.05, beta=0.002):
    # first-order phase detector, second-order loop; state (phase, freq)
    y = x * cmath.exp(-1j * s.phase)
    e = math.atan2(y.imag, y.real)
    freq = s.freq + beta * e
    return (y, freq), type(s)((s.phase + freq + alpha * e) % (2 * math.pi), freq)


def run(block, x, chunk):
    src, sc = FromArray(x), Scope()
    src >> block >> sc
    Diagram(sc, fs=1.0).run(n=len(x), chunk=chunk)
    return sc.data


# unit


def test_ports_params_and_info():
    nl = Nonlinear(halfwave, params={"knee": 0.1})
    assert nl.inputs == ("x",) and nl.outputs == ("y",)
    assert dict(nl.params) == {"knee": 0.1} and nl.name.startswith("halfwave")
    p = Nonlinear(polar, outputs=("mag", "ph"))
    assert p.inputs == ("i", "q") and p.outputs == ("mag", "ph")
    t = Nonlinear(schmitt, state=-1.0)
    assert t.inputs == ("x",) and t.info() == "(stateful)"
    with pytest.raises(TypeError, match="no parameter"):
        Nonlinear(halfwave, params={"kne": 1})
    with pytest.raises(TypeError, match=r"\(x, state\)"):
        Nonlinear(lambda x: x, state=0.0)
    with pytest.raises(ValueError, match="one entry per output"):
        Nonlinear(polar, outputs=("mag", "ph"), dtype=(float,))
    with pytest.raises(ValueError, match="jit"):
        Nonlinear(halfwave, jit="yes")


def test_array_state_inside_tuple_untouched_by_probe_and_reset():
    # regression: the dtype probe ran f on the live state; reset() shared arrays
    def counter(x, s):
        buf, n = s
        buf[0] += 1.0
        return buf[0], (buf, n + 1.0)

    init = (np.zeros(1), 0.0)
    nl = Nonlinear(counter, state=init)
    np.testing.assert_array_equal(run(nl, np.ones(100), 10), np.arange(1, 101))
    assert init[0][0] == 0.0
    nl.reset()
    assert nl.state[0][0] == 0.0


def test_same_name_input_and_output_needs_explicit_accessor():
    def gate(x, strobe):
        return x * strobe, strobe

    g = Nonlinear(gate, outputs=("y", "strobe"))
    with pytest.raises(KeyError, match="both an input and an output"):
        g["strobe"]
    assert g.inp("strobe").kind == "in" and g.out("strobe").kind == "out"


def test_runtime_parameter_change():
    nl, sc = Nonlinear(halfwave, params={"knee": 0.0}), Scope()
    FromArray(np.ones(20)) >> nl >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=10)
    nl.params["knee"] = 0.5
    d.run(n=10)
    np.testing.assert_allclose(sc.data, [1.0] * 10 + [0.5] * 10)


def test_int_results_become_float():
    nl = Nonlinear(lambda x: 1 if x > 0 else 0)
    y = nl(np.array([-1.0, 2.0]))
    assert y.dtype == float and list(y) == [0.0, 1.0]


@needs_numba
def test_uncompilable_f_falls_back_with_a_warning():
    class Table:
        k = 1.0

    t = Table()

    def weird(x):
        return t.k * x  # numba cannot type a plain python object

    nl = Nonlinear(weird)
    with pytest.warns(RuntimeWarning, match="runs in Python"):
        y = run(nl, X[:100], 50)
    np.testing.assert_allclose(y, X[:100])
    assert nl.mode == "python"
    import numba

    with pytest.raises(numba.core.errors.TypingError):
        run(Nonlinear(weird, jit=True), X[:100], 50)


# integration


def test_multi_in_multi_out():
    n = np.arange(100) * 0.1
    p, m, ph = Nonlinear(polar, outputs=("mag", "ph")), Scope(), Scope()
    FromArray(np.cos(n)) >> p["i"]
    FromArray(np.sin(n)) >> p["q"]
    p["mag"] >> m
    p["ph"] >> ph
    Diagram(m, fs=1.0).run(n=100, chunk=32)
    np.testing.assert_allclose(m.data, 1.0)
    np.testing.assert_allclose(np.unwrap(ph.data), n, atol=1e-12)


def test_in_a_compiled_loop_with_named_state():
    def integ(x, s, k=0.1):
        acc = s.acc + k * x
        return acc, type(s)(acc)

    e, nl, sc = sl.Sum("+-"), Nonlinear(integ, state=state(acc=0.0)), Scope()
    sl.Const(1.0) >> e["x0"]
    e >> nl >> sc
    nl >> sl.Delay(1) >> e["x1"]
    d = Diagram(sc, fs=1.0)
    d.run(n=400, chunk=100)
    assert d.loops(100)[0].mode == ("numba" if HAVE_NUMBA else "python")
    # theory: a[n] = 0.9 a[n-1] + 0.1, a[0] = 0.1 -> 1 - 0.9^(n+1)
    np.testing.assert_allclose(sc.data, 1 - 0.9 ** np.arange(1, 401), atol=1e-12)


# comparison


@needs_numba
@pytest.mark.parametrize("chunk", [1, 17, 5000])
def test_numba_equals_python_any_chunk(chunk):
    y = run(Nonlinear(schmitt, state=-1.0), X, chunk)
    ref = run(Nonlinear(schmitt, state=-1.0, jit=False), X, 5000)
    np.testing.assert_array_equal(y, ref)


def test_equals_func_on_vectorisable_function():
    y = run(Nonlinear(math.tanh), X, 333)
    np.testing.assert_allclose(y, run(sl.Func(np.tanh), X, 333), rtol=1e-15)


@needs_numba
def test_kernel_in_a_loop_equals_python():
    def make(mode):
        add, nl, sc = sl.Sum("+-"), Nonlinear(schmitt, state=-1.0), Scope()
        FromArray(X) >> add["x0"]
        add >> nl >> sl.Delay(2) >> sl.Gain(0.3) >> add["x1"]
        add >> sc
        Diagram(sc, fs=1.0, compile_loops=mode).run(n=len(X), chunk=500)
        return sc.data

    np.testing.assert_array_equal(make(True), make("python"))


def test_chunk_invariant():
    assert_chunk_invariant(lambda: Nonlinear(schmitt, state=-1.0), X[:500])
    assert_chunk_invariant(lambda: Nonlinear(polar, outputs=("mag", "ph")),
                           {"i": X[:300], "q": X[::-1][:300]})  # fmt: skip


# theory


def test_halfwave_equals_closed_form():
    np.testing.assert_allclose(
        run(Nonlinear(halfwave), X, 333), np.where(X > 0.2, X - 0.2, 0)
    )


def test_schmitt_switches_only_outside_the_band():
    y = run(Nonlinear(schmitt, state=-1.0), X, 333)
    assert set(np.unique(y)) == {-1.0, 1.0}
    sw = np.flatnonzero(np.diff(y))
    assert np.all(np.abs(X[sw + 1]) > 0.5)


def test_moving_max_with_array_state():
    def moving_max(x, buf):
        buf[1:] = buf[:-1]
        buf[0] = x
        return buf.max(), buf

    nl = Nonlinear(moving_max, state=np.full(8, -np.inf))
    ref = np.array([X[max(0, k - 7) : k + 1].max() for k in range(len(X))])
    np.testing.assert_allclose(run(nl, X, 100), ref)
    nl.reset()
    np.testing.assert_array_equal(nl.state, np.full(8, -np.inf))


def test_pll_locks_to_the_input_frequency():
    blk = Nonlinear(pll, state=state(phase=0, freq=0), outputs=("y", "w"),
                    dtype=(complex, float))  # fmt: skip
    src, sc = sl.Sine(500.0, complex=True, phase=1.0), Scope()
    src >> blk
    blk["w"] >> sc
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Diagram(sc, fs=1e5).run(n=20_000, chunk=4096)
    assert abs(sc.data[-1000:].mean() * 1e5 / (2 * np.pi) - 500.0) < 1.0
    assert blk.mode == ("numba" if HAVE_NUMBA else "python")
    assert blk.state.freq > 0 and isinstance(blk.state.phase, float)
    blk.reset()
    assert blk.state.phase == 0.0 and type(blk.state).__name__ == "State"


def test_inputs_can_be_renamed_like_func():
    nl = Nonlinear(lambda a, b: a - b, inputs=("plus", "minus"))
    f = sl.Func(lambda a, b: a - b, inputs=("plus", "minus"))
    assert nl.inputs == f.inputs == ("plus", "minus")
    np.testing.assert_array_equal(nl(plus=X[:5], minus=X[:5]), np.zeros(5))
