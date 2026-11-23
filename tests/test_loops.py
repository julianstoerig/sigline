import numpy as np
import pytest
import scipy.signal as sig

import sigline as sl
from _blocks import Add, Record, Scale
from sigline import AlgebraicLoopError, Diagram
from sigline._numba import HAVE_NUMBA
from sigline.loops import LoopCluster, cycles
from sigline.testing import assert_chunk_invariant

X = np.random.default_rng(5).standard_normal(3000)


def comb(x, a=0.5, D=4, name="comb"):
    """y[n] = x[n] + a y[n - D]; returns (diagram, scope, blocks)."""
    src, add, dl, g, sc = (
        sl.FromArray(x, name="src"),
        sl.Sum(name="add"),
        sl.Delay(D, name="z"),
        sl.Gain(a, name="g"),
        sl.Scope(name),
    )
    src >> add["x0"]
    add >> dl >> g >> add["x1"]
    add >> sc
    return Diagram(sc, fs=1.0), sc, (src, add, dl, g)


# unit


def test_cycles_finds_each_loop_once():
    a, b, d, e = (Scale(name=n) for n in "abde")
    c = Add(name="c")
    a >> b >> a
    c >> d >> e >> c["a"]
    b >> c["b"]
    loops = cycles([a, b, c, d, e])
    names = sorted([x.name for x in lp] for lp in loops)
    assert names == [["a", "b"], ["d", "e", "c"]]  # each by creation
    assert cycles([a, c]) == []  # lines to blocks outside the list do not count


def test_cluster_schedules_state_blocks_first():
    d, sc, (src, add, dl, g) = comb(X)
    (lp,) = d.loops()
    assert isinstance(lp, LoopCluster) and lp.mode == "python"
    assert lp.members == [add, dl, g] and lp.name == "loop[add, z, g]"
    assert lp.state == [dl] and lp.order == [g, add]
    assert d.loops(chunk=4) == []  # a delay of 4 breaks the loop at chunk 4


def test_loop_no_block_delays_is_algebraic():
    a, b, rec = Scale(name="a"), Add(name="b"), Record()
    sl.Const(1.0) >> b["a"]
    b >> a >> b["b"]
    b >> rec
    with pytest.raises(AlgebraicLoopError, match="at one sample through a, b"):
        Diagram(rec, fs=1.0).run(n=8)
    with pytest.raises(ValueError, match="compile_loops"):
        Diagram(rec, fs=1.0, compile_loops="numba")


def test_loop_across_sample_rates_needs_a_long_delay():
    add, sc = sl.Sum(), sl.Scope()
    sl.Const(1.0) >> add["x0"]
    add >> sl.Upsample(2) >> sl.Downsample(2) >> sl.Delay(1) >> add["x1"]
    add >> sc
    with pytest.raises(AlgebraicLoopError, match="several sample rates"):
        Diagram(sc, fs=1.0).run(n=8)


def test_compile_loops_switch_takes_effect_at_once():
    d, sc, _ = comb(X)
    d.run(n=1000)
    d.compile_loops = False
    with pytest.raises(AlgebraicLoopError, match="loop at chunk=1000"):
        d.run(n=1000, chunk=1000)
    d.run(n=8, chunk=4)  # a delay of 4 breaks the loop at chunk 4


# integration


def test_two_loops_in_series_and_blocks_around():
    src, sc = sl.FromArray(X), sl.Scope()
    add1, add2 = sl.Sum(), sl.Sum("+-")
    src >> sl.Gain(2.0) >> add1["x0"]
    add1 >> sl.Delay(1) >> sl.Gain(0.5) >> add1["x1"]
    add1 >> add2["x0"]
    add2 >> sl.Delay(3) >> sl.Gain(0.25) >> add2["x1"]
    add2 >> sl.Gain(-1.0) >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=len(X), chunk=500)
    assert len(d.loops(500)) == 2
    y = sig.lfilter([1], [1, -0.5], 2 * X)
    y = sig.lfilter([1], [1, 0, 0, 0.25], y)
    np.testing.assert_allclose(sc.data, -y, atol=1e-12)


def test_probe_inside_a_loop():
    d, sc, (src, add, dl, g) = comb(X)
    p = d.probe(g)
    d.run(n=len(X))
    np.testing.assert_allclose(p.data, 0.5 * np.r_[np.zeros(4), sc.data[:-4]])


def test_complex_signals():
    z = X[:600] + 1j * X[::-1][:600]
    d, sc, _ = comb(z, a=0.5j)
    d.run(n=600)
    np.testing.assert_allclose(sc.data, sig.lfilter([1], [1, 0, 0, 0, -0.5j], z))


def test_error_inside_a_loop_names_the_block():
    def bad(x):
        raise ZeroDivisionError("boom")

    add, sc = sl.Sum(), sl.Scope()
    sl.Const(1.0) >> add["x0"]
    add >> sl.Delay(2) >> sl.Func(bad, name="bad") >> add["x1"]
    add >> sc
    with pytest.raises(ZeroDivisionError) as err:
        Diagram(sc, fs=1.0).run(n=8)
    assert any("in block bad" in n for n in err.value.__notes__)


# comparison


def test_per_sample_equals_vectorised():
    # chunk 4 breaks the loop at the delay; chunk 1000 needs a cluster
    d1, sc1, _ = comb(X)
    d1.run(n=len(X), chunk=4)
    d2, sc2, _ = comb(X)
    d2.run(n=len(X), chunk=1000)
    assert d1.loops(4) == [] and d2.loops(1000)
    np.testing.assert_array_equal(sc1.data, sc2.data)


def test_chunk_invariant():
    def make():
        add = sl.Sum(name="add")
        add >> sl.Delay(3) >> sl.Gain(-0.7) >> add["x1"]
        return sl.Subsystem(inputs={"x": add["x0"]}, outputs={"y": add})

    assert_chunk_invariant(make, X[:400], chunks=(1, 3, 7, 64, 400))


# theory


def test_comb_equals_lfilter():
    d, sc, _ = comb(X, a=-0.9, D=5)
    d.run(n=len(X))
    np.testing.assert_allclose(sc.data, sig.lfilter([1], np.r_[1, 0, 0, 0, 0, 0.9], X))


def test_loop_through_a_filter_equals_its_closed_loop_response():
    # y = x - h * y[n - 1], with h an FIR: denominator 1 + z^-1 H(z)
    h = np.array([0.3, -0.2, 0.1])
    add, sc = sl.Sum("+-"), sl.Scope()
    sl.FromArray(X) >> add["x0"]
    add >> sl.Delay(1) >> sl.FIR(h) >> add["x1"]
    add >> sc
    Diagram(sc, fs=1.0).run(n=len(X))
    np.testing.assert_allclose(sc.data, sig.lfilter([1], np.r_[1, h], X), atol=1e-12)


# compiled loops

needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba")


def loop_through(make, x=X, chunk=500, mode=True):
    """src -> add -> make() -> Delay(1) -> Gain(0.5) -> add; returns diagram, scope."""
    add, sc = sl.Sum("+-"), sl.Scope()
    sl.FromArray(x) >> add["x0"]
    add >> make() >> sl.Delay(1) >> sl.Gain(0.5) >> add["x1"]
    add >> sc
    d = Diagram(sc, fs=8.0, compile_loops=mode)
    d.run(n=len(x), chunk=chunk)
    return d, sc


KERNELS = {
    "gain": lambda: sl.Gain(-0.8),
    "gain complex": lambda: sl.Gain(0.5 + 0.5j),
    "sum": lambda: sl.Sum("-"),
    "func": lambda: sl.Func(np.tanh),
    "delay": lambda: sl.Delay(3),
    "delay 0": lambda: sl.Delay(0),
    "fir": lambda: sl.FIR([0.5, -0.3, 0.2, 0.1]),
    "fir 1 tap": lambda: sl.FIR([0.7]),
    "iir sos": lambda: sl.IIR(sos=sig.butter(3, 0.2, output="sos")),
    "iir ba": lambda: sl.IIR(b=[0.2, 0.1], a=[1.0, -0.5, 0.1]),
    "mixer cos": lambda: sl.mixer(1.0),
    "mixer exp-": lambda: sl.mixer(1.0, lo="exp-", phase=0.3),
}


@needs_numba
@pytest.mark.parametrize("name", KERNELS)
def test_kernel_equals_python(name):
    # comparison: every kernel against the per-sample python path
    d1, sc1 = loop_through(KERNELS[name])
    d2, sc2 = loop_through(KERNELS[name], mode="python")
    assert d1.loops(500)[0].mode == "numba" and d2.loops(500)[0].mode == "python"
    np.testing.assert_allclose(sc1.data, sc2.data, rtol=1e-12, atol=1e-12)


@needs_numba
def test_product_kernel_divides():
    def make(mode):
        prod, sc = sl.Product("*/"), sl.Scope()
        sl.FromArray(1 + X**2) >> prod["x0"]
        prod >> sl.Delay(1, init=1.0) >> sl.Func(lambda v: 1 + 0.5 * v) >> prod["x1"]
        prod >> sc
        Diagram(sc, fs=1.0, compile_loops=mode).run(n=len(X), chunk=500)
        return sc.data

    np.testing.assert_allclose(make(True), make("python"), rtol=1e-12)


@needs_numba
def test_first_chunk_in_python_then_compiled_and_after_reset():
    d, sc, _ = comb(X)
    d.run(n=500, chunk=500)
    (lp,) = d.loops(500)
    assert lp.mode == "python" and lp.source is None
    d.run(n=500, chunk=500)
    assert lp.mode == "numba" and "def loop(" in lp.source
    d.reset().run(n=500, chunk=500)
    assert lp.mode == "python"  # state was made anew
    d.run(n=len(X) - 500, chunk=500)
    assert lp.mode == "numba"
    np.testing.assert_allclose(sc.data, sig.lfilter([1], [1, 0, 0, 0, -0.5], X))


@needs_numba
def test_state_and_parameters_move_between_block_and_kernel():
    # compiled, then python, with a gain change between: as all python
    def run(first):
        d, sc, (src, add, dl, g) = comb(X, D=3)
        d.compile_loops = first
        d.run(n=1500, chunk=500)
        g.g = -0.3
        d.compile_loops = "python"
        d.run(n=1500, chunk=500)
        return sc.data, dl.buf

    (y1, buf1), (y2, buf2) = run(True), run("python")
    np.testing.assert_allclose(y1, y2, rtol=1e-12)
    np.testing.assert_allclose(buf1, buf2, rtol=1e-12)


@needs_numba
def test_compiled_survives_a_rebuild():
    d, sc, _ = comb(X)
    d.run(n=1000, chunk=500)
    (lp,) = d.loops(500)
    sl.Gain(1.0) >> sl.Scope()  # lines elsewhere make diagrams rebuild
    assert d.loops(500) == [lp] and lp.mode == "numba"


@needs_numba
def test_block_without_kernel_warns_and_runs_in_python():
    class Half(sl.Block):
        def step(self, x):
            return 0.5 * x

    add, sc = sl.Sum(), sl.Scope()
    sl.FromArray(X) >> add["x0"]
    add >> sl.Delay(4) >> Half(name="half") >> add["x1"]
    add >> sc
    d = Diagram(sc, fs=1.0)
    with pytest.warns(
        RuntimeWarning, match="runs in Python.*no compiled kernel for Half"
    ):
        d.run(n=len(X), chunk=500)
    assert d.loops(500)[0].mode == "python"
    np.testing.assert_allclose(sc.data, sig.lfilter([1], [1, 0, 0, 0, -0.5], X))


@needs_numba
def test_func_numba_cannot_type_warns_and_runs_in_python():
    class Table:
        k = 1.0

    t = Table()

    def table(x):
        return t.k * x  # numba cannot type a plain python object

    with pytest.warns(RuntimeWarning, match="runs in Python"):
        d, sc = loop_through(lambda: sl.Func(table), chunk=500)
    assert d.loops(500)[0].mode == "python"
    np.testing.assert_allclose(sc.data, sig.lfilter([1], [1, 0.5], X))


@needs_numba
def test_compiled_pi_control_loop_equals_closed_loop_transfer_function():
    # theory: notebook 02's loop, step response against T = L / (1 + L)
    fs_c, kp, ki, a = 1000.0, 0.8, 40.0, 0.95
    b_pi, a_pi, b_p, a_p = [kp + ki / fs_c, -kp], [1.0, -1.0], [1 - a], [1.0, -a]
    err, plant, y = sl.Sum("+-"), sl.IIR(b=b_p, a=a_p), sl.Scope()
    sl.Step() >> err["x0"]
    err >> sl.IIR(b=b_pi, a=a_pi) >> plant >> y
    plant >> sl.Delay(1) >> err["x1"]
    d = Diagram(y, fs=fs_c)
    d.run(n=3000, chunk=1000)
    assert d.loops(1000)[0].mode == "numba"
    num = np.polymul(np.polymul(b_pi, b_p), [0.0, 1.0])
    den = np.polyadd(np.polymul(a_pi, a_p), num)
    np.testing.assert_allclose(y.data, sig.lfilter(num, den, np.ones(3000)), atol=1e-12)
