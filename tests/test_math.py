import numpy as np
import pytest

from sigline import (
    Const,
    Diagram,
    FromArray,
    FuncChunk,
    Gain,
    Product,
    Scope,
    Sine,
    Sum,
    Unwrap,
)
from sigline.testing import assert_chunk_invariant

rng = np.random.default_rng(5)
XR = rng.standard_normal(400)
XC = XR + 1j * rng.standard_normal(400)


def through(block, x, chunk=37):
    src, sc = FromArray(x), Scope()
    src >> block >> sc
    Diagram(sc, fs=1.0).run(n=len(x), chunk=chunk)
    return sc.data


# gain


def test_gain_scalar_complex_and_per_channel():
    np.testing.assert_allclose(Gain(2.5)(XR), 2.5 * XR)
    np.testing.assert_allclose(Gain(1j)(XC), 1j * XC)
    v = np.ones((3, 2))
    np.testing.assert_allclose(Gain(np.array([1.0, -2.0]))(v), [[1, -2]] * 3)


def test_gain_change_between_runs():
    g, sc = Gain(1.0), Scope()
    FromArray(np.ones(4)) >> g >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=2)
    g.g = 3.0
    d.run(n=2)
    np.testing.assert_array_equal(sc.data, [1, 1, 3, 3])


def test_gain_chunk_invariant():
    assert_chunk_invariant(lambda: Gain(1.7 - 0.2j), XC)


def test_gain_in_diagram_equals_standalone():
    np.testing.assert_allclose(through(Gain(-0.3), XC), Gain(-0.3)(XC))


def test_gain_scales_sine_amplitude():
    # theory: k cos(.) has peak k and rms k / sqrt(2) over whole periods
    s, sc = Sine(100.0), Scope()
    s >> Gain(4.0) >> sc
    Diagram(sc, fs=8000.0).run(n=8000, chunk=256)
    assert np.isclose(np.max(np.abs(sc.data)), 4.0)
    assert np.isclose(np.sqrt(np.mean(sc.data**2)), 4 / np.sqrt(2))


# sum


def test_sum_signs_and_ports():
    s = Sum("+-+")
    assert s.inputs == ("x0", "x1", "x2")
    np.testing.assert_allclose(s(XR, XR, XR), XR)
    np.testing.assert_allclose(Sum()(XR, XC), XR + XC)


@pytest.mark.parametrize("signs", ["", "+*", "+ -", "x"])
def test_sum_rejects_invalid_signs(signs):
    with pytest.raises(ValueError, match="signs must be"):
        Sum(signs)


def test_sum_rejects_different_widths_in_a_diagram():
    s = Sum()
    FromArray(np.ones((4, 3))) >> s["x0"]  # one value per sample would expand
    FromArray(np.ones((4, 2))) >> s["x1"]
    with pytest.raises(ValueError, match="input widths differ"):
        Diagram(s, fs=1.0).run(n=4)


def test_sum_chunk_invariant():
    assert_chunk_invariant(lambda: Sum("+-"), {"x0": XC, "x1": XR})


def test_sum_equals_gains_into_sum():
    # comparison: Sum("+-") == Sum("++") with Gain(-1) on the second input
    a, b = Scope(), Scope()
    s1 = Sum("+-")
    FromArray(XR) >> s1["x0"]
    FromArray(XC) >> s1["x1"]
    s1 >> a
    s2 = Sum("++")
    FromArray(XR) >> s2["x0"]
    FromArray(XC) >> Gain(-1) >> s2["x1"]
    s2 >> b
    Diagram(a, fs=1.0).run(n=400, chunk=33)
    Diagram(b, fs=1.0).run(n=400, chunk=33)
    np.testing.assert_allclose(a.data, b.data)


def test_sum_of_shifted_sines_cancels():
    # theory: cos(phi) + cos(phi + pi) = 0
    s, sc = Sum(), Scope()
    Sine(440.0) >> s["x0"]
    Sine(440.0, phase=np.pi) >> s["x1"]
    s >> sc
    Diagram(sc, fs=8000.0).run(n=4000, chunk=128)
    assert np.max(np.abs(sc.data)) < 1e-12


# product


def test_product_ports_and_values():
    p = Product(3)
    assert p.inputs == ("x0", "x1", "x2")
    np.testing.assert_allclose(p(XR, XR, XC), XR**2 * XC)
    np.testing.assert_allclose(Product()(XC, np.conj(XC)), np.abs(XC) ** 2)


def test_product_divides_by_ops():
    p = Product("*/*")
    assert p.inputs == ("x0", "x1", "x2") and p.info() == "* / *"
    np.testing.assert_allclose(p(XC, XR, XR), XC)
    assert Product(3).ops == "***"
    with pytest.raises(ValueError, match="ops must be"):
        Product("*+")


def test_division_by_a_sine_equals_multiplying_by_its_inverse():
    # comparison: x / s == x * (1 / s) where s has no zeros
    s1, s2, a, b = Sine(1.0, phase=0.3), Sine(1.0, phase=0.3), Scope(), Scope()
    p1, p2 = Product("*/"), Product()
    FromArray(XR) >> p1["x0"]
    s1 >> FuncChunk(lambda x: 2.0 + x) >> p1["x1"]
    FromArray(XR) >> p2["x0"]
    s2 >> FuncChunk(lambda x: 1 / (2.0 + x)) >> p2["x1"]
    p1 >> a
    p2 >> b
    Diagram(a, fs=100.0).run(n=400, chunk=33)
    Diagram(b, fs=100.0).run(n=400, chunk=33)
    np.testing.assert_allclose(a.data, b.data, rtol=1e-12)


def test_product_rejects_different_widths_in_a_diagram():
    p = Product()
    FromArray(np.ones((4, 2))) >> p["x0"]  # one value per sample would expand
    FromArray(np.ones((4, 3))) >> p["x1"]
    with pytest.raises(ValueError, match="input widths differ"):
        Diagram(p, fs=1.0).run(n=4)


def test_product_chunk_invariant():
    assert_chunk_invariant(lambda: Product(), {"x0": XC, "x1": XR})


def test_product_with_constant_equals_gain():
    # comparison: x * Const(k) == Gain(k)
    p, a = Product(), Scope()
    FromArray(XC) >> p["x0"]
    Const(-2.5) >> p["x1"]
    p >> a
    Diagram(a, fs=1.0).run(n=400, chunk=50)
    np.testing.assert_allclose(a.data, through(Gain(-2.5), XC))


def test_product_of_sines_is_sum_and_difference_frequency():
    # theory: cos(a) cos(b) = (cos(a - b) + cos(a + b)) / 2
    fs, f1, f2 = 8000.0, 1000.0, 300.0
    p, sc = Product(), Scope()
    Sine(f1) >> p["x0"]
    Sine(f2) >> p["x1"]
    p >> sc
    Diagram(sc, fs=fs).run(n=2000, chunk=123)
    t = np.arange(2000) / fs
    ref = 0.5 * (np.cos(2 * np.pi * (f1 - f2) * t) + np.cos(2 * np.pi * (f1 + f2) * t))
    np.testing.assert_allclose(sc.data, ref, atol=1e-9)


# func


def test_func_ports_from_signature_and_ufuncs():
    def f(a, b, scale=2.0):
        return scale * (a - b)

    fb = FuncChunk(f)
    assert fb.inputs == ("a", "b") and fb.name.startswith("f")
    np.testing.assert_allclose(fb(XR, XR / 2), XR)
    assert FuncChunk(np.abs).inputs == ("x",)
    np.testing.assert_allclose(FuncChunk(np.abs)(XC), np.abs(XC))


def test_func_several_outputs_and_explicit_ports():
    pol = FuncChunk(lambda z: (np.abs(z), np.angle(z)), outputs=("mag", "ph"))
    m, ph = pol(XC)
    np.testing.assert_allclose(m * np.exp(1j * ph), XC)
    g = FuncChunk(lambda *a: a[0] * a[1], inputs=("u", "v"))
    assert g.inputs == ("u", "v")
    np.testing.assert_allclose(g(XR, XR), XR**2)


def test_func_y_argument_needs_explicit_ports():
    f = FuncChunk(lambda y: 2 * y)
    with pytest.raises(KeyError, match="both an input and an output"):
        f["y"]
    src, sc = FromArray(XR), Scope()
    src >> f.inp("y")
    f.out("y") >> sc
    Diagram(sc, fs=1.0).run(n=400)
    np.testing.assert_allclose(sc.data, 2 * XR)


def test_func_chunk_invariant():
    assert_chunk_invariant(
        lambda: FuncChunk(lambda i, q: i + 1j * q), {"i": XR, "q": XR[::-1]}
    )


def test_func_in_diagram_equals_plain_call():
    np.testing.assert_allclose(through(FuncChunk(np.tanh), XR), np.tanh(XR))


def test_func_builds_complex_exponential_from_iq():
    # theory: cos + j sin = exp(j phi)
    iq, sc = FuncChunk(lambda i, q: i + 1j * q), Scope()
    Sine(250.0) >> iq["i"]
    Sine(250.0, phase=-np.pi / 2) >> iq["q"]
    iq >> sc
    Diagram(sc, fs=8000.0).run(n=1000, chunk=99)
    ref = np.exp(2j * np.pi * 250.0 * np.arange(1000) / 8000.0)
    np.testing.assert_allclose(sc.data, ref, atol=1e-12)


# plain functions in chains


def test_functions_in_chains_become_func_blocks():
    src, sc = FromArray(XC), Scope()
    f = src >> np.abs
    assert isinstance(f, FuncChunk) and f.name.startswith("absolute")
    f >> (lambda x: 2 * x) >> sc
    Diagram(sc, fs=1.0).run(n=400, chunk=50)
    np.testing.assert_allclose(sc.data, 2 * np.abs(XC))


def test_functions_right_to_left_equal_left_to_right():
    a, b = Scope(), Scope()
    FromArray(XR) >> np.tanh >> a
    b << np.tanh << FromArray(XR)
    Diagram(a, fs=1.0).run(n=400)
    Diagram(b, fs=1.0).run(n=400)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_allclose(a.data, np.tanh(XR))


# inputs reach step in port order, whatever order the lines were drawn in


def test_inputs_in_port_order_not_line_order():
    s, p, f, sc = (
        Sum("+-"),
        Product("*/"),
        FuncChunk(lambda a, b: a - 2 * b),
        [Scope() for _ in range(3)],
    )
    for blk, out in zip((s, p, f), sc, strict=True):
        Const(1.0) >> blk[blk.inputs[1]]  # second input first
        Const(4.0) >> blk[blk.inputs[0]]
        blk >> out
    Diagram(*sc, fs=1.0).run(n=2)
    assert [o.data[0] for o in sc] == [3.0, 4.0, 2.0]
    # standalone calls by keyword, out of order
    assert s(x1=np.array([1.0]), x0=np.array([4.0]))[0] == 3.0


# scalar signals next to vector signals


def test_sum_and_product_repeat_scalar_signals_over_channels():
    v = np.arange(6.0).reshape(3, 2)
    s = np.array([10.0, 20.0, 30.0])
    np.testing.assert_array_equal(Sum("+-")(v, s), v - s[:, None])
    np.testing.assert_array_equal(Product("*/")(v, s), v / s[:, None])
    # chunk as long as the width: plain broadcasting would pair s with channels
    np.testing.assert_array_equal(Product()(v[:2], s[:2]), v[:2] * s[:2, None])


def test_scalar_signal_meets_vector_signal_in_a_diagram():
    v = np.random.default_rng(1).standard_normal((50, 3))
    p, sc = Product(), Scope()
    FromArray(v) >> p["x0"]
    Sine(0.05) >> p["x1"]
    p >> sc
    Diagram(sc, fs=1.0).run(n=50, chunk=2)
    lo = np.cos(2 * np.pi * 0.05 * np.arange(50))
    np.testing.assert_allclose(sc.data, v * lo[:, None], atol=1e-12)


# unwrap


def test_unwrap_ramp_skip_and_vector():
    ph = np.angle(np.exp(1j * np.arange(0.0, 12.0, 2.0)))
    np.testing.assert_allclose(Unwrap()(ph), np.arange(0.0, 12.0, 2.0), atol=1e-12)
    y = Unwrap(skip=3)(ph)
    np.testing.assert_array_equal(y[:3], ph[:3])  # passed through wrapped
    np.testing.assert_allclose(y[3:], ph[2] + np.array([2.0, 4.0, 6.0]), atol=1e-12)
    v = np.stack([ph, -ph], axis=1)
    np.testing.assert_allclose(Unwrap()(v), np.unwrap(v, axis=0), atol=1e-12)
    assert Unwrap(skip=5).info() == "skip 5" and Unwrap().info() == ""


def test_unwrap_phase_of_a_tone_in_a_diagram():
    # theory: the phase of exp(j 2 pi f t) grows as 2 pi f t
    src, u, sc = Sine(0.37, complex=True), Unwrap(), Scope()
    src >> np.angle >> u >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=500, chunk=33)
    np.testing.assert_allclose(sc.data, 2 * np.pi * 0.37 * np.arange(500), atol=1e-9)


@pytest.mark.parametrize("skip", [0, 40])
def test_unwrap_equals_numpy_for_any_chunking(skip):
    ph = np.angle(np.exp(1j * np.cumsum(np.random.default_rng(2).normal(0.5, 1, 400))))
    ref = assert_chunk_invariant(lambda: Unwrap(skip=skip), ph, rtol=0, atol=0)["y"]
    want = ph.copy()
    want[max(skip - 1, 0) :] = np.unwrap(ph[max(skip - 1, 0) :])
    np.testing.assert_allclose(ref, want, atol=1e-12)


# func parameters, as for nonlinear


def test_func_arguments_with_defaults_are_parameters():
    f = FuncChunk(lambda x, k=2.0, *, off=0.0: k * x + off, params={"off": 1.0})
    assert f.inputs == ("x",) and f.params == {"k": 2.0, "off": 1.0}
    np.testing.assert_array_equal(f(np.array([1.0, 2.0])), [3.0, 5.0])
    f.params["k"] = -1.0
    np.testing.assert_array_equal(f(np.array([1.0])), [0.0])
    with pytest.raises(TypeError, match="no parameter"):
        FuncChunk(lambda x, k=2.0: k * x, params={"kk": 1.0})
    assert FuncChunk(np.abs).params == {} and FuncChunk(np.angle).params == {
        "deg": False
    }


def test_func_parameter_change_between_runs_in_a_diagram():
    f, sc = FuncChunk(lambda x, k=1.0: k * x), Scope()
    FromArray(np.ones(20)) >> f >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=10, chunk=3)
    f.params["k"] = 0.5
    d.run(n=10, chunk=3)
    np.testing.assert_array_equal(sc.data, [1.0] * 10 + [0.5] * 10)
