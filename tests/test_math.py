import numpy as np
import pytest

from sigline import Const, Diagram, FromArray, Func, Gain, Product, Scope, Sine, Sum
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


def test_sum_rejects_different_widths_in_a_diagram():
    s = Sum()
    FromArray(np.ones(4)) >> s["x0"]
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


def test_product_rejects_different_widths_in_a_diagram():
    p = Product()
    FromArray(np.ones(4)) >> p["x0"]
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

    fb = Func(f)
    assert fb.inputs == ("a", "b") and fb.name.startswith("f")
    np.testing.assert_allclose(fb(XR, XR / 2), XR)
    assert Func(np.abs).inputs == ("x",)
    np.testing.assert_allclose(Func(np.abs)(XC), np.abs(XC))


def test_func_several_outputs_and_explicit_ports():
    pol = Func(lambda z: (np.abs(z), np.angle(z)), outputs=("mag", "ph"))
    m, ph = pol(XC)
    np.testing.assert_allclose(m * np.exp(1j * ph), XC)
    g = Func(lambda *a: a[0] * a[1], inputs=("u", "v"))
    assert g.inputs == ("u", "v")
    np.testing.assert_allclose(g(XR, XR), XR**2)


def test_func_y_argument_needs_explicit_ports():
    f = Func(lambda y: 2 * y)
    with pytest.raises(KeyError, match="both an input and an output"):
        f["y"]
    src, sc = FromArray(XR), Scope()
    src >> f.inp("y")
    f.out("y") >> sc
    Diagram(sc, fs=1.0).run(n=400)
    np.testing.assert_allclose(sc.data, 2 * XR)


def test_func_chunk_invariant():
    assert_chunk_invariant(
        lambda: Func(lambda i, q: i + 1j * q), {"i": XR, "q": XR[::-1]}
    )


def test_func_in_diagram_equals_plain_call():
    np.testing.assert_allclose(through(Func(np.tanh), XR), np.tanh(XR))


def test_func_builds_complex_exponential_from_iq():
    # theory: cos + j sin = exp(j phi)
    iq, sc = Func(lambda i, q: i + 1j * q), Scope()
    Sine(250.0) >> iq["i"]
    Sine(250.0, phase=-np.pi / 2) >> iq["q"]
    iq >> sc
    Diagram(sc, fs=8000.0).run(n=1000, chunk=99)
    ref = np.exp(2j * np.pi * 250.0 * np.arange(1000) / 8000.0)
    np.testing.assert_allclose(sc.data, ref, atol=1e-12)
