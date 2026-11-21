import numpy as np

from sigline import Diagram, FromArray, Gain, Scope, Sine
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
