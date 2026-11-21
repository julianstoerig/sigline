import numpy as np
import pytest

from _blocks import Accumulate
from sigline import Const, Diagram, FromArray, Scope, Signal, Source, Step


class Ramp(Source):
    def generate(self, n):
        return n.astype(float)


# source base


def test_source_counts_samples_and_resets():
    r = Ramp()
    np.testing.assert_array_equal(np.r_[r(3), r(2)], np.arange(5))
    r.reset()
    np.testing.assert_array_equal(r(2), [0, 1])


def test_source_output_does_not_advance():
    r = Ramp()
    np.testing.assert_array_equal(r.output(3), r.output(3))
    assert r.k == 0


def test_generate_required():
    with pytest.raises(NotImplementedError, match="generate"):
        Source()(3)


# const


def test_const_values_and_dtype():
    np.testing.assert_array_equal(Const(2.5)(4), np.full(4, 2.5))
    assert Const(1 + 2j)(3).dtype == complex
    np.testing.assert_array_equal(Const()(2), [1.0, 1.0])


def test_const_in_diagram_integrates_to_ramp():
    # theory: the running sum of a constant c is c * (n + 1)
    c, sc = Const(0.5), Scope()
    c >> Accumulate() >> sc
    Diagram(sc, fs=1.0).run(n=40, chunk=7)
    np.testing.assert_allclose(sc.data, 0.5 * (np.arange(40) + 1))


def test_const_value_change_between_runs():
    c, sc = Const(1.0), Scope()
    c >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=3)
    c.value = -1.0
    d.run(n=2)
    np.testing.assert_array_equal(sc.data, [1, 1, 1, -1, -1])


# step


def test_step_switches_at_T():
    s = Step(0.3, after=2.0, before=-1.0, fs=10.0)
    np.testing.assert_array_equal(s(5), [-1, -1, -1, 2, 2])


def test_step_defaults_unit_step_at_zero():
    np.testing.assert_array_equal(Step(fs=1.0)(3), [1.0, 1.0, 1.0])


def test_step_between_samples_switches_at_next_sample():
    np.testing.assert_array_equal(Step(0.25, fs=10.0)(5), [0, 0, 0, 1, 1])


@pytest.mark.parametrize("chunk", [1, 4, 100])
def test_step_response_of_accumulator_is_ramp(chunk):
    # theory: running sum of a unit step at sample k0 is max(0, n - k0 + 1)
    st, sc = Step(0.5), Scope()
    st >> Accumulate() >> sc
    Diagram(sc, fs=20.0).run(n=60, chunk=chunk)
    n = np.arange(60)
    np.testing.assert_array_equal(sc.data, np.maximum(0, n - 10 + 1))


# signal


def test_signal_evaluates_fn_at_sample_times():
    np.testing.assert_allclose(Signal(np.cos, fs=4.0)(3), np.cos([0, 0.25, 0.5]))


def test_signal_complex_and_vector_outputs():
    z = Signal(lambda t: np.exp(1j * t), fs=1.0)(2)
    assert z.dtype == complex
    v = Signal(lambda t: np.stack([t, -t], axis=1), fs=1.0)(3)
    assert v.shape == (3, 2)


def test_signal_fn_can_be_swapped_between_runs():
    s, sc = Signal(lambda t: t), Scope()
    s >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=2)
    s.fn = lambda t: -t
    d.run(n=2)
    np.testing.assert_array_equal(sc.data, [0, 1, -2, -3])


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_signal_step_equals_step_source(chunk):
    a, b = Signal(lambda t: (t >= 0.35).astype(float)), Step(0.35)
    sa, sb = Scope(), Scope()
    a >> sa
    b >> sb
    Diagram(sa, fs=20.0).run(n=40, chunk=chunk)
    Diagram(sb, fs=20.0).run(n=40, chunk=chunk)
    np.testing.assert_array_equal(sa.data, sb.data)


def test_signal_sine_matches_closed_form():
    fs, f = 8000.0, 1234.5
    s, sc = Signal(lambda t: np.sin(2 * np.pi * f * t)), Scope()
    s >> sc
    Diagram(sc, fs=fs).run(0.1, chunk=500)
    n = np.arange(800)
    np.testing.assert_allclose(sc.data, np.sin(2 * np.pi * f * n / fs), atol=1e-12)


# from array


def test_fromarray_pads_with_zeros_or_loops():
    np.testing.assert_array_equal(FromArray([1, 2, 3])(5), [1, 2, 3, 0, 0])
    np.testing.assert_array_equal(
        FromArray([1, 2, 3], loop=True)(7), [1, 2, 3, 1, 2, 3, 1]
    )


def test_fromarray_keeps_dtype_and_vector_shape():
    z = FromArray(np.array([1j, 2j]))(3)
    assert z.dtype == complex and z[2] == 0
    v = FromArray(np.ones((4, 3)))(6)
    assert v.shape == (6, 3) and np.all(v[4:] == 0)


def test_fromarray_reset_plays_again():
    s = FromArray([5.0, 6.0])
    s(2)
    s.reset()
    np.testing.assert_array_equal(s(2), [5, 6])


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_fromarray_through_diagram_equals_data(chunk):
    x = np.random.default_rng(0).standard_normal(1000)
    src, sc = FromArray(x), Scope()
    src >> sc
    Diagram(sc, fs=1.0).run(n=1000, chunk=chunk)
    np.testing.assert_array_equal(sc.data, x)


def test_fromarray_into_accumulator_matches_cumsum():
    x = np.random.default_rng(1).standard_normal(300)
    src, sc = FromArray(x), Scope()
    src >> Accumulate() >> sc
    Diagram(sc, fs=1.0).run(n=300, chunk=37)
    np.testing.assert_allclose(sc.data, np.cumsum(x), atol=1e-12)
