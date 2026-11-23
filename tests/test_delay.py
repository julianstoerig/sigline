from contextlib import nullcontext

import numpy as np
import pytest

from sigline import (
    AlgebraicLoopError,
    Const,
    Delay,
    Diagram,
    FromArray,
    Gain,
    Scope,
    Sine,
    Sum,
)
from sigline._numba import HAVE_NUMBA
from sigline.testing import assert_chunk_invariant

X = np.random.default_rng(2).standard_normal(500)


# unit


def test_delays_and_fills_with_init():
    np.testing.assert_array_equal(Delay(2, init=9.0)(np.arange(5.0)), [9, 9, 0, 1, 2])
    np.testing.assert_array_equal(Delay(0)(np.arange(3.0)), [0, 1, 2])


def test_vector_and_complex_signals():
    v = np.arange(12.0).reshape(6, 2)
    np.testing.assert_array_equal(Delay(2)(v), np.r_[np.zeros((2, 2)), v[:4]])
    z = Delay(1)(np.array([1j, 2j]))
    np.testing.assert_array_equal(z, [0, 1j])


def test_feedthrough_depends_on_chunk_length():
    d = Delay(4)
    assert not d.feedthrough(4) and d.feedthrough(5)


@pytest.mark.parametrize("n", [0, 1, 13, 100])
def test_chunk_invariant(n):
    assert_chunk_invariant(lambda: Delay(n), X + 1j * X[::-1])


# integration: feedback through a delay; the loop limit


def test_feedback_comb_equals_recursion():
    # comparison: y[n] = x[n] + 0.5 y[n - D] in the diagram and as a python loop
    D = 16
    src, add, dl, sc = FromArray(X), Sum(), Delay(D), Scope()
    src >> add["x0"]
    add >> sc
    add >> dl >> Gain(0.5) >> add["x1"]
    Diagram(sc, fs=1.0).run(n=500, chunk=D)
    y = np.zeros(500)
    for i in range(500):
        y[i] = X[i] + (0.5 * y[i - D] if i >= D else 0.0)
    np.testing.assert_allclose(sc.data, y, atol=1e-12)


def test_loop_with_delay_shorter_than_chunk():
    # per sample at chunk 8, vectorised at chunk 4: same result; or an error
    add, sc = Sum(), Scope()
    Const(1.0) >> add["x0"]
    add >> Delay(4) >> add["x1"]
    add >> sc
    Diagram(sc, fs=1.0).run(n=16, chunk=8)
    np.testing.assert_array_equal(sc.data, np.arange(16) // 4 + 1)
    sc.reset()
    Diagram(sc, fs=1.0).reset().run(n=16, chunk=4)
    np.testing.assert_array_equal(sc.data, np.arange(16) // 4 + 1)
    with pytest.raises(AlgebraicLoopError):
        Diagram(sc, fs=1.0, compile_loops=False).run(n=16, chunk=8)


# theory: a delayed sine is a phase-shifted sine


def test_delayed_sine_is_phase_shifted():
    fs, f, D = 8000.0, 440.0, 7
    s, sc = Sine(f), Scope()
    s >> Delay(D) >> sc
    Diagram(sc, fs=fs).run(n=1000, chunk=64)
    n = np.arange(D, 1000)
    np.testing.assert_allclose(
        sc.data[D:], np.cos(2 * np.pi * f * (n - D) / fs), atol=1e-12
    )


# delay in seconds


def test_tau_sets_the_delay_in_seconds():
    d = Delay(tau=0.003, fs=1000.0)
    assert d.n == 3 and d.info() == "3 ms"
    np.testing.assert_array_equal(d(np.arange(1.0, 6.0)), [0, 0, 0, 1, 2])
    with pytest.raises(ValueError, match="not a whole number"):
        _ = Delay(tau=0.0015, fs=1000.0).n
    with pytest.raises(ValueError, match="tau needs fs"):
        _ = Delay(tau=0.1).n
    d.n = 2  # setting n replaces tau
    assert d.tau is None and d.n == 2


def test_tau_and_n_give_the_same_feedback_loop():
    # comparison: Delay(tau=0.01) at 1 kHz is Delay(10); theory: impulse
    # response of y = x - 0.5 y[n - 10] is (-0.5)^k at n = 10 k
    def run(delay):
        add, sc = Sum(), Scope()
        src = FromArray(np.r_[1.0, np.zeros(99)])
        src >> add["x0"]
        add >> delay >> Gain(-0.5) >> add["x1"]
        add >> sc
        Diagram(sc, fs=1000.0).run(n=100, chunk=10)
        return sc.data

    a, b = run(Delay(tau=0.01)), run(Delay(10))
    np.testing.assert_array_equal(a, b)
    k = np.arange(100)
    np.testing.assert_allclose(a, np.where(k % 10 == 0, (-0.5) ** (k // 10), 0.0))


def test_tau_delay_chunk_invariant():
    assert_chunk_invariant(lambda: Delay(tau=0.013), X, fs=1000.0)


def test_vector_init_gives_a_vector_signal_from_the_start():
    np.testing.assert_array_equal(Delay(2, init=[1.0, -1.0])(np.zeros((3, 2))),
                                  [[1, -1], [1, -1], [0, 0]])  # fmt: skip


@pytest.mark.parametrize("chunk", [3, 100])
def test_vector_loop_equals_each_channel(chunk):
    # a loop of vector signals: per sample at chunk 100, vectorised at 3
    import scipy.signal as sig

    v = np.stack([X, -0.5 * X], axis=1)
    add, sc = Sum(), Scope()
    FromArray(v) >> add["x0"]
    add >> Delay(3, init=np.zeros(2)) >> Gain(0.5) >> add["x1"]
    add >> sc
    # per sample, vector signals stay in python, with a warning
    slow = HAVE_NUMBA and chunk > 3
    with pytest.warns(RuntimeWarning, match="vector") if slow else nullcontext():
        Diagram(sc, fs=1.0).run(n=len(X), chunk=chunk)
    ref = sig.lfilter([1], [1, 0, 0, -0.5], v, axis=0)
    np.testing.assert_allclose(sc.data, ref, atol=1e-12)
