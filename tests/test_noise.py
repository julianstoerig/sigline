import numpy as np
import pytest

from sigline import Diagram, Noise, Scope
from sigline.testing import assert_chunk_invariant


def record(src, n, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=1.0).run(n=n, chunk=chunk)
    return sc.data


# unit


def test_seed_reproduces_and_differs():
    a, b, c = Noise(seed=1)(100), Noise(seed=1)(100), Noise(seed=2)(100)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_output_is_stable_until_update_and_reset_replays():
    n = Noise(seed=3)
    np.testing.assert_array_equal(n.output(5), n.output(5))
    first = n(5)
    n(5)
    n.reset()
    np.testing.assert_array_equal(n(5), first)


def test_complex_dtype_and_std_scaling():
    z = Noise(2.0, complex=True)(10)
    assert z.dtype == complex
    np.testing.assert_allclose(Noise(2.0, seed=4)(10), 2 * Noise(1.0, seed=4)(10))


@pytest.mark.parametrize("cplx", [False, True])
def test_chunk_invariant(cplx):
    assert_chunk_invariant(lambda: Noise(0.7, seed=9, complex=cplx), 600)


# comparison: equals numpy's generator called once


def test_equals_numpy_generator():
    x = record(Noise(1.5, seed=11), 5000, chunk=333)
    np.testing.assert_allclose(x, 1.5 * np.random.default_rng(11).standard_normal(5000))
    z = record(Noise(1.0, seed=12, complex=True), 5000, chunk=333)
    iq = np.random.default_rng(12).standard_normal((5000, 2))
    np.testing.assert_allclose(z, (iq[:, 0] + 1j * iq[:, 1]) / np.sqrt(2))


# theory: moments; tolerances are 5 standard errors for n = 200 000


def test_real_noise_moments():
    n, std = 200_000, 0.8
    x = record(Noise(std, seed=5), n)
    assert abs(np.mean(x)) < 5 * std / np.sqrt(n)
    assert abs(np.var(x) / std**2 - 1) < 5 * np.sqrt(2 / n)
    # white: lag-1 autocorrelation near zero
    assert abs(np.mean(x[1:] * x[:-1]) / std**2) < 5 / np.sqrt(n)


def test_complex_noise_is_circular():
    n, std = 200_000, 1.3
    z = record(Noise(std, seed=6, complex=True), n)
    assert abs(np.mean(np.abs(z) ** 2) / std**2 - 1) < 5 * np.sqrt(1 / n)
    # circular: E[z^2] = 0, i.e. equal I/Q power and no I/Q correlation
    assert abs(np.mean(z**2)) / std**2 < 5 / np.sqrt(n)
