import numpy as np
import pytest

import sigline as sl
from sigline import Diagram, RandomSymbols, Scope
from sigline.testing import assert_chunk_invariant


def record(src, n, fs=1.0, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=fs).run(n=n, chunk=chunk)
    return sc.data


# unit


def test_points_types_and_info():
    x = RandomSymbols(0.25, sl.psk(4), fs=1.0)(40)
    assert x.dtype == complex and np.all(np.isin(x[::4], sl.psk(4)))
    assert np.all(x[np.arange(40) % 4 != 0] == 0)
    assert RandomSymbols(1.0, [-1, 1], fs=1.0)(5).dtype.kind == "i"
    assert RandomSymbols(2400.0).info() == "2.4 kBd, Gaussian"
    assert RandomSymbols(2400.0, sl.qam(16)).info() == "2.4 kBd, 16 points"


@pytest.mark.parametrize("points", [None, [-1.0, 1.0]])
def test_chunk_invariant(points):
    assert_chunk_invariant(lambda: RandomSymbols(0.37, points, seed=2), 600)


# comparison: one symbol drawn per symbol instant, from one stream


def test_equals_numpy_draws_at_rounded_instants():
    rs, n = 0.137, 5000
    x = record(RandomSymbols(rs, sl.qam(16), seed=4), n, chunk=77)
    k = np.arange(int(n * rs) + 2)
    idx = np.rint(k / rs).astype(int)
    k, idx = k[idx < n], idx[idx < n]
    want = np.zeros(n, complex)
    want[idx] = sl.qam(16)[np.random.default_rng(4).integers(16, size=len(k))]
    np.testing.assert_array_equal(x, want)


# theory: symbol rate and unit power of gaussian symbols


def test_rate_and_power():
    fs, rs, n = 48e3, 3288.0, 480_000
    x = record(RandomSymbols(rs, seed=1), n, fs=fs, chunk=4096)
    on = x != 0
    assert abs(on.sum() - n * rs / fs) <= 1
    assert abs(np.mean(np.abs(x[on]) ** 2) - 1) < 5 * np.sqrt(1 / on.sum())
