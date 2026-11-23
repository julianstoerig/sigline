import numpy as np

from sigline import Diagram, Pulses, Scope
from sigline.testing import assert_chunk_invariant


def record(src, n, fs, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=fs).run(n=n, chunk=chunk)
    return sc.data


# unit


def test_whole_sample_edges_and_info():
    # 3 samples on every 10, from sample 4; edges are exact multiples
    x = Pulses(0.3, 1.0, delay=0.4, fs=10.0)(30)
    np.testing.assert_array_equal(np.flatnonzero(x), [4, 5, 6, 14, 15, 16, 24, 25, 26])
    assert Pulses(1e-3, 20e-3).info() == "1 ms / 20 ms"


def test_chunk_invariant():
    assert_chunk_invariant(lambda: Pulses(7.0, 23.0, delay=3.0), 600)


# comparison: equals a train built with numpy


def test_equals_a_tiled_pattern():
    x = record(Pulses(0.001, 0.02, fs=1e4), 10_000, 1e4, chunk=333)
    np.testing.assert_array_equal(x, np.tile(np.r_[np.ones(10), np.zeros(190)], 50))


# theory: duty cycle width / period


def test_duty_cycle():
    x = record(Pulses(1e-3, 7e-3, delay=2e-3), 700_000, 1e6)
    assert abs(x.mean() - 1 / 7) < 1e-5
