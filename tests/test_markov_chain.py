import numpy as np
import pytest

from sigline import Diagram, MarkovChain, Scope
from sigline.testing import assert_chunk_invariant

GE = [[0.999, 0.001], [0.02, 0.98]]  # gilbert-elliott: rare, long bursts


def record(src, n, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=1.0).run(n=n, chunk=chunk)
    return sc.data


def by_loop(u, P, state):
    out = []
    for ui in u:
        out.append(state)
        state = int(np.searchsorted(np.cumsum(P[state]), ui, side="right"))
    return np.array(out)


# unit


def test_certain_transitions_and_arguments():
    np.testing.assert_array_equal(MarkovChain(np.eye(3), start=2, fs=1.0)(3), [2, 2, 2])
    cycle = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    np.testing.assert_array_equal(MarkovChain(cycle, fs=1.0)(5), [0, 1, 2, 0, 1])
    assert MarkovChain(GE).info() == "2 states"
    with pytest.raises(ValueError, match="square"):
        MarkovChain([[1.0, 0.0]])
    with pytest.raises(ValueError, match="sum to 1"):
        MarkovChain([[0.5, 0.4], [0.0, 1.0]])


def test_chunk_invariant():
    assert_chunk_invariant(lambda: MarkovChain(GE, seed=3), 2000)


# comparison: equals a python loop over the same draws


def test_equals_a_loop():
    P = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    x = record(MarkovChain(P, seed=5), 5000, chunk=77)
    np.testing.assert_array_equal(
        x, by_loop(np.random.default_rng(5).random(5000), P, 0)
    )


# theory: time in the bad state and mean burst length


def test_gilbert_elliott_statistics():
    chain = MarkovChain(GE, seed=4)
    bad = record(chain, 2_000_000, chunk=8192)
    p_gb, p_bg = 0.001, 0.02
    assert chain.stationary()[1] == pytest.approx(p_gb / (p_gb + p_bg))
    # about 1900 bursts of geometric length: 2.3 % standard error of their
    # mean, about 3 % for the time in bursts
    assert abs(bad.mean() / (p_gb / (p_gb + p_bg)) - 1) < 0.1
    edges = np.diff(np.r_[0, bad, 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    assert abs(lengths.mean() * p_bg - 1) < 0.05
