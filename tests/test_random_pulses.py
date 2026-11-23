import numpy as np
import pytest

from sigline import Diagram, RandomPulses, Scope
from sigline.testing import assert_chunk_invariant


def record(src, n, fs=1.0, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=fs).run(n=n, chunk=chunk)
    return sc.data


def by_loop(u, p, n):
    # each sample below p starts a pulse of n samples
    on, left = np.zeros(len(u)), 0
    for i, ui in enumerate(u):
        if ui < p:
            left = n
        if left:
            on[i], left = 1.0, left - 1
    return on


# unit


def test_certain_and_impossible_starts_and_arguments():
    np.testing.assert_array_equal(RandomPulses(p=1.0, fs=1.0)(4), np.ones(4))
    np.testing.assert_array_equal(RandomPulses(p=0.0, fs=1.0)(4), np.zeros(4))
    assert RandomPulses(rate=5.0, fs=100.0).p == 0.05
    assert RandomPulses(rate=5.0, duration=2e-3).info() == "5/s, 2 ms"
    with pytest.raises(ValueError, match="exactly one"):
        RandomPulses(rate=1.0, p=0.1)
    with pytest.raises(ValueError, match="exactly one"):
        RandomPulses()


def test_a_pulse_runs_on_into_the_next_chunk():
    src = RandomPulses(p=0.02, duration=30.0, seed=4, fs=1.0)
    a = np.concatenate([src(7) for _ in range(40)])
    src.reset()
    np.testing.assert_array_equal(src(280), a)
    assert a.any() and not a.all()


def test_chunk_invariant():
    assert_chunk_invariant(lambda: RandomPulses(p=0.05, duration=6.0, seed=2), 600)


# comparison: equals a loop over the same uniform draws


@pytest.mark.parametrize("duration", [0.0, 1.0, 9.0])
def test_equals_a_loop(duration):
    x = record(RandomPulses(p=0.03, duration=duration, seed=6), 5000, chunk=77)
    u = np.random.default_rng(6).random(5000)
    np.testing.assert_array_equal(x, by_loop(u, 0.03, max(1, int(duration))))


# theory: a sample is on if a pulse started in the last n samples


def test_occupancy_and_start_rate():
    fs, rate, n, N = 1e5, 50.0, 100, 2_000_000
    x = record(RandomPulses(rate, duration=n / fs, seed=2), N, fs=fs, chunk=8192)
    p = rate / fs
    occupied = 1 - (1 - p) ** n
    # about 1000 pulses, so a 3 % standard error
    assert abs(x.mean() / occupied - 1) < 0.1
