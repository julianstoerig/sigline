from contextlib import nullcontext

import numpy as np
import pytest

from _blocks import Add, Delay, Ramp, Record, Scale
from sigline import AlgebraicLoopError, Diagram, StateBlock
from sigline._numba import HAVE_NUMBA


class Impulse(StateBlock):
    inputs = ()

    def reset(self):
        self.n = 0

    def output(self, L):
        return (np.arange(self.n, self.n + L) == 0).astype(float)

    def update(self, L):
        self.n += L


def comb(a, D):
    """y = x + a * y delayed by D, as a diagram."""
    src, add, dly, g, rec = Impulse(), Add(), Delay(D), Scale(a), Record()
    src >> add["a"]
    dly >> g >> add["b"]
    add >> dly
    add >> rec
    return Diagram(rec, fs=1.0), rec


# unit tests


def test_state_block_has_no_feedthrough_and_output_keeps_state():
    imp = Impulse()
    assert imp.feedthrough(1) is False
    np.testing.assert_array_equal(imp.output(3), [1, 0, 0])
    np.testing.assert_array_equal(imp.output(3), [1, 0, 0])
    np.testing.assert_array_equal(imp(3), [1, 0, 0])
    np.testing.assert_array_equal(imp(3), [0, 0, 0])


def test_state_block_standalone_delay():
    d = Delay(3)
    x = np.arange(1.0, 6.0)
    np.testing.assert_array_equal(d(x[:2]), [0, 0])
    np.testing.assert_array_equal(d(x[2:]), [0, 1, 2])


def test_missing_output_raises():
    with pytest.raises(NotImplementedError, match="output"):
        StateBlock()(np.zeros(2))


def test_schedule_splits_state_and_feedthrough_blocks():
    d, rec = comb(0.5, 4)
    nft, order = d.schedule(4)
    assert {type(b).__name__ for b in nft} == {"Impulse", "Delay"}
    assert all(b.feedthrough(4) for b in order)
    # a delay shorter than the chunk needs its input: the loop runs per sample
    nft, order = d.schedule(8)
    assert [type(b).__name__ for b in nft] == ["Impulse"]
    assert [type(x).__name__ for x in order] == ["LoopCluster", "Record"]
    with pytest.raises(AlgebraicLoopError):
        Diagram(rec, fs=1.0, compile_loops=False).schedule(8)


# integration: feedback through a delay, vectorised or per sample


def test_loop_with_delay_shorter_than_chunk_runs_per_sample():
    d, rec = comb(0.5, 4)
    # these test blocks have no compiled kernels: python, with a warning
    with (
        pytest.warns(RuntimeWarning, match="runs in Python")
        if HAVE_NUMBA
        else nullcontext()
    ):
        d.run(n=16, chunk=5)
    n = np.arange(16)
    np.testing.assert_array_equal(rec.data, 0.5 ** (n // 4) * (n % 4 == 0))
    with pytest.raises(AlgebraicLoopError, match="Delay"):
        Diagram(rec, fs=1.0, compile_loops=False).run(n=16, chunk=5)


def test_delay_shorter_than_chunk_works_outside_loops():
    ramp, rec = Ramp(), Record()
    ramp >> Delay(3) >> rec
    Diagram(rec, fs=1.0).run(n=20, chunk=8)
    np.testing.assert_array_equal(rec.data, np.r_[np.zeros(3), np.arange(17)])


# comparison: diagram loop versus a per-sample python recursion


@pytest.mark.parametrize("chunk", [1, 2, 5])
def test_feedback_loop_equals_sample_by_sample_recursion(chunk):
    a, D, N = -0.8, 5, 60
    x = np.random.default_rng(3).standard_normal(N)

    class Noise(StateBlock):
        inputs = ()

        def reset(self):
            self.i = 0

        def output(self, L):
            return x[self.i : self.i + L]

        def update(self, L):
            self.i += L

    src, add, dly, g, rec = Noise(), Add(), Delay(D), Scale(a), Record()
    src >> add["a"]
    dly >> g >> add["b"]
    add >> dly
    add >> rec
    Diagram(rec, fs=1.0).run(n=N, chunk=chunk)
    y = np.zeros(N)
    for i in range(N):
        y[i] = x[i] + (a * y[i - D] if i >= D else 0.0)
    np.testing.assert_allclose(rec.data, y, rtol=1e-12, atol=1e-12)


# theory: impulse response of the comb is a^k at multiples of D


@pytest.mark.parametrize("chunk", [1, 3, 4])
def test_comb_impulse_response_matches_closed_form(chunk):
    a, D, N = 0.5, 4, 40
    d, rec = comb(a, D)
    d.run(n=N, chunk=chunk)
    n = np.arange(N)
    h = np.where(n % D == 0, a ** (n // D), 0.0)
    np.testing.assert_allclose(rec.data, h, rtol=1e-12)
