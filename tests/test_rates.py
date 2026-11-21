from fractions import Fraction

import numpy as np
import pytest

from _blocks import Add, Ramp, Record, Scale
from sigline import Block, Diagram, Scope


class Up(Block):
    """zero-stuffing upsampler by an integer factor."""

    def __init__(self, k, **kw):
        self.k = k
        self.resampling_ratio = Fraction(k)
        super().__init__(**kw)

    def step(self, x):
        y = np.zeros(len(x) * self.k)
        y[:: self.k] = x
        return y


class Down(Block):
    """keeps every m-th sample; granularity keeps chunks multiples of m."""

    def __init__(self, m, **kw):
        self.m = m
        self.resampling_ratio = Fraction(1, m)
        super().__init__(**kw)

    def step(self, x):
        return x[:: self.m]


# unit tests


def test_rates_propagate_along_the_lines():
    src, up, down, rec = Ramp(), Up(4), Down(2), Record()
    src >> up >> down >> rec
    d = Diagram(rec, fs=1000.0)
    assert d.rates() == {
        src.name: (1000.0, 1000.0),
        up.name: (1000.0, 4000.0),
        down.name: (4000.0, 2000.0),
        rec.name: (2000.0, 2000.0),
    }
    assert (up.fs, down.fs, rec.fs) == (1000.0, 4000.0, 2000.0)
    assert d.granularity == 1


def test_source_with_own_rate_and_granularity():
    slow, rec = Ramp(fs=250.0), Record()
    slow >> rec
    d = Diagram(rec, fs=1000.0)
    assert d.granularity == 4
    assert d.rates()[rec.name] == (250.0, 250.0)
    with pytest.raises(ValueError, match="multiple of 4"):
        d.run(n=6)
    d.run(n=40, chunk=10)  # chunk rounds down to 8
    np.testing.assert_array_equal(rec.data, np.arange(10))
    assert [len(c) for c in rec.chunks] == [2, 2, 2, 2, 2]


def test_rate_mismatch_at_a_block_is_rejected():
    add = Add()
    Ramp() >> add["a"]
    Ramp() >> Up(2) >> add["b"]
    with pytest.raises(ValueError, match="sample-rate mismatch"):
        Diagram(add, fs=1.0)


def test_resampling_ratio_must_be_a_fraction():
    class Bad(Block):
        resampling_ratio = 0.5

        def step(self, x):
            return x

    b = Bad()
    Ramp() >> b
    with pytest.raises(TypeError, match="must be a Fraction or int"):
        Diagram(b, fs=1.0)


def test_user_parameter_named_rate_is_harmless():
    class Decay(Block):
        def __init__(self, rate, **kw):
            self.rate = rate  # a decay rate, nothing to do with sample rates
            super().__init__(**kw)

        def step(self, x):
            return x * np.exp(-self.rate)

    src, b, sc = Ramp(), Decay(0.5), Scope()
    src >> b >> sc
    d = Diagram(sc, fs=1.0)
    d.run(n=4)
    assert d.rates()[b.name] == (1.0, 1.0)
    np.testing.assert_allclose(sc.data, np.arange(4) * np.exp(-0.5))


def test_standalone_call_uses_resampling_ratio():
    np.testing.assert_array_equal(Up(3)(np.array([1.0, 2.0])), [1, 0, 0, 2, 0, 0])
    np.testing.assert_array_equal(Down(2)(np.arange(6.0)), [0, 2, 4])


# integration and comparison: up then down by the same factor is identity


@pytest.mark.parametrize("chunk", [1, 5, 64])
def test_up_down_roundtrip_any_chunking(chunk):
    src, rec = Ramp(), Record()
    src >> Up(3) >> Scale(2.0) >> Down(3) >> rec
    Diagram(rec, fs=1.0).run(n=60, chunk=chunk)
    np.testing.assert_array_equal(rec.data, 2 * np.arange(60))


# theory: decimating a sampled sine gives the sine sampled at the lower rate


def test_decimated_sine_matches_lower_rate_closed_form():
    class Sine(Block):
        inputs = ()

        def __init__(self, f, **kw):
            self.f = f
            super().__init__(**kw)

        def reset(self):
            self.n = 0

        def step_source(self, L):
            n = np.arange(self.n, self.n + L)
            self.n += L
            return np.cos(2 * np.pi * self.f * n / self.fs)

    fs, f, m = 48_000.0, 1234.0, 6
    src, rec = Sine(f), Record()
    src >> Down(m) >> rec
    d = Diagram(rec, fs=fs)
    assert d.granularity == m
    d.run(0.1, chunk=1000)  # chunk rounds down to 996
    k = np.arange(len(rec.data))
    assert len(k) == 0.1 * fs / m
    np.testing.assert_allclose(
        rec.data, np.cos(2 * np.pi * f * k / (fs / m)), atol=1e-9
    )
