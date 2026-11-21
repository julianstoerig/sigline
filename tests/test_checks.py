import numpy as np
import pytest

from _blocks import Accumulate, Add, Ramp, Record, Scale
from sigline import Block, Diagram


class NanAt(Block):
    """passes x through but writes NaN at absolute sample k."""

    def __init__(self, k, **kw):
        self.k = k
        super().__init__(**kw)

    def reset(self):
        self.n = 0

    def step(self, x):
        y = x.astype(float).copy()
        i = self.k - self.n
        if 0 <= i < len(y):
            y[i] = np.nan
        self.n += len(x)
        return y


class Truncate(Block):
    """writes into a real buffer, which drops imaginary parts."""

    def step(self, x):
        y = np.zeros(len(x))
        y[:] = x
        return y


class ToComplex(Block):
    def step(self, x):
        return x * (1 + 1j)


def nan_chain(k, check="first"):
    src, bad, rec = Ramp(), NanAt(k), Record()
    src >> bad >> Scale(2.0) >> rec
    return Diagram(rec, fs=1.0, check=check), bad


# unit tests


def test_nan_names_the_first_failing_block_and_sample():
    d, bad = nan_chain(3)
    with pytest.raises(FloatingPointError, match=rf"{bad.name}\['y'\].*sample 3"):
        d.run(n=8, chunk=8)


def test_dropped_imaginary_part_is_reported():
    src, cplx, trunc, rec = Ramp(), ToComplex(), Truncate(), Record()
    src >> cplx >> trunc >> rec
    with pytest.raises(TypeError, match=rf"{trunc.name}.*dropped the imaginary"):
        Diagram(rec, fs=1.0).run(n=4)


def test_opt_in_width_and_real_input_checks():
    class SameWidth(Add):
        same_shape_inputs = True

    class Wide(Block):
        def step(self, x):
            return np.stack([x, x], axis=1)

    src, add = Ramp(), SameWidth()
    src >> add["a"]
    src >> Wide() >> add["b"]
    with pytest.raises(ValueError, match="input widths differ"):
        Diagram(add, fs=1.0).run(n=4)

    class RealOnly(Scale):
        real_inputs = True

    src, real = Ramp(), RealOnly()
    src >> ToComplex() >> real
    with pytest.raises(TypeError, match="expects real signals"):
        Diagram(real, fs=1.0).run(n=4)


def test_block_exceptions_get_a_note_naming_the_block():
    class Fails(Block):
        def step(self, x):
            raise ZeroDivisionError("boom")

    src, f = Ramp(), Fails()
    src >> f
    with pytest.raises(ZeroDivisionError) as info:
        Diagram(f, fs=1.0, check=False).run(n=4)
    assert any(f"in block {f.name} (Fails)" in note for note in info.value.__notes__)
    assert len(info.value.__notes__) == 1


def test_invalid_check_value_rejected():
    with pytest.raises(ValueError, match="check must be"):
        Diagram(Ramp(), fs=1.0, check="always")


# integration: when the checks run


def test_first_chunk_only_by_default_every_chunk_with_true():
    d, _ = nan_chain(12)  # NaN in the second chunk of 8
    d.run(n=16, chunk=8)  # passes unnoticed
    d, _ = nan_chain(12, check=True)
    with pytest.raises(FloatingPointError):
        d.run(n=16, chunk=8)


def test_reset_rearms_the_first_chunk_check_and_false_disables():
    d, bad = nan_chain(3)
    with pytest.raises(FloatingPointError):
        d.run(n=8)
    d.reset()
    with pytest.raises(FloatingPointError):
        d.run(n=8)
    d, _ = nan_chain(3, check=False)
    d.run(n=8)


# comparison: checks never change results


@pytest.mark.parametrize("check", ["first", True, False])
def test_results_independent_of_check_mode(check):
    src, rec = Ramp(), Record()
    src >> ToComplex() >> Accumulate() >> rec
    Diagram(rec, fs=1.0, check=check).run(n=50, chunk=7)
    n = np.arange(50)
    np.testing.assert_array_equal(rec.data, (1 + 1j) * n * (n + 1) / 2)
