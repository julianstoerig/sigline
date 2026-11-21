import numpy as np
import pytest

from _blocks import Accumulate
from sigline import Const, Diagram, Scope, Source


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
