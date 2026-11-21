import numpy as np
import pytest

from _blocks import Ramp, Scale
from sigline import Diagram, Scope


def recorded(chunk, n=100, **kw):
    src, sc = Ramp(), Scope(**kw)
    src >> sc
    Diagram(sc, fs=10.0).run(n=n, chunk=chunk)
    return sc


# unit


def test_records_and_copies_input():
    sc = Scope()
    x = np.arange(4.0)
    sc(x)
    x[:] = -1
    np.testing.assert_array_equal(sc.data, [0, 1, 2, 3])
    assert Scope().data.size == 0


def test_label_names_the_scope():
    assert Scope("rx").name == "rx" and Scope("rx").label == "rx"
    sc = Scope()
    assert sc.label == sc.name


def test_reset_forgets():
    sc = recorded(7)
    sc.reset()
    assert sc.data.size == 0


# integration


def test_records_a_chain_in_a_diagram():
    src, sc = Ramp(), Scope()
    src >> Scale(3.0) >> sc
    d = Diagram(sc, fs=4.0)
    d.run(n=10, chunk=3)
    np.testing.assert_array_equal(sc.data, 3 * np.arange(10))
    np.testing.assert_array_equal(sc.t, d.t)


# comparison: keep and every equal slicing the full record, for any chunking


@pytest.mark.parametrize("chunk", [1, 3, 16, 100])
@pytest.mark.parametrize(("keep", "every"), [(None, 3), (20, 1), (7, 4)])
def test_keep_and_every_equal_slicing(chunk, keep, every):
    full = recorded(chunk)
    sc = recorded(chunk, keep=keep, every=every)
    ref_t = full.t[::every]
    ref = full.data[::every]
    if keep is not None:
        ref, ref_t = ref[-keep:], ref_t[-keep:]
    np.testing.assert_array_equal(sc.data, ref)
    np.testing.assert_allclose(sc.t, ref_t)


# theory: absolute time of sample k is k * every / fs


def test_time_axis_is_absolute():
    sc = recorded(9, n=1000, keep=50, every=5)
    k = np.arange(1000)[::5][-50:]
    np.testing.assert_allclose(sc.t, k / 10.0)
    np.testing.assert_array_equal(sc.data, k)
