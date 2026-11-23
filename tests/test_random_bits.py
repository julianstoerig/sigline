import numpy as np

import sigline as sl
from sigline import Diagram, RandomBits, Scope
from sigline.testing import assert_chunk_invariant


def record(src, n, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=1.0).run(n=n, chunk=chunk)
    return sc.data


# unit


def test_bits_reproduce_and_info():
    a = RandomBits(seed=3, fs=1.0)(50)
    assert set(np.unique(a)) <= {0, 1} and a.dtype.kind == "i"
    np.testing.assert_array_equal(RandomBits(seed=3, fs=1.0)(50), a)
    assert RandomBits(seed=3).info() == "seed 3"


def test_chunk_invariant():
    assert_chunk_invariant(lambda: RandomBits(seed=2), 600)


# integration: bits through a mapper and a demapper come back


def test_bits_through_mapper_and_demapper():
    bits, back = RandomBits(seed=5), Scope()
    bits >> sl.mapper(sl.qam(16)) >> sl.demapper(sl.qam(16)) >> back
    Diagram(back, fs=4.0).run(n=4000, chunk=400)
    np.testing.assert_array_equal(back.data, record(RandomBits(seed=5), 4000))


# comparison and theory: numpy's integers; half are ones


def test_equals_numpy_and_is_fair():
    n = 200_000
    x = record(RandomBits(seed=7), n, chunk=777)
    np.testing.assert_array_equal(x, np.random.default_rng(7).integers(0, 2, n))
    assert abs(x.mean() - 0.5) < 5 * 0.5 / np.sqrt(n)
