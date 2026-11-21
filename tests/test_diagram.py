import numpy as np
import pytest

from _blocks import Accumulate, Add, Ramp, Record, Scale
from sigline import AlgebraicLoopError, Block, Diagram


def chain():
    ramp, rec = Ramp(), Record()
    ramp >> Scale(2.0) >> Accumulate() >> rec
    return ramp, rec


# unit tests


def test_discovers_every_connected_block_from_any_member():
    ramp, rec = chain()
    a = Diagram(rec, fs=1.0).blocks
    b = Diagram(ramp, fs=1.0).blocks
    assert a == b and len(a) == 4
    assert [blk._id for blk in a] == sorted(blk._id for blk in a)


def test_unconnected_input_rejected():
    add = Add()
    Ramp() >> add["a"]
    with pytest.raises(ValueError, match=r"unconnected input\(s\) \['b'\]"):
        Diagram(add, fs=1.0)


def test_algebraic_loop_detected():
    a, b = Scale(), Scale()
    a >> b >> a
    d = Diagram(a, fs=1.0)
    with pytest.raises(AlgebraicLoopError, match="algebraic loop"):
        d.run(n=4)


def test_fs_set_on_blocks_and_mismatch_rejected():
    ramp, rec = chain()
    Diagram(rec, fs=48e3)
    assert all(b.fs == 48e3 for b in Diagram(rec, fs=48e3).blocks)
    with pytest.raises(ValueError, match="differs from the diagram rate"):
        Diagram(rec, fs=8e3)


def test_lookup_time_axis_and_argument_checks():
    ramp, rec = chain()
    d = Diagram(rec, fs=4.0)
    assert d[rec.name] is rec
    with pytest.raises(KeyError):
        d["nope"]
    with pytest.raises(ValueError, match="give T"):
        d.run()
    d.run(1.5)
    assert d.n == 6
    np.testing.assert_array_equal(d.t, np.arange(6) / 4.0)


# integration: fan-out, fan-in and a diamond that needs the topological order


def test_fanout_fanin_diamond():
    ramp, add, rec = Ramp(), Add(), Record()
    ramp >> Scale(2.0) >> add["a"]
    ramp >> Accumulate() >> add["b"]
    add >> Scale(-1.0) >> rec
    d = Diagram(rec, fs=1.0)
    order = d.compile(8)
    assert order.index(add) > max(order.index(p.block) for p in add._sources.values())
    d.run(n=20, chunk=8)
    n = np.arange(20)
    np.testing.assert_array_equal(rec.data, -(2 * n + n * (n + 1) / 2))


# comparison: chunk sizes, continued runs, reset, standalone calls


@pytest.mark.parametrize("chunk", [1, 3, 64, 1000])
def test_any_chunking_equals_standalone_one_shot(chunk):
    ramp, rec = chain()
    Diagram(rec, fs=1.0).run(n=1000, chunk=chunk)
    ref = Accumulate()(Scale(2.0)(Ramp()(1000)))
    np.testing.assert_array_equal(rec.data, ref)
    expected_chunks = -(-1000 // chunk)
    assert len(rec.chunks) == expected_chunks


def test_continue_and_reset_reproduce_one_run():
    ramp, rec = chain()
    d = Diagram(rec, fs=1.0)
    d.run(n=100, chunk=7).run(n=50, chunk=33)
    assert d.n == 150
    first = rec.data.copy()
    d.reset()
    assert d.n == 0 and rec.data.size == 0
    d.run(n=150, chunk=150)
    np.testing.assert_array_equal(rec.data, first)


def test_parameter_change_between_runs_takes_effect():
    ramp, scale, rec = Ramp(), Scale(1.0), Record()
    ramp >> scale >> rec
    d = Diagram(rec, fs=1.0)
    d.run(n=5)
    scale.k = 10.0
    d.run(n=5)
    np.testing.assert_array_equal(rec.data, np.r_[np.arange(5), 10 * np.arange(5, 10)])


# theory: a sampled sine through the diagram matches the closed form


def test_sine_source_matches_closed_form():
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
            return np.sin(2 * np.pi * self.f * n / self.fs)

    fs, f = 8000.0, 440.0
    src, rec = Sine(f), Record()
    src >> Scale(0.5) >> rec
    Diagram(rec, fs=fs).run(0.25, chunk=300)
    t = np.arange(2000) / fs
    np.testing.assert_allclose(rec.data, 0.5 * np.sin(2 * np.pi * f * t), atol=1e-12)
