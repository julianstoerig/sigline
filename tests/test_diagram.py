import numpy as np
import pytest

from _blocks import Accumulate, Add, Delay, Ramp, Record, Scale
from sigline import AlgebraicLoopError, Block, ContinuousLTI, Diagram, Scope, Sine


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


def test_fs_set_on_blocks_and_contradiction_rejected():
    ramp, rec = chain()
    assert all(b.fs == 48e3 for b in Diagram(rec, fs=48e3).blocks)
    src, scale = Ramp(), Scale(fs=8e3)
    src >> scale
    with pytest.raises(ValueError, match="its inputs run at 48000 Hz"):
        Diagram(scale, fs=48e3)


def test_a_block_runs_again_at_another_rate():
    ramp, rec = chain()
    Diagram(rec, fs=48e3).run(n=4)
    assert all(b.fs == 8e3 for b in Diagram(rec, fs=8e3).blocks)


def test_a_rate_given_later_pins_and_none_unpins():
    src, scale = Ramp(), Scale()
    src >> scale
    d = Diagram(scale, fs=48e3)
    scale.fs = 8e3
    with pytest.raises(ValueError, match="its inputs run at 48000 Hz"):
        d.run(n=4)
    scale.fs = None
    d.run(n=4)
    assert scale.fs == 48e3


def test_a_probe_sink_pinned_to_another_rate_is_rejected():
    ramp, rec = chain()
    d = Diagram(rec, fs=48e3)
    with pytest.raises(ValueError, match="its inputs run at 48000 Hz"):
        d.probe(ramp, sink=Scope(fs=8e3))
    assert d.probe(ramp, sink=Scope()).fs == 48e3


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
    _, order = d.schedule(8)
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


# summary


# a source without fs follows the diagram it runs in, also the second one


def test_a_source_takes_the_rate_of_the_next_diagram():
    sine, sc = Sine(1e3), Scope()
    sine >> sc
    Diagram(sc, fs=8e3).run(n=8)
    d = Diagram(sc, fs=16e3)
    d.reset()
    d.run(n=16)
    np.testing.assert_allclose(sc.data, np.cos(2 * np.pi * np.arange(16) / 16))


# comparison: a block run before at another rate equals a fresh one


def test_reused_continuous_block_equals_a_fresh_one():
    def low_pass():
        lp, sc = ContinuousLTI([1.0], [0.01, 1.0]), Scope()
        Sine(50.0) >> lp >> sc
        return sc

    reused, fresh = low_pass(), low_pass()
    Diagram(reused, fs=1e3).run(n=200, chunk=64)
    d = Diagram(reused, fs=4e3)
    d.reset()
    d.run(n=200, chunk=64)
    Diagram(fresh, fs=4e3).run(n=200, chunk=64)
    np.testing.assert_array_equal(reused.data, fresh.data)


def test_summary_lists_schedule_and_lines():
    src, add, dly, rec = (
        Ramp(name="src"),
        Add(name="add"),
        Delay(4, name="z"),
        Record(name="out"),
    )
    src >> add["a"]
    dly >> add["b"]
    add >> dly
    add >> rec
    text = Diagram(rec, fs=8000.0).summary(chunk=4)
    assert text.splitlines() == [
        "Diagram fs=8000 Hz, 4 blocks, chunk=4",
        "  state  z",
        "  step   src",
        "  step   add",
        "  step   out",
        "lines:",
        "  src['y'] -> add['a']",
        "  z['y'] -> add['b']",
        "  add['y'] -> z['x']",
        "  add['y'] -> out['x']",
    ]
    text = Diagram(rec, fs=8000.0).summary(chunk=8)
    assert text.splitlines()[1:4] == [
        "  step   src",
        "  loop   loop[add, z]",
        "  step   out",
    ]
