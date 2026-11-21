import numpy as np
import pytest

from _blocks import Accumulate, Add, Ramp, Record, Scale, Split
from sigline import Block, Diagram, Port

# unit tests: ports and lines


def test_default_names_count_per_class():
    a, b = Scale(), Scale()
    assert a.name != b.name
    assert a.name.startswith("Scale") and b.name.startswith("Scale")
    assert Scale(name="amp").name == "amp"


def test_getitem_inp_out_give_ports():
    s = Split()
    assert isinstance(s["x"], Port)
    assert (s["x"].block, s["x"].name, s["x"].kind) == (s, "x", "in")
    assert s["re"].kind == "out" and s.out("im").name == "im"
    assert s.inp("x").block is s
    with pytest.raises(KeyError, match="no port"):
        s["nope"]
    with pytest.raises(KeyError, match="no input"):
        s.inp("re")
    with pytest.raises(KeyError, match="no output"):
        s.out("x")


def test_port_both_input_and_output_needs_explicit_access():
    class Loop(Block):
        inputs = ("y",)

    b = Loop()
    with pytest.raises(KeyError, match="both an input and an output"):
        b["y"]
    assert b.inp("y").kind == "in" and b.out("y").kind == "out"


def test_rshift_records_connection_and_returns_destination():
    a, b, c = Scale(), Scale(), Add()
    assert (a >> b) is b
    assert (b._sources["x"].block, b._sources["x"].name) == (a, "y")
    assert [(p.block, p.name) for p in a._sinks] == [(b, "x")]
    assert (b >> c["a"]) is c
    assert (c._sources["a"].block, c._sources["a"].name) == (b, "y")
    # fan-out: one output drives several inputs
    a >> c["b"]
    assert len(a._sinks) == 2


def test_line_errors():
    a, b, add, split = Scale(), Scale(), Add(), Split()
    a >> b
    with pytest.raises(ValueError, match="already driven"):
        Scale() >> b
    with pytest.raises(ValueError, match="pick one"):
        a >> add
    with pytest.raises(ValueError, match="pick one"):
        split >> a
    with pytest.raises(ValueError, match="input port"):
        add["a"] >> b
    with pytest.raises(ValueError, match="output port"):
        a >> split["re"]
    with pytest.raises(ValueError, match="no inputs"):
        a >> Ramp()
    with pytest.raises(TypeError, match="cannot connect"):
        a >> 3


# standalone use


def test_block_as_plain_function():
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_array_equal(Scale(3)(x), 3 * x)
    np.testing.assert_array_equal(Add()(x, b=x), 2 * x)
    re, im = Split()(np.array([1 + 2j, 3 - 4j]))
    np.testing.assert_array_equal(re, [1, 3])
    np.testing.assert_array_equal(im, [2, -4])


def test_scalar_output_is_broadcast_and_wrong_length_rejected():
    class Five(Block):
        def step(self, x):
            return 5.0

    class Short(Block):
        def step(self, x):
            return x[:-1]

    np.testing.assert_array_equal(Five()(np.zeros(4)), np.full(4, 5.0))
    with pytest.raises(ValueError, match="returned 2 samples, expected 3"):
        Short()(np.zeros(3))


def test_missing_step_raises():
    with pytest.raises(NotImplementedError, match="step"):
        Block()(np.zeros(2))


# comparison: chunked calls equal one call (state carries over)


@pytest.mark.parametrize("chunk", [1, 3, 10, 100])
def test_stateful_block_chunked_equals_one_shot(chunk):
    x = np.random.default_rng(0).standard_normal(100)
    acc = Accumulate()
    y = np.concatenate([acc(x[i : i + chunk]) for i in range(0, len(x), chunk)])
    np.testing.assert_allclose(y, Accumulate()(x), rtol=1e-12, atol=1e-12)


def test_reset_restores_initial_state():
    acc = Accumulate()
    x = np.ones(5)
    first = acc(x)
    acc(x)
    acc.reset()
    np.testing.assert_array_equal(acc(x), first)


# theory: closed forms


def test_source_and_accumulator_match_closed_form():
    ramp, acc = Ramp(), Accumulate()
    n = np.arange(50)
    y = np.concatenate([acc(ramp(10)) for _ in range(5)])
    # sum_{k=0}^{n} k = n (n + 1) / 2
    np.testing.assert_array_equal(y, n * (n + 1) / 2)


# << draws lines right to left


def test_lshift_mirrors_rshift_and_returns_the_source():
    a, b, c = Ramp(), Scale(), Scale()
    assert (c << b << a) is a
    assert (b._sources["x"].block, c._sources["x"].block) == (a, b)
    add = Add()
    assert (add["b"] << a) is a and add._sources["b"].block is a
    with pytest.raises(ValueError, match="already driven"):
        b << Ramp()
    with pytest.raises(ValueError, match="output port"):
        a["y"] << b


def test_lshift_chain_gives_the_same_result_as_rshift():
    def run(build):
        src, rec = Ramp(), Record()
        build(src, Scale(2.0), Accumulate(), rec)
        Diagram(rec, fs=1.0).run(n=50, chunk=7)
        return rec.data

    fwd = run(lambda s, g, acc, r: s >> g >> acc >> r)
    back = run(lambda s, g, acc, r: r << acc << g << s)
    np.testing.assert_array_equal(fwd, back)
