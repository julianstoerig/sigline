import numpy as np
import pytest

from _blocks import Split
from sigline import (
    Const,
    Diagram,
    FromArray,
    Func,
    Gain,
    Product,
    Scope,
    Sine,
    Sum,
)

rng = np.random.default_rng(6)
X = rng.standard_normal(300)
Y = rng.uniform(1.0, 2.0, 300)  # no zeros: safe to divide by


def evaluate(make):
    """build make(a, b) from two array sources, run it, return the output."""
    a, b, sc = FromArray(X, name="a"), FromArray(Y, name="b"), Scope()
    make(a, b) >> sc
    Diagram(sc, fs=1.0).run(n=300, chunk=37)
    return sc.data


# comparison: every operator equals the same numpy expression


@pytest.mark.parametrize(
    ("make", "ref"),
    [
        (lambda a, b: a + b, X + Y),
        (lambda a, b: a - b, X - Y),
        (lambda a, b: a * b, X * Y),
        (lambda a, b: a / b, X / Y),
        (lambda a, b: 2.5 * a, 2.5 * X),
        (lambda a, b: a * 2.5, 2.5 * X),
        (lambda a, b: a / 4, X / 4),
        (lambda a, b: a + 1, X + 1),
        (lambda a, b: 1 + a, 1 + X),
        (lambda a, b: a - 1, X - 1),
        (lambda a, b: 1 - a, 1 - X),
        (lambda a, b: 3 / b, 3 / Y),
        (lambda a, b: -a, -X),
        (lambda a, b: +a, X),
        (lambda a, b: np.float64(2) * a, 2 * X),
        (lambda a, b: (a + b) * (a - b) / b, (X + Y) * (X - Y) / Y),
        (lambda a, b: a + b + a - b, 2 * X),
    ],
    ids=lambda v: None if isinstance(v, np.ndarray) else "",
)
def test_operator_equals_numpy(make, ref):
    np.testing.assert_allclose(evaluate(make), ref, rtol=1e-12, atol=1e-12)


# unit: which blocks the operators create


def test_operators_create_the_expected_blocks():
    a, b = Const(1.0), Const(2.0)
    s = a - b
    assert isinstance(s, Sum) and s.signs == "+-"
    assert (s._sources["x0"].block, s._sources["x1"].block) == (a, b)
    assert isinstance(a * b, Product) and isinstance(a / b, Product)
    g = a / 4
    assert isinstance(g, Gain) and g.g == 0.25 and g._sources["x"].block is a
    assert (+a) is a
    c = 1 - a
    assert isinstance(c._sources["x0"].block, Const) and c._sources["x1"].block is a


def test_ports_take_part_in_arithmetic():
    # theory: |z|^2 = re^2 + im^2
    z = np.exp(0.1j * np.arange(200)) * 3.0
    sp, sc = Split(), Scope()
    FromArray(z) >> sp
    (sp["re"] * sp["re"] + sp["im"] * sp["im"]) >> sc
    Diagram(sc, fs=1.0).run(n=200, chunk=13)
    np.testing.assert_allclose(sc.data, 9.0)


def test_precedence_with_lines():
    # + binds tighter than >>: a + b >> sc is (a + b) >> sc
    a, b, sc = Sine(1.0), Sine(1.0, phase=np.pi), Scope()
    total = a + b >> sc
    assert total is sc
    sc2 = Scope()
    sc2 << a * 0.5 - b
    d = Diagram(sc, sc2, fs=100.0)
    d.run(n=100)
    assert np.max(np.abs(sc.data)) < 1e-12
    ref = 1.5 * np.cos(2 * np.pi * np.arange(100) / 100)
    np.testing.assert_allclose(sc2.data, ref, atol=1e-12)


def test_per_channel_gain_from_an_array():
    v = np.ones((10, 2))
    sc = Scope()
    FromArray(v) * np.array([1.0, -3.0]) >> sc
    Diagram(sc, fs=1.0).run(n=10)
    np.testing.assert_array_equal(sc.data, v * [1, -3])


def test_operator_errors():
    a = Const(1.0)
    with pytest.raises(TypeError, match="scalar or a signal"):
        a + np.ones(3)
    with pytest.raises(ValueError, match="pick one"):
        Split() + a
    with pytest.raises(TypeError, match="cannot connect"):
        a + "text"
    with pytest.raises(ValueError, match="has outputs"):
        a * Scope()


def test_expression_shows_up_as_blocks_in_the_diagram():
    a, b = FromArray(X, name="a"), FromArray(Y, name="b")
    out = (a + b) * 2 >> Func(np.abs, name="mag")
    d = Diagram(out, fs=1.0)
    kinds = sorted(type(blk).__name__ for blk in d.blocks)
    assert kinds == ["FromArray", "FromArray", "Func", "Gain", "Sum"]


def test_operators_create_no_spare_blocks():
    from sigline import Block

    a = Const(1.0)
    before = Block._counter["__all__"]
    _ = a + 1.0  # a Const and a Sum, nothing else
    assert Block._counter["__all__"] - before == 2
    before = Block._counter["__all__"]
    _ = a / a  # one Product
    assert Block._counter["__all__"] - before == 1
