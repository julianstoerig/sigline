import numpy as np
import pytest

from _blocks import Accumulate, Add, Delay, Ramp, Record, Split
from sigline import Block, Const, FromArray, Signal, Step
from sigline.testing import assert_chunk_invariant

X = np.random.default_rng(5).standard_normal(500)


class Leaky(Block):
    """forgets its state between chunks: must be caught."""

    def step(self, x):
        return np.cumsum(x)


class Peek(Block):
    """looks one sample ahead within the chunk: must be caught."""

    def step(self, x):
        return np.r_[x[1:], 0.0]


# unit


def test_returns_single_chunk_reference():
    ref = assert_chunk_invariant(Accumulate, X)
    np.testing.assert_allclose(ref["y"], np.cumsum(X), atol=1e-10)


@pytest.mark.parametrize("bad", [Leaky, Peek])
def test_catches_chunk_dependent_blocks(bad):
    with pytest.raises(AssertionError, match="differs from the one-chunk run"):
        assert_chunk_invariant(bad, X)


def test_rejects_sinks_and_too_short_inputs():
    with pytest.raises(ValueError, match="no outputs"):
        assert_chunk_invariant(Record, X)
    with pytest.raises(ValueError, match="shorter than one granule"):
        assert_chunk_invariant(Accumulate, X[:0])


# integration: multi-input and multi-output blocks


def test_multi_input_and_multi_output():
    assert_chunk_invariant(Add, {"a": X, "b": X[::-1]})
    ref = assert_chunk_invariant(Split, X + 1j * X[::-1])
    np.testing.assert_array_equal(ref["im"], X[::-1])


# every block so far


@pytest.mark.parametrize(
    ("make", "x"),
    [
        (lambda: Const(2.0), 200),
        (lambda: Step(0.3), 200),
        (lambda: Signal(lambda t: np.sin(t / 7)), 200),
        (lambda: FromArray(X, loop=True), 1200),
        (Ramp, 200),
        (lambda: Delay(13), X),
    ],
    ids=["Const", "Step", "Signal", "FromArray", "Source subclass", "StateBlock"],
)
def test_blocks_are_chunk_invariant(make, x):
    assert_chunk_invariant(make, x, fs=100.0)


# comparison: chunked block equals the one-shot plain function call


def test_reference_equals_standalone_call():
    ref = assert_chunk_invariant(lambda: Delay(5), X)
    np.testing.assert_array_equal(ref["y"], Delay(5)(X))
