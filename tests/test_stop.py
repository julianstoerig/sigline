import numpy as np
import pytest

from _blocks import Ramp, Record
from sigline import Block, Diagram


class StopAbove(Block):
    """passes x through and asks to stop once x exceeds a threshold."""

    outputs = ("y",)

    def __init__(self, limit, **kw):
        self.limit = limit
        super().__init__(**kw)

    def step(self, x):
        if np.any(x > self.limit):
            self.request_stop(f"x above {self.limit}")
        return x


def build(limit):
    src, stop, rec = Ramp(), StopAbove(limit), Record()
    src >> stop >> rec
    return Diagram(rec, fs=1.0), stop, rec


# unit tests


def test_request_stop_ends_run_after_the_chunk_and_reports_reason():
    d, stop, rec = build(25)
    assert d.stopped is None
    d.run(n=100, chunk=10)
    assert d.n == 30 and len(rec.data) == 30
    assert d.stopped == f"{stop.name}: x above 25"


def test_next_run_continues_and_clears_stopped():
    d, stop, rec = build(1e9)
    stop.request_stop()
    d.run(n=20, chunk=5)
    assert d.n == 5 and d.stopped.endswith("stop requested")
    d.run(n=20, chunk=5)
    assert d.n == 25 and d.stopped is None
    np.testing.assert_array_equal(rec.data, np.arange(25))


def test_run_to_the_end_leaves_stopped_none():
    d, stop, rec = build(1e9)
    d.run(n=50, chunk=7)
    assert d.stopped is None and d.n == 50


# comparison and theory: the run stops at the end of the first chunk that
# contains a sample above the limit, i.e. after ceil((limit + 1) / chunk) chunks


@pytest.mark.parametrize(("limit", "chunk"), [(0, 1), (9, 10), (10, 10), (99, 7)])
def test_stop_position_matches_closed_form(limit, chunk):
    d, stop, rec = build(limit)
    d.run(n=10_000, chunk=chunk)
    first_above = limit + 1  # ramp value k sits at sample k
    expected = -(-(first_above + 1) // chunk) * chunk
    assert d.n == expected
    np.testing.assert_array_equal(rec.data, np.arange(expected))
