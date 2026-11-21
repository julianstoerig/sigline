import time

import numpy as np
import pytest

from _blocks import Ramp
from sigline import Block, Diagram, Noise, Scope


def chain(chunk_sleep=0.0):
    class Slow(Block):
        def step(self, x):
            time.sleep(chunk_sleep)
            return x

    src, sc = Noise(seed=1), Scope()
    src >> Slow() >> sc
    return Diagram(sc, fs=10_000.0), sc


# unit


def test_start_returns_at_once_and_wait_finishes():
    d, sc = chain()
    assert d.start(n=40_000, chunk=1000, realtime=False) is d
    assert d.wait(timeout=10)
    assert not d.running and d.n == 40_000 and len(sc.data) == 40_000


def test_stop_ends_an_endless_run_on_a_chunk_boundary():
    d, sc = chain()
    d.start(chunk=500)
    time.sleep(0.2)
    assert d.running
    d.stop()
    assert not d.running and d.n > 0 and d.n % 500 == 0
    assert len(sc.data) == d.n


def test_busy_diagram_refuses_run_and_reset():
    d, _ = chain()
    d.start(chunk=100)
    try:
        with pytest.raises(RuntimeError, match="running in the background"):
            d.run(n=100)
        with pytest.raises(RuntimeError, match="running in the background"):
            d.reset()
        with pytest.raises(RuntimeError, match="running in the background"):
            d.start()
    finally:
        d.stop()


def test_errors_reach_the_caller():
    class Fails(Block):
        def step(self, x):
            raise ZeroDivisionError("boom")

    src, f = Ramp(), Fails()
    src >> f
    d = Diagram(f, fs=1.0)
    d.start(n=10, realtime=False)
    with pytest.raises(ZeroDivisionError, match="boom"):
        d.wait(timeout=10)


def test_stop_requests_end_background_runs():
    class StopAt(Block):
        outputs = ()

        def step(self, x):
            if x[-1] >= 99:
                self.request_stop("enough")
            return {}

    src, s = Ramp(), StopAt()
    src >> s
    d = Diagram(s, fs=1000.0)
    d.start(chunk=10, realtime=False)
    d.wait(timeout=10)
    assert d.n == 100 and d.stopped.endswith("enough")


# comparison: a background run gives exactly the result of run()


def test_background_equals_foreground():
    d1, a = chain()
    d1.run(n=20_000, chunk=999)
    d2, b = chain()
    d2.start(n=20_000, chunk=999, realtime=False).wait(timeout=10)
    np.testing.assert_array_equal(a.data, b.data)


# theory: real-time pacing takes n / fs of wall time; overruns when too slow


def test_realtime_pacing_takes_n_over_fs():
    d, _ = chain()
    t0 = time.perf_counter()
    d.start(0.3, chunk=500).wait(timeout=10)
    elapsed = time.perf_counter() - t0
    # never faster than real time; the upper bound only guards against hangs
    assert 0.29 <= elapsed < 2.0


def test_too_slow_blocks_count_overruns():
    d, _ = chain(chunk_sleep=0.02)  # 20 ms of work per 10 ms chunk
    d.start(n=1000, chunk=100).wait(timeout=10)
    assert d.overruns == 10
