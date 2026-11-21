import threading
import time

import numpy as np
import pytest

from sigline import Diagram, FromQueue, Gain, SampleBuffer, Scope, ToQueue

X = np.random.default_rng(4).standard_normal(4000)


def producer(q, x, chunk, period):
    def run():
        for i in range(0, len(x), chunk):
            q.push(x[i : i + chunk])
            time.sleep(period)

    th = threading.Thread(target=run)
    th.start()
    return th


# unit: the buffer


def test_buffer_fifo_underrun_and_overflow():
    b = SampleBuffer(capacity=5)
    b.push([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(b.pull(2), [1, 2])
    np.testing.assert_array_equal(b.pull(3), [3, 0, 0])
    assert b.underruns == 2
    b.push(np.arange(8.0))  # 3 too many: the oldest go
    assert b.overflows == 3 and len(b) == 5
    np.testing.assert_array_equal(b.pull(5), [3, 4, 5, 6, 7])


def test_buffer_keeps_complex_and_vector_samples():
    b = SampleBuffer()
    b.push(np.ones((3, 2)) * 1j)
    out = b.pull(4)
    assert out.shape == (4, 2) and out.dtype == complex and np.all(out[3] == 0)


def test_buffer_wait_timeout_and_wake():
    b = SampleBuffer()
    t0 = time.perf_counter()
    assert np.all(b.pull(3, timeout=0.1) == 0)
    assert time.perf_counter() - t0 >= 0.09
    threading.Timer(0.05, b.wake).start()
    t0 = time.perf_counter()
    b.pull(3, timeout=None)  # would wait forever without wake
    assert time.perf_counter() - t0 < 1.0
    b.rearm()
    threading.Timer(0.05, b.push, args=([1.0, 2.0, 3.0],)).start()
    np.testing.assert_array_equal(b.pull(3, timeout=None), [1, 2, 3])


# unit: the blocks


def test_fromqueue_and_toqueue_standalone():
    q = FromQueue(timeout=0)
    q.push([5.0, 6.0])
    np.testing.assert_array_equal(q(3), [5, 6, 0])
    assert q.underruns == 1 and q.external_clock is False
    assert FromQueue().external_clock is True
    t = ToQueue(capacity=2)
    t(np.arange(3.0))
    assert t.overflows == 1
    np.testing.assert_array_equal(t.pull(2), [1, 2])


@pytest.mark.parametrize("chunk", [1, 7, 64, 500])
def test_any_chunking_gives_the_pushed_data(chunk):
    # reset() drops queued samples by design, so assert_chunk_invariant's replay
    # after reset does not apply; compare chunk sizes directly instead
    q, sc = FromQueue(timeout=0), Scope()
    q.push(X[:500])
    q >> sc
    Diagram(sc, fs=1.0).run(n=500, chunk=chunk)
    np.testing.assert_array_equal(sc.data, X[:500])
    assert q.underruns == 0


# integration: threads on both ends of a running diagram


def test_producer_paces_a_background_run():
    src, g, sink = FromQueue(), Gain(2.0), ToQueue()
    src >> g >> sink
    d = Diagram(sink, fs=8000.0)
    th = producer(src, X, chunk=400, period=0.01)
    t0 = time.perf_counter()
    d.start(n=4000, chunk=256)  # realtime off: FromQueue sets the pace
    d.wait(timeout=10)
    th.join()
    assert time.perf_counter() - t0 >= 0.09  # >= 9 waits for the producer
    np.testing.assert_array_equal(sink.pull(4000), 2 * X)
    assert src.underruns == 0 and sink.underruns == 0


def test_stop_wakes_a_waiting_source():
    src, sc = FromQueue(), Scope()
    src >> sc
    d = Diagram(sc, fs=1000.0)
    d.start(chunk=100)
    time.sleep(0.1)
    t0 = time.perf_counter()
    d.stop()
    assert time.perf_counter() - t0 < 1.0 and not d.running
    # a new run can wait again
    th = producer(src, X[:300], chunk=100, period=0.01)
    d.start(n=300, chunk=100).wait(timeout=10)
    th.join()
    np.testing.assert_array_equal(sc.data[-300:], X[:300])


# comparison and theory: a loopback through two queues returns the data in
# order, whatever the producer's and the diagram's chunk sizes


@pytest.mark.parametrize(("push", "chunk"), [(1, 64), (333, 100), (4000, 7)])
def test_loopback_preserves_every_sample(push, chunk):
    src, sink = FromQueue(), ToQueue()
    src >> sink
    d = Diagram(sink, fs=1.0)
    th = producer(src, X, chunk=push, period=0)
    d.start(n=len(X) // chunk * chunk, chunk=chunk).wait(timeout=20)
    th.join()
    got = sink.pull(len(X) // chunk * chunk)
    np.testing.assert_array_equal(got, X[: len(got)])


# back-pressure: a sink that waits for its reader


def test_push_waits_for_space():
    b = SampleBuffer(capacity=4)
    b.push([1.0, 2.0, 3.0, 4.0])
    threading.Timer(0.05, b.pull, args=(2,)).start()
    t0 = time.perf_counter()
    b.push([5.0, 6.0], timeout=None)
    assert time.perf_counter() - t0 >= 0.04 and b.overflows == 0
    np.testing.assert_array_equal(b.pull(4), [3, 4, 5, 6])


def test_blocking_sink_is_paced_by_its_reader():
    # theory: a reader taking 100 samples every 10 ms drains 1000 samples in
    # about 100 ms, and with back-pressure nothing is dropped
    src, sink = FromQueue(timeout=0), ToQueue(capacity=200, block=True)
    src.push(X[:1000])
    src >> sink
    d = Diagram(sink, fs=1.0)
    assert sink.external_clock
    got = []

    def reader():
        while sum(map(len, got)) < 1000:
            got.append(sink.pull(100, timeout=1.0))
            time.sleep(0.01)

    th = threading.Thread(target=reader)
    t0 = time.perf_counter()
    th.start()
    d.start(n=1000, chunk=50)
    d.wait(timeout=10)
    th.join()
    assert time.perf_counter() - t0 >= 0.07
    np.testing.assert_array_equal(np.concatenate(got), X[:1000])
    assert sink.overflows == 0


def test_stop_wakes_a_waiting_sink():
    src, sink = FromQueue(timeout=0), ToQueue(capacity=10, block=True)
    src >> sink
    d = Diagram(sink, fs=1.0)
    d.start(chunk=10)
    time.sleep(0.1)  # nobody reads: the run waits for space
    t0 = time.perf_counter()
    d.stop()
    assert time.perf_counter() - t0 < 1.0
