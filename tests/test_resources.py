import numpy as np
import pytest

from sigline import Diagram, Scope, Source


class Device(Source):
    """fake device: counts opens and closes, refuses to run while closed."""

    def __init__(self, **kw):
        self.opened = self.closed = 0
        self.is_open = False
        super().__init__(**kw)

    def open(self):
        self.opened += 1
        self.is_open = True

    def close(self):
        self.closed += 1
        self.is_open = False

    def generate(self, n):
        if not self.is_open:
            raise RuntimeError("device not open")
        return n.astype(float)


def build():
    dev, sc = Device(), Scope()
    dev >> sc
    return Diagram(sc, fs=1000.0), dev, sc


def test_block_hooks_default_to_no_ops():
    s = Scope()
    s.open()
    s.close()


def test_run_opens_once_and_keeps_open_between_runs():
    d, dev, sc = build()
    d.run(n=10)
    d.run(n=10)
    assert (dev.opened, dev.closed, dev.is_open) == (1, 0, True)
    np.testing.assert_array_equal(sc.data, np.arange(20))
    d.close()
    d.close()
    assert (dev.opened, dev.closed) == (1, 1)


def test_background_run_closes_at_its_end():
    d, dev, _ = build()
    d.start(n=100, realtime=False).wait(timeout=10)
    assert (dev.opened, dev.closed, dev.is_open) == (1, 1, False)
    d.start(n=100, realtime=False).wait(timeout=10)
    assert (dev.opened, dev.closed) == (2, 2)


def test_with_block_stops_and_closes():
    d, dev, _ = build()
    with d:
        d.start(chunk=10)
    assert not d.running and not dev.is_open and dev.closed == 1


def test_close_also_after_errors():
    d, dev, _ = build()

    def boom():
        raise ValueError("boom")

    d.start(chunk=10)
    d.call(boom)
    with pytest.raises(ValueError, match="boom"):
        d.wait(timeout=10)
    assert not dev.is_open
