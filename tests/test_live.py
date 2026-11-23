import time

import matplotlib
import numpy as np
import pytest

import sigline.live as live
from sigline import Block, Diagram, Noise, Scope, Sine, watch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def diagram():
    s, sc = Sine(5.0), Scope("tone", keep=500)
    s >> sc
    return Diagram(sc, fs=1000.0), sc


# unit


def test_mode_detection():
    assert live._mode() == "headless"  # Agg, no kernel


def test_draws_frames_until_the_run_ends():
    d, sc = diagram()
    frames = []

    def draw(fig):
        t, y = sc.snapshot()
        frames.append(len(y))
        fig.subplots().plot(t, y)

    d.start(0.4, chunk=50)
    fig = watch(d, draw, interval=0.05)
    # at least one frame while running and a last one after the end, which
    # sees the whole recording; the count in between depends on machine speed
    assert not d.running and len(frames) >= 2 and frames[-1] == 400
    assert frames == sorted(frames)  # the recording only grows (up to keep)
    assert len(fig.axes) == 1


def test_default_draw_plots_every_scope():
    d, _ = diagram()
    d.start(0.1, chunk=50)
    fig = watch(d, interval=0.02)
    assert len(fig.axes) == 1 and fig.axes[0].get_xlabel() == "t / s"
    src = Noise()
    src >> Block()  # no scope anywhere
    with pytest.raises(ValueError, match="no Scope"):
        watch(Diagram(src, fs=1.0))


def test_duration_stops_watching_not_the_run():
    d, _ = diagram()
    d.start(chunk=50)
    try:
        t0 = time.monotonic()
        watch(d, interval=0.02, duration=0.15)
        assert 0.14 <= time.monotonic() - t0 < 1.0 and d.running
    finally:
        d.stop()


def test_draw_errors_stop_the_diagram():
    d, _ = diagram()
    d.start(chunk=50)

    def bad(fig):
        raise RuntimeError("draw failed")

    with pytest.raises(RuntimeError, match="draw failed"):
        watch(d, bad, interval=0.01)
    assert not d.running


def test_run_errors_surface_through_watch():
    class Fails(Block):
        def step(self, x):
            if x[0] > 0.5:
                raise ValueError("late failure")
            return x

    s, f, sc = Sine(1.0), Fails(), Scope()
    s >> f >> sc
    d = Diagram(sc, fs=100.0)
    d.start(chunk=10)
    with pytest.raises(ValueError, match="late failure"):
        watch(d, interval=0.01)


def test_jupyter_mode_updates_one_display(monkeypatch):
    updates = []

    class Handle:
        def update(self, fig):
            updates.append(fig)

    shown = []

    def display(fig, display_id):
        shown.append(fig)
        return Handle()

    ipd = pytest.importorskip("IPython.display")
    monkeypatch.setattr(live, "_mode", lambda: "jupyter")
    monkeypatch.setattr(ipd, "display", display)
    d, sc = diagram()
    d.start(0.2, chunk=50)
    assert watch(d, interval=0.03) is None  # the cell shows it already
    assert len(shown) == 1 and len(updates) >= 2
    # comparison: the last frame shows the final recording
    np.testing.assert_array_equal(updates[-1].axes[0].lines[0].get_ydata(), sc.data)


def test_drawing_time_counts_towards_the_interval(monkeypatch):
    # theory: frames start every max(interval, draw time), not every
    # interval + draw time. a fake clock makes this exact; real sleeps
    # overshoot by up to 100 ms on slow ci runners

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, s):
            self.now += s

    clock = Clock()
    monkeypatch.setattr(live, "time", clock)

    class Run:  # stands in for a diagram running for 1 s of fake time
        running = property(lambda self: clock.now < 1.0)

        def stop(self):
            pass

        def wait(self):
            pass

    starts = []

    def draw(fig, cost):
        starts.append(clock.now)
        clock.now += cost  # drawing takes this long

    for cost, gap in ((0.05, 0.1), (0.25, 0.25)):
        starts.clear()
        watch(Run(), lambda fig, c=cost: draw(fig, c), interval=0.1)
        np.testing.assert_allclose(np.diff(starts), gap)
        clock.now = 0.0


def test_kernel_detection_accepts_any_kernel_shell(monkeypatch):
    ip = pytest.importorskip("IPython")

    class Shell:
        kernel = object()
        config = {}

    monkeypatch.setattr(ip, "get_ipython", lambda: Shell())
    monkeypatch.setattr(
        matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline"
    )
    assert live._mode() == "jupyter"


# one call: start paced, then watch


def test_diagram_watch_starts_and_shows_the_run():
    d, sc = diagram()
    t0 = time.perf_counter()
    fig = d.watch(0.5, speed=5.0, interval=0.02)
    assert not d.running and d.n == 500 and len(sc.data) == 500
    assert 0.09 < time.perf_counter() - t0 < 1.0 and len(fig.axes) == 1


def test_diagram_watch_only_watches_a_running_diagram():
    d, sc = diagram()
    d.start(0.3, chunk=50)
    d.watch(5.0, interval=0.02)  # T ignored: already running
    assert d.n == 300


# window=True: a tk window from any backend; a stand-in here, no display


class TclError(Exception):
    pass


class FakeTk:
    def __init__(self, close_after=None):
        self.updates, self.destroyed, self.close_after = 0, False, close_after
        self.window = self

    def update(self):
        self.updates += 1
        if self.close_after is not None and self.updates > self.close_after:
            raise TclError("can't invoke 'update': application has been destroyed")

    def destroy(self):
        self.destroyed = True


def test_window_draws_into_its_own_tk_window(monkeypatch):
    tk = FakeTk()
    monkeypatch.setattr(live, "_tk_window", lambda fig: tk)
    d, sc = diagram()
    fig = d.watch(0.3, speed=3.0, interval=0.02, window=True)
    assert not d.running and d.n == 300
    assert tk.updates > 2 and tk.destroyed and len(fig.axes) == 1


def test_closing_the_window_stops_the_run(monkeypatch):
    tk = FakeTk(close_after=3)
    monkeypatch.setattr(live, "_tk_window", lambda fig: tk)
    d, sc = diagram()
    d.watch(10.0, interval=0.02, window=True)  # returns at once, no error
    assert not d.running and d.n < 10_000


def test_no_window_stops_the_run_and_raises(monkeypatch):
    def fail(fig):
        raise RuntimeError("no display")

    monkeypatch.setattr(live, "_tk_window", fail)
    d, sc = diagram()
    with pytest.raises(RuntimeError, match="no display"):
        d.watch(10.0, window=True)
    assert not d.running
