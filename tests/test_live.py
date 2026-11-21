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
    assert not d.running and len(frames) >= 4
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
    watch(d, interval=0.03)
    assert len(shown) == 1 and len(updates) >= 2
    # comparison: the last frame shows the final recording
    np.testing.assert_array_equal(updates[-1].axes[0].lines[0].get_ydata(), sc.data)
