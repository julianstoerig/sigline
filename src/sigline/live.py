"""Live plots of a diagram running in the background."""

from __future__ import annotations

import time

__all__ = ["watch"]

_FILE_BACKENDS = {"agg", "pdf", "pgf", "ps", "svg", "cairo", "template"}


def _mode() -> str:
    """Return ``"jupyter"``, ``"window"`` or ``"headless"``."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except ImportError:
        shell = None
    # any kernel-backed shell: jupyter, vs code, ...
    in_kernel = shell is not None and (
        hasattr(shell, "kernel") or "IPKernelApp" in getattr(shell, "config", {})
    )
    if in_kernel and ("inline" in backend or backend in _FILE_BACKENDS):
        return "jupyter"
    if backend in _FILE_BACKENDS or "inline" in backend:
        return "headless"
    return "window"


def _default_draw(d):
    if not d.scopes():
        raise ValueError("the diagram has no Scope; pass draw=")

    def draw(fig):
        scopes = d.scopes()
        axes = fig.subplots(len(scopes), 1, sharex=True, squeeze=False)[:, 0]
        d.plot(axes=axes)

    return draw


def watch(d, draw=None, *, interval=0.2, duration=None, figsize=(8, 4)):
    """Redraw a figure while ``d`` runs in the background.

    Start the diagram first (:meth:`Diagram.start`). Each frame clears the
    figure and calls ``draw(fig)``; read scopes there with
    :meth:`Scope.snapshot`. In Jupyter the output cell updates; with a GUI
    backend a window does. Interrupting (Ctrl+C, stop button) stops ``d``.

    Parameters
    ----------
    d : Diagram
    draw : callable, optional
        ``draw(fig)``; default: every Scope over time, one axis each.
    interval : float, default 0.2
        Seconds between frame starts (more if drawing takes longer).
    duration : float, optional
        Stop watching (not the diagram) after this many seconds.
    figsize : tuple, default (8, 4)

    Returns
    -------
    matplotlib.figure.Figure or None
        With the last frame; ``None`` in Jupyter, where the output cell
        already shows it.

    Raises
    ------
    ValueError
        No ``draw`` and no Scope in ``d``.
    """
    import matplotlib.pyplot as plt

    draw = draw or _default_draw(d)
    mode = _mode()
    fig = plt.figure(figsize=figsize)
    handle = None
    t_end = None if duration is None else time.monotonic() + duration

    def frame():
        nonlocal handle
        fig.clear()
        draw(fig)
        if mode == "jupyter":
            from IPython.display import display

            if handle is None:
                handle = display(fig, display_id=True)
            else:
                handle.update(fig)
        elif mode == "window":
            fig.canvas.draw_idle()
            plt.pause(interval)

    try:
        while True:
            t_frame = time.monotonic()
            alive = d.running
            frame()
            if not alive or (t_end is not None and time.monotonic() >= t_end):
                break
            if mode != "window":
                # drawing time counts towards the interval
                time.sleep(max(0.0, interval - (time.monotonic() - t_frame)))
    except KeyboardInterrupt:
        d.stop()
    except BaseException:
        d.stop()
        raise
    finally:
        if mode == "jupyter":
            plt.close(fig)  # the display keeps the last frame
    if not d.running:
        d.wait()  # re-raise an error of the run
    return None if mode == "jupyter" else fig  # jupyter already shows it
