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


def _tk_window(fig):
    # a tk window for fig, whatever the backend (inline in a notebook too)
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureManagerTk

    manager = FigureManagerTk.create_with_canvas(FigureCanvasTkAgg, fig, 1)
    manager.set_window_title("sigline")
    manager.show()
    return manager


def watch(
    d,
    draw=None,
    *,
    interval=None,
    duration=None,
    figsize=(8, 4),
    window=False,
    _begin=None,
):
    """Redraw a figure while ``d`` runs in the background.

    Start the diagram first (:meth:`Diagram.start`). Each frame clears the
    figure and calls ``draw(fig)``; read scopes there with
    :meth:`Scope.snapshot`. In Jupyter the output cell updates; with a GUI
    backend a window does. Interrupting (Ctrl+C, stop button) stops ``d``.

    For smooth animation, let ``draw(fig)`` build the plot once and return
    an ``update()`` function: each frame then calls only ``update()``, which
    changes the artists (``line.set_data(...)``) and returns those it
    changed. In a window only these are redrawn, over a saved background
    (blitting): tens of frames per second. Axes limits and the rest of the
    figure stay as ``draw`` left them.

    A notebook sends each frame as an image, a few per second; a window
    draws faster and smoother. ``window=True`` opens a Tk window from a
    notebook too, without changing the backend; at the end it closes and
    the last frame stays in the notebook. Closing it stops ``d``.

    Parameters
    ----------
    d : Diagram
    draw : callable, optional
        ``draw(fig)``, or ``draw(fig) -> update`` with ``update() ->
        artists``; default: every Scope over time, one axis each.
    interval : float, optional
        Seconds between frame starts (more if drawing takes longer);
        default 0.2, or 1/30 when ``draw`` returns ``update``.
    duration : float, optional
        Stop watching (not the diagram) after this many seconds.
    figsize : tuple, default (8, 4)
    window : bool, default False
        Draw into a Tk window (needs a display and Tk).

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
    tk = None
    if window and mode != "window":
        from matplotlib.figure import Figure

        fig = Figure(figsize=figsize)
        try:
            tk = _tk_window(fig)
        except BaseException:
            d.stop()  # no window, e.g. no display: do not leave the run going
            raise
    else:
        fig = plt.figure(figsize=figsize)
    live = tk is not None or mode == "window"  # a canvas on screen
    handle, update, artists, background = None, None, [], None
    t_end = None if duration is None else time.monotonic() + duration

    def forget_background(event=None):
        nonlocal background
        background = None

    fig.canvas.mpl_connect("resize_event", forget_background)

    def pump(seconds):
        # keep the window responsive while waiting
        t_stop = time.monotonic() + seconds
        while True:
            if tk is not None:
                tk.window.update()
            else:
                fig.canvas.flush_events()
            left = t_stop - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 0.01))

    def show():
        # the whole figure, as an image or in the window
        nonlocal handle
        if tk is not None:
            fig.canvas.draw()
            tk.window.update()
        elif mode == "jupyter":
            from IPython.display import display

            if handle is None:
                handle = display(fig, display_id=True)
            else:
                handle.update(fig)
        elif mode == "window":
            fig.canvas.draw_idle()
            plt.pause(0.001)

    def blit():
        # redraw only the artists update() returned, over the background
        nonlocal background
        canvas = fig.canvas
        if background is None:
            for a in artists:
                a.set_animated(True)
            canvas.draw()
            background = canvas.copy_from_bbox(fig.bbox)
        canvas.restore_region(background)
        for a in artists:
            fig.draw_artist(a)
        canvas.blit(fig.bbox)
        if tk is not None:
            tk.window.update()
        else:
            canvas.flush_events()

    def frame():
        nonlocal update, artists
        if update is None:
            fig.clear()
            result = draw(fig)
            if not callable(result):
                show()
                return
            update = result
        artists = list(update() or [])
        if live:
            blit()
        else:
            show()

    first = True
    try:
        if _begin is not None:
            _begin()  # Diagram.watch: start once the window stands
        while True:
            t_frame = time.monotonic()
            alive = d.running
            frame()
            if first:
                first = False
                if interval is None:
                    interval = 1 / 30 if update is not None else 0.2
            if not alive or (t_end is not None and time.monotonic() >= t_end):
                break
            wait = max(0.0, interval - (time.monotonic() - t_frame))
            if live:
                pump(wait)
            else:
                # drawing time counts towards the interval
                time.sleep(wait)
    except KeyboardInterrupt:
        d.stop()
    except BaseException as err:
        d.stop()
        if tk is None or not _closed(err):
            raise
    finally:
        for a in artists:
            a.set_animated(False)  # the last frame drawn whole
        if tk is not None:
            try:
                tk.destroy()
            except Exception:  # noqa: BLE001 - closed by the user already
                pass
            if mode == "jupyter":
                from IPython.display import display

                display(fig)  # the last frame stays in the notebook
        elif mode == "jupyter":
            if update is not None:
                show()  # with the artists drawn normally again
            plt.close(fig)  # the display keeps the last frame
        elif mode == "window" and update is not None:
            fig.canvas.draw_idle()
    if not d.running:
        d.wait()  # re-raise an error of the run
    return None if mode == "jupyter" else fig  # jupyter already shows it


def _closed(err):
    # tk raises TclError once its window is gone
    return type(err).__name__ == "TclError"
