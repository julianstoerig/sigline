"""Continuous-time blocks, integrated between sample instants.

Inputs are held over each sample interval (zero-order hold, as a DAC does).
The state is integrated from ``t_k`` to ``t_k + 1/fs``, the output sampled
at ``t_k``. So the output at sample ``k`` depends on inputs up to ``k - 1``
only (unless the output map uses the input), and a loop through the block
needs no other delay: it runs as a loop cluster. Open loop, any chunking
gives identical results.
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy.integrate import solve_ivp

from .core import StateBlock

__all__ = ["ODE"]

_SOLVER_KEYS = {"rtol", "atol", "max_step", "first_step", "jac", "vectorized"}


def _n_positional(fn):
    # positional parameters of fn; many for *args
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 2
    if any(p.kind == p.VAR_POSITIONAL for p in ps):
        return 99
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in ps)


class ODE(StateBlock):
    """Continuous system ``dx/dt = f(t, x, *u)``, output ``y = g(t, x[, *u])``.

    Parameters
    ----------
    f : callable
        ``f(t, x, *u) -> dx/dt``, one ``u`` per input.
    x0 : array_like
        Initial state.
    inputs, outputs : tuple of str, default ("u",), ("y",)
        Port names; ``inputs=()`` makes an autonomous system, a source.
    g : callable, optional
        ``g(t, x)``, or ``g(t, x, *u)`` with direct feedthrough. Default: the
        state; with one output and several states a vector signal ``(L, n)``.
    method : str, default "RK45"
        A ``scipy.integrate.solve_ivp`` method (``"RK45"``, ``"LSODA"``,
        ``"BDF"``, ``"Radau"``, ...), or ``"RK4"``: classic Runge-Kutta with
        ``substeps`` fixed steps per sample, without error control but
        compiled inside loop clusters when ``f`` and ``g`` compile with
        numba.
    substeps : int, default 4
        RK4 steps per sample.
    **kw
        ``rtol`` (default 1e-8), ``atol`` (1e-10), ``max_step``,
        ``first_step``, ``jac``, ``vectorized`` go to ``solve_ivp``; the rest
        to :class:`Block`.

    Attributes
    ----------
    x : numpy.ndarray
        Current state.
    k : int
        Samples integrated so far.

    Raises
    ------
    RuntimeError
        ``solve_ivp`` fails.

    Examples
    --------
    A first-order low-pass, ``dx/dt = (u - x) / tau``, driven by a held 1:

    >>> import numpy as np
    >>> from sigline import ODE
    >>> lp = ODE(lambda t, x, u: (u - x) / 0.01, [0.0], fs=1000.0)
    >>> y = lp(np.ones(30))
    >>> round(float(y[10]), 4), round(1 - float(np.exp(-1)), 4)
    (0.6321, 0.6321)
    """

    real_inputs = True  # a real state cannot take complex inputs

    def __init__(
        self,
        f,
        x0,
        *,
        inputs=("u",),
        outputs=("y",),
        g=None,
        method="RK45",
        substeps=4,
        **kw,
    ):
        self.solver_opts = {k: kw.pop(k) for k in list(kw) if k in _SOLVER_KEYS}
        self.solver_opts.setdefault("rtol", 1e-8)
        self.solver_opts.setdefault("atol", 1e-10)
        self.f, self.g, self.method = f, g, method
        self.x0 = np.atleast_1d(np.asarray(x0, float))
        self.substeps = int(substeps)
        self.inputs, self.outputs = tuple(inputs), tuple(outputs)
        self._g_uses_u = g is not None and _n_positional(g) > 2
        super().__init__(**kw)

    def reset(self):
        """Restore ``x0`` and time 0."""
        self.x = self.x0.copy()
        self.k = 0

    def info(self):
        """Return the number of states and the method."""
        n = len(self.x0)
        return f"{n} state{'s' * (n != 1)}, {self.method}"

    def feedthrough(self, L):
        """Return True if ``g`` uses the input, or for chunks of more than one."""
        return self._g_uses_u or L > 1

    # one sample
    def _y(self, u):
        t = self.k / self.fs
        if self.g is None:
            return self.x.copy()
        if self._g_uses_u:
            return np.atleast_1d(self.g(t, self.x, *u))
        return np.atleast_1d(self.g(t, self.x))

    def _advance(self, u):
        t0 = self.k / self.fs
        if self.method == "RK4":
            h = 1.0 / self.fs / self.substeps
            x, t = self.x, t0

            def f(tt, xx):
                return np.asarray(self.f(tt, xx, *u), dtype=np.result_type(xx, *u))

            for _ in range(self.substeps):
                k1 = f(t, x)
                k2 = f(t + h / 2, x + h / 2 * k1)
                k3 = f(t + h / 2, x + h / 2 * k2)
                k4 = f(t + h, x + h * k3)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            self.x = x
        else:
            sol = solve_ivp(
                self.f,
                (t0, t0 + 1 / self.fs),
                self.x,
                method=self.method,
                args=tuple(u),
                **self.solver_opts,
            )
            if not sol.success:
                raise RuntimeError(
                    f"{self.name}: solve_ivp failed at t={t0}: {sol.message}"
                )
            self.x = sol.y[:, -1]
        self.k += 1

    def _pack(self, ys):
        ys = np.asarray(ys)  # (L, outputs)
        if len(self.outputs) == 1:
            return ys[:, 0] if ys.shape[1] == 1 else ys
        return tuple(ys[:, i] for i in range(len(self.outputs)))

    # block api
    def output(self, L):
        """Return the output at the current sample (used for ``L == 1``)."""
        return self._pack([self._y(())])

    def update(self, L, **u):
        """Integrate over ``L`` samples with the inputs held per sample."""
        cols = [u[k] for k in self.inputs]
        for i in range(L):
            self._advance([c[i] for c in cols])

    def step(self, **u):
        """Return the outputs of one chunk, integrating sample by sample."""
        cols = [u[k] for k in self.inputs]
        ys = []
        for i in range(len(cols[0])):
            ui = [c[i] for c in cols]
            ys.append(self._y(ui))
            self._advance(ui)
        return self._pack(ys)

    def step_source(self, L):
        """Return ``L`` samples of an autonomous system."""
        ys = []
        for _ in range(L):
            ys.append(self._y(()))
            self._advance(())
        return self._pack(ys)
