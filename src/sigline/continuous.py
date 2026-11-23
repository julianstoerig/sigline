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
import scipy.signal as sig
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ._numba import HAVE_NUMBA, njit
from .core import StateBlock

__all__ = ["ContinuousLTI", "ODE"]

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


def _zoh_loops(Ad, Bd, C, D, x, u, y):
    # y[i] = C x + D u[i]; x <- Ad x + Bd u[i]; explicit loops for numba
    n, m, p = x.shape[0], u.shape[1], C.shape[0]
    tmp = np.empty_like(x)
    for i in range(u.shape[0]):
        for r in range(p):
            acc = y[i, r]
            for j in range(n):
                acc += C[r, j] * x[j]
            for j in range(m):
                acc += D[r, j] * u[i, j]
            y[i, r] = acc
        for r in range(n):
            # start from a term, never from tmp: its memory is uninitialised
            acc = Bd[r, 0] * u[i, 0]
            for j in range(1, m):
                acc += Bd[r, j] * u[i, j]
            for j in range(n):
                acc += Ad[r, j] * x[j]
            tmp[r] = acc
        x[:] = tmp
    return x


def _zoh_numpy(Ad, Bd, C, D, x, u, y):
    # the same with matrix products, for runs without numba
    for i in range(u.shape[0]):
        y[i] = C @ x + D @ u[i]
        x = Ad @ x + Bd @ u[i]
    return x


_zoh = njit(_zoh_loops) if HAVE_NUMBA else _zoh_numpy


class ContinuousLTI(ODE):
    """Continuous LTI system from a transfer function or state space.

    ``ContinuousLTI(num, den)`` or ``ContinuousLTI(A=, B=, C=, D=)``. With
    ``method="zoh"`` the system is discretised exactly for inputs held over
    each sample (matrix exponential): exact, and fast, compiled with numba.
    A ``solve_ivp`` method integrates numerically instead, e.g. as a cross
    check.

    Parameters
    ----------
    num, den : array_like, optional
        Transfer function coefficients, highest power of ``s`` first.
    A, B, C, D : array_like, optional
        State space, instead of ``num, den``.
    x0 : array_like, optional
        Initial state; default zeros.
    method : str, default "zoh"
        ``"zoh"``, or a method for :class:`ODE`.
    input, output : str, default "u", "y"
        Port names.
    **kw
        As for :class:`ODE`.

    Examples
    --------
    ``1 / (tau s + 1)`` driven by a held 1 reaches ``1 - e^-1`` after ``tau``:

    >>> import numpy as np
    >>> from sigline import ContinuousLTI
    >>> lp = ContinuousLTI([1.0], [0.01, 1.0], fs=1000.0)
    >>> y = lp(np.ones(30))
    >>> round(float(y[10]), 6) == round(1 - float(np.exp(-1)), 6)
    True
    """

    def __init__(
        self,
        num=None,
        den=None,
        *,
        A=None,
        B=None,
        C=None,
        D=None,
        x0=None,
        method="zoh",
        input="u",
        output="y",
        **kw,
    ):
        if num is not None:
            A, B, C, D = sig.tf2ss(num, den)
        self.A, self.B, self.C, self.D = (
            np.atleast_2d(np.asarray(m, float)) for m in (A, B, C, D)
        )
        self.zoh = method == "zoh"
        self._Ad = self._Bd = self._fs_d = None
        x0 = np.zeros(self.A.shape[0]) if x0 is None else x0
        if np.any(self.D):

            def g(t, x, u):
                return self.C @ x + self.D @ np.atleast_1d(u)
        else:

            def g(t, x):
                return self.C @ x

        def f(t, x, u):
            return self.A @ x + self.B @ np.atleast_1d(u)

        super().__init__(
            f,
            x0,
            inputs=(input,),
            outputs=(output,),
            g=g,
            method="RK45" if self.zoh else method,
            **kw,
        )

    @property
    def real_inputs(self):
        """Exact ZOH is linear and takes complex inputs; integrators do not."""
        return not self.zoh

    def info(self):
        """Return the order and the method."""
        how = "exact ZOH" if self.zoh else self.method
        return f"order {self.A.shape[0]}, {how}"

    def _discretise(self):
        # Ad, Bd for the current fs, once
        if self._fs_d != self.fs:
            n, m = self.B.shape
            M = np.block([[self.A, self.B], [np.zeros((m, n + m))]]) / self.fs
            E = expm(M)
            self._Ad, self._Bd, self._fs_d = E[:n, :n], E[:n, n:], self.fs
        return self._Ad, self._Bd

    def _run(self, u):
        # ZOH over a chunk: outputs (L, p), state advanced
        Ad, Bd = self._discretise()
        u = np.asarray(u).reshape(len(u), -1)
        dt = np.result_type(self.x, u)
        y = np.zeros((len(u), self.C.shape[0]), dt)
        args = [np.ascontiguousarray(a, dt) for a in (Ad, Bd, self.C, self.D)]
        self.x = _zoh(*args, self.x.astype(dt), np.ascontiguousarray(u, dt), y)
        self.k += len(u)
        return y

    def update(self, L, **u):
        """Advance over ``L`` samples."""
        if not self.zoh:
            return super().update(L, **u)
        self._run(u[self.inputs[0]])

    def step(self, **u):
        """Return the outputs of one chunk."""
        if not self.zoh:
            return super().step(**u)
        return self._pack(self._run(u[self.inputs[0]]))

    def bode(self, f):
        """Return the continuous response ``H(j 2 pi f)`` of a SISO system.

        Parameters
        ----------
        f : array_like
            Frequencies / Hz.

        Returns
        -------
        numpy.ndarray
            Complex response.
        """
        b, a = sig.ss2tf(self.A, self.B, self.C, self.D)
        _, H = sig.freqs(b[0], a, worN=2 * np.pi * np.asarray(f))
        return H
