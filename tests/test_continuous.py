import math

import numpy as np
import pytest
import scipy.signal as sig
from scipy.integrate import solve_ivp

import sigline as sl
from sigline import ODE, ContinuousLTI, Diagram, Scope
from sigline._numba import HAVE_NUMBA
from sigline.testing import assert_chunk_invariant

needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="needs numba")


def oscillator(t, x, u):
    # damped mass on a spring, driven by u
    return [x[1], -4 * x[1] - 100 * x[0] + 100 * u]


def position(t, x):
    return x[0]


def nonlinear_plant(t, x, u):
    return np.array([x[1], -x[0] - 0.3 * x[1] + math.tanh(u)])


def closed_loop(make_plant, chunk, mode=True, n=2000):
    # step reference, error, gain 3, plant, unit feedback
    e, sc = sl.Sum("+-"), Scope()
    sl.Step(0.1) >> e["x0"]
    e >> sl.Gain(3.0) >> make_plant() >> sc
    sc._sources["x"].block >> e["x1"]
    d = Diagram(sc, fs=100.0, compile_loops=mode)
    d.run(n=n, chunk=chunk)
    return d, sc.data


# unit


def test_ports_info_and_feedthrough():
    ode = ODE(oscillator, [0, 0], g=position, rtol=1e-6, name="osc")
    assert ode.inputs == ("u",) and ode.outputs == ("y",) and ode.name == "osc"
    assert ode.solver_opts == {"rtol": 1e-6, "atol": 1e-10}
    assert ode.info() == "2 states, RK45" and ode.x0.dtype == float
    assert not ode.feedthrough(1) and ode.feedthrough(2)
    direct = ODE(lambda t, x, u: -x, [1.0], g=lambda t, x, u: x[0] + u)
    assert direct.feedthrough(1)  # g uses the input


def test_reset_restores_the_initial_state():
    ode = ODE(lambda t, x, u: u - x, [0.5], fs=10.0)
    ode(np.ones(20))
    assert ode.k == 20 and ode.x[0] != 0.5
    ode.reset()
    assert ode.k == 0 and ode.x[0] == 0.5


def test_complex_input_rejected():
    src, sc = sl.Sine(1.0, complex=True), Scope()
    src >> ODE(lambda t, x, u: [-x[0] + u], [0.0]) >> sc
    with pytest.raises(TypeError, match="expects real signals"):
        Diagram(sc, fs=10.0).run(n=10)


def test_vector_output_without_g():
    ode = ODE(oscillator, [0.0, 0.0], fs=200.0)
    y = ode(np.ones(5))
    assert y.shape == (5, 2)


# integration


def test_loop_through_an_ode_at_any_chunk():
    plant = lambda: ODE(nonlinear_plant, [0.0, 0.0], g=position, method="RK4")  # noqa: E731
    ref = closed_loop(plant, chunk=1)[1]
    for chunk in (7, 400):
        np.testing.assert_allclose(
            closed_loop(plant, chunk)[1][: len(ref)], ref, atol=1e-12
        )


def test_solve_ivp_ode_in_a_loop_runs_in_python():
    plant = lambda: ODE(nonlinear_plant, [0.0, 0.0], g=position)  # noqa: E731
    if HAVE_NUMBA:
        with pytest.warns(RuntimeWarning, match="method 'RK45'"):
            d, y = closed_loop(plant, chunk=100, n=300)
    else:
        d, y = closed_loop(plant, chunk=100, n=300)
    assert d.loops(100)[0].mode == "python"
    ref = closed_loop(lambda: ODE(nonlinear_plant, [0.0, 0.0], g=position, method="RK4",
                                  substeps=16), chunk=100, n=300)[1]  # fmt: skip
    np.testing.assert_allclose(y, ref, atol=1e-6)


# comparison


def test_chunk_invariant():
    u = np.sin(np.arange(300) * 0.05)
    assert_chunk_invariant(lambda: ODE(oscillator, [0.1, 0], g=position), u, fs=200.0)
    assert_chunk_invariant(
        lambda: ODE(oscillator, [0.1, 0], g=position, method="RK4"), u, fs=200.0
    )


def test_rk4_close_to_rk45():
    u = np.ones(400)
    y45 = ODE(oscillator, [0, 0], g=position, fs=200.0)(u)
    y4 = ODE(oscillator, [0, 0], g=position, method="RK4", substeps=8, fs=200.0)(u)
    np.testing.assert_allclose(y4, y45, atol=1e-7)


@needs_numba
def test_compiled_rk4_loop_equals_python():
    plant = lambda: ODE(nonlinear_plant, [0.0, 0.0], g=position, method="RK4")  # noqa: E731
    d1, y1 = closed_loop(plant, chunk=400)
    d2, y2 = closed_loop(plant, chunk=400, mode="python")
    assert d1.loops(400)[0].mode == "numba" and d2.loops(400)[0].mode == "python"
    np.testing.assert_allclose(y1, y2, atol=1e-12)


# theory


def test_step_response_equals_one_solve_ivp():
    fs = 200.0
    osc, sc = ODE(oscillator, [0, 0], outputs=("pos",), g=position), Scope()
    sl.Step(0.1) >> osc >> sc
    Diagram(sc, fs=fs).run(2.0, chunk=64)
    # the step at 0.1 s falls on a sample, so one integration from there
    sol = solve_ivp(lambda t, x: oscillator(t, x, 1.0), (0.1, 2.0), [0, 0],
                    t_eval=np.arange(20, 400) / fs, rtol=1e-10, atol=1e-12)  # fmt: skip
    np.testing.assert_allclose(sc.data[:20], 0.0)
    np.testing.assert_allclose(sc.data[20:], sol.y[0], atol=1e-6)


def test_held_input_low_pass_equals_exact_zoh():
    # dx/dt = (u - x) / tau with u held: x[k+1] = a x[k] + (1 - a) u[k]
    fs, tau = 100.0, 0.05
    u = np.random.default_rng(1).standard_normal(300)
    y = ODE(lambda t, x, u: (u - x) / tau, [0.0], fs=fs)(u)
    a, ref, x = math.exp(-1 / (fs * tau)), np.empty(300), 0.0
    for k in range(300):
        ref[k], x = x, a * x + (1 - a) * u[k]
    np.testing.assert_allclose(y, ref, atol=1e-8)


def test_autonomous_oscillator_is_a_cosine():
    w = 2 * np.pi * 5
    osc, sc = ODE(lambda t, x: [x[1], -(w**2) * x[0]], [1.0, 0.0], inputs=(),
                  g=position), Scope()  # fmt: skip
    osc >> sc
    d = Diagram(sc, fs=1000.0)
    d.run(1.0, chunk=100)
    np.testing.assert_allclose(sc.data, np.cos(w * d.t), atol=1e-6)


# ContinuousLTI

FS, TS, TAU, KP, KI = 1000.0, 1e-3, 0.1, 2.0, 20.0


def pi_plant(chunk, mode=True, method="zoh", n=3000):
    """discrete PI controller, continuous first-order plant, unit feedback."""
    e, pi, sc = sl.Sum("+-"), sl.IIR(b=[KP + KI * TS, -KP], a=[1, -1]), Scope()
    plant = ContinuousLTI([1], [TAU, 1], method=method)
    sl.Step(0.0) >> e["x0"]
    e >> pi >> plant >> sc
    plant >> e["x1"]
    d = Diagram(sc, fs=FS, compile_loops=mode)
    d.run(n=n // chunk * chunk, chunk=chunk)
    return d, sc.data


def reference_pi(n=3000):
    # exact zoh plant and the same pi, as a plain loop
    a = math.exp(-TS / TAU)
    y = u = e_prev = 0.0
    out = []
    for _ in range(n):
        out.append(y)
        e = 1.0 - y
        u += (KP + KI * TS) * e - KP * e_prev
        e_prev = e
        y = a * y + (1 - a) * u
    return np.array(out)


def test_lti_from_tf_and_state_space():
    a = ContinuousLTI([1.0], [0.1, 1.0])
    assert a.A.shape == (1, 1) and a.x.tolist() == [0.0]
    assert a.info() == "order 1, exact ZOH" and a.real_inputs is False
    b = ContinuousLTI(A=[[0, 1], [-100, -4]], B=[[0], [100]], C=[[1, 0]], D=[[0]],
                      method="RK45")  # fmt: skip
    assert b.info() == "order 2, RK45" and b.real_inputs is True
    assert not a.feedthrough(1) and ContinuousLTI([1.0, 0.0], [1.0, 1.0]).feedthrough(1)


def test_lti_bode_equals_closed_form():
    f = np.array([0.1, 1.0, 10.0, 100.0])
    H = ContinuousLTI([1.0], [TAU, 1.0]).bode(f)
    np.testing.assert_allclose(H, 1 / (1 + 2j * np.pi * f * TAU))


def test_lti_takes_complex_inputs():
    u = np.exp(1j * np.arange(200) * 0.1)
    y = ContinuousLTI([1.0], [TAU, 1.0], fs=FS)(u)
    re = ContinuousLTI([1.0], [TAU, 1.0], fs=FS)(u.real)
    im = ContinuousLTI([1.0], [TAU, 1.0], fs=FS)(u.imag)
    np.testing.assert_allclose(y, re + 1j * im, atol=1e-14)


@pytest.mark.parametrize("chunk", [1, 7, 1000])
@pytest.mark.parametrize("mode", [True, "python"])
def test_pi_loop_with_lti_plant_equals_reference(chunk, mode):
    d, y = pi_plant(chunk, mode)
    np.testing.assert_allclose(y, reference_pi()[: len(y)], atol=1e-10)
    if chunk > 1:
        compiled = mode is True and HAVE_NUMBA
        assert d.loops(chunk)[0].mode == ("numba" if compiled else "python")


def test_lti_by_integration_in_a_loop_close_to_zoh():
    if HAVE_NUMBA:
        with pytest.warns(RuntimeWarning, match="zoh compiles"):
            _, y = pi_plant(500, method="RK45", n=1000)
    else:
        _, y = pi_plant(500, method="RK45", n=1000)
    np.testing.assert_allclose(y, reference_pi(1000), atol=1e-6)


def test_lti_chunk_invariant():
    u = np.random.default_rng(3).standard_normal(400)
    assert_chunk_invariant(lambda: ContinuousLTI([100.0], [1, 2, 100]), u, fs=FS)


@needs_numba
def test_compiled_zoh_equals_numpy_zoh():
    from sigline.continuous import _zoh, _zoh_numpy

    lti = ContinuousLTI(A=[[0, 1], [-100, -4]], B=[[0], [100]], C=[[1, 0], [0, 1]],
                        D=[[0], [0.5]], fs=FS)  # fmt: skip
    Ad, Bd = lti._discretise()
    u = np.random.default_rng(4).standard_normal((300, 1))
    y1, y2 = np.zeros((300, 2)), np.zeros((300, 2))
    x1 = _zoh(Ad, Bd, lti.C, lti.D, np.zeros(2), u, y1)
    x2 = _zoh_numpy(Ad, Bd, lti.C, lti.D, np.zeros(2), u, y2)
    np.testing.assert_allclose(y1, y2, atol=1e-13)
    np.testing.assert_allclose(x1, x2, atol=1e-13)


def test_lti_equals_scipy_zoh_discretisation():
    num, den = [100.0], [1.0, 2.0, 100.0]
    u = np.random.default_rng(0).standard_normal(2000)
    y = ContinuousLTI(num, den, fs=FS)(u)
    ss = sig.cont2discrete(sig.tf2ss(num, den), 1 / FS, method="zoh")
    np.testing.assert_allclose(y, sig.dlsim(ss, u)[1][:, 0], atol=1e-12)


def test_lti_step_response_is_exact_at_the_samples():
    y = ContinuousLTI([1.0], [TAU, 1.0], fs=FS)(np.ones(500))
    np.testing.assert_allclose(y, 1 - np.exp(-np.arange(500) / FS / TAU), atol=1e-14)


def test_zoh_never_reads_uninitialised_memory(monkeypatch):
    # regression: sums started from np.empty memory, NaN * 0 is NaN
    from sigline.continuous import _zoh_loops

    monkeypatch.setattr(np, "empty_like", lambda a: np.full_like(a, np.nan))
    lti = ContinuousLTI([100.0], [1, 2, 100], fs=FS)
    Ad, Bd = lti._discretise()
    y = np.zeros((50, 1))
    _zoh_loops(Ad, Bd, lti.C, lti.D, np.zeros(2), np.ones((50, 1)), y)
    assert np.all(np.isfinite(y))
