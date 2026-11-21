import numpy as np
import pytest
import scipy.signal as sig

from sigline import IIR, Diagram, FromArray, Scope, Step
from sigline.testing import assert_chunk_invariant

rng = np.random.default_rng(8)
XR = rng.standard_normal(1500)
XC = XR + 1j * rng.standard_normal(1500)
SOS = sig.butter(4, 0.2, output="sos")


# unit


def test_sos_and_ba_equal_scipy():
    np.testing.assert_allclose(IIR(sos=SOS)(XR), sig.sosfilt(SOS, XR), atol=1e-12)
    b, a = [0.2, 0.1], [1, -0.7]
    np.testing.assert_allclose(IIR(b=b, a=a)(XC), sig.lfilter(b, a, XC), atol=1e-12)
    np.testing.assert_allclose(IIR(b=[0.5, 0.5])(XR), sig.lfilter([0.5, 0.5], 1, XR))


def test_vector_signals_filter_each_channel():
    v = np.stack([XR, XR[::-1]], axis=1)
    np.testing.assert_allclose(IIR(sos=SOS)(v), sig.sosfilt(SOS, v, axis=0), atol=1e-12)


def test_needs_coefficients():
    with pytest.raises(ValueError, match="give sos"):
        IIR()


@pytest.mark.parametrize(
    "make",
    [lambda: IIR(sos=SOS), lambda: IIR(b=[0.2, 0.1], a=[1, -0.7])],
    ids=["sos", "ba"],
)
def test_chunk_invariant(make):
    assert_chunk_invariant(make, XC)


# comparison: sos and b, a forms of the same filter agree in a diagram


@pytest.mark.parametrize("chunk", [1, 64, 1500])
def test_sos_equals_ba_form(chunk):
    b, a = sig.butter(3, 0.15)
    outs = []
    for blk in (IIR(sos=sig.tf2sos(b, a)), IIR(b=b, a=a)):
        src, sc = FromArray(XR), Scope()
        src >> blk >> sc
        Diagram(sc, fs=1.0).run(n=1500, chunk=chunk)
        outs.append(sc.data)
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-10)


# theory: step responses and frequency response of a first-order lowpass


def test_first_order_lowpass_step_response():
    # y[n] = a y[n-1] + (1 - a) u[n]  ->  y[n] = 1 - a^(n+1) for a unit step
    a = 0.9
    st, lp, sc = Step(), IIR(b=[1 - a], a=[1, -a]), Scope()
    st >> lp >> sc
    Diagram(sc, fs=1.0).run(n=100, chunk=7)
    np.testing.assert_allclose(sc.data, 1 - a ** (np.arange(100) + 1), atol=1e-12)


def test_pi_controller_step_response_is_ramp():
    # u[n] = u[n-1] + (kp + ki Ts) e[n] - kp e[n-1]; unit step -> kp + ki Ts (n + 1)
    kp, ki, fs = 2.0, 50.0, 1000.0
    st, pi, sc = Step(), IIR(b=[kp + ki / fs, -kp], a=[1, -1]), Scope()
    st >> pi >> sc
    Diagram(sc, fs=fs).run(n=200, chunk=33)
    np.testing.assert_allclose(sc.data, kp + ki / fs * (np.arange(200) + 1), atol=1e-12)


def test_freqz_of_first_order_lowpass():
    a, fs = 0.8, 100.0
    f, H = IIR(b=[1 - a], a=[1, -a], fs=fs).freqz(n=256)
    z = np.exp(2j * np.pi * f / fs)
    np.testing.assert_allclose(H, (1 - a) / (1 - a / z), atol=1e-12)
    f2, H2 = IIR(sos=SOS, fs=fs).freqz(n=256)
    np.testing.assert_allclose(np.abs(H2[0]), 1.0, atol=1e-12)
