import numpy as np
import pytest
import scipy.signal as sig

from sigline import FIR, Diagram, FromArray, Scope, Sine
from sigline.testing import assert_chunk_invariant

rng = np.random.default_rng(7)
X = rng.standard_normal(2000) + 1j * rng.standard_normal(2000)
TAPS = sig.firwin(33, 0.2)
LONG = sig.firwin(201, 0.2)


def filtered(taps, x, chunk):
    src, f, sc = FromArray(x), FIR(taps), Scope()
    src >> f >> sc
    Diagram(sc, fs=1.0).run(n=len(x), chunk=chunk)
    return sc.data


# unit


def test_equals_lfilter_real_complex_vector():
    xr = X.real
    np.testing.assert_allclose(FIR(TAPS)(xr), sig.lfilter(TAPS, 1, xr), atol=1e-12)
    np.testing.assert_allclose(FIR(TAPS)(X), sig.lfilter(TAPS, 1, X), atol=1e-12)
    v = np.stack([X.real, X.imag], axis=1)
    np.testing.assert_allclose(
        FIR(TAPS)(v), sig.lfilter(TAPS, 1, v, axis=0), atol=1e-12
    )


def test_single_tap_is_a_gain():
    np.testing.assert_allclose(FIR([2.0])(X), 2 * X)


def test_taps_change_keeps_history():
    # switching taps between chunks equals filtering with each over the joint input
    f = FIR(TAPS)
    a = f(X[:1000])
    f.taps = sig.firwin(21, 0.1)
    b = f(X[1000:])
    np.testing.assert_allclose(a, sig.lfilter(TAPS, 1, X)[:1000], atol=1e-12)
    np.testing.assert_allclose(b, sig.lfilter(f.taps, 1, X)[1000:], atol=1e-12)


@pytest.mark.parametrize("taps", [TAPS, LONG], ids=["direct", "fft"])
def test_chunk_invariant(taps):
    assert_chunk_invariant(lambda: FIR(taps), X)


# comparison: fft path equals direct path, diagram equals scipy one-shot


@pytest.mark.parametrize("chunk", [1, 100, 2000])
def test_long_filter_any_chunk_equals_lfilter(chunk):
    np.testing.assert_allclose(
        filtered(LONG, X, chunk), sig.lfilter(LONG, 1, X), atol=1e-10
    )


# theory: moving average response |H| = |sin(pi f M) / (M sin(pi f))|


def test_freqz_of_moving_average():
    M, fs = 8, 1000.0
    f, H = FIR(np.ones(M) / M, fs=fs).freqz(n=512)
    x = np.pi * f[1:] / fs
    np.testing.assert_allclose(
        np.abs(H[1:]), np.abs(np.sin(M * x) / (M * np.sin(x))), atol=1e-12
    )


def test_lowpass_attenuates_stopband_sine():
    # a sine at the passband passes, one in the stopband is suppressed by > 50 dB
    fs = 1000.0
    for f0, expect_pass in [(30.0, True), (300.0, False)]:
        s, lp, sc = Sine(f0), FIR(sig.firwin(101, 100.0, fs=fs)), Scope()
        s >> lp >> sc
        Diagram(sc, fs=fs).run(n=4000, chunk=256)
        rms = np.sqrt(np.mean(sc.data[200:] ** 2)) * np.sqrt(2)
        assert (abs(rms - 1) < 0.01) if expect_pass else (rms < 10 ** (-50 / 20))
