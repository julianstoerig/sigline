import matplotlib
import numpy as np
import pytest
import scipy.signal as sig

from sigline import Diagram, Noise, Scope, Sine, Sum

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FS = 1000.0


def record(src, n=8192, chunk=512):
    sc = Scope("rx")
    src >> sc
    Diagram(sc, fs=FS).run(n=n, chunk=chunk)
    return sc


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# unit: plots


def test_plot_real_and_complex():
    ax = record(Sine(50.0)).plot()
    assert len(ax.lines) == 1 and ax.get_xlabel() == "t / s"
    _, ax = plt.subplots()
    record(Sine(50.0, complex=True)).plot(ax=ax)
    assert [ln.get_label() for ln in ax.lines] == ["rx (I)", "rx (Q)"]


def test_psd_labels_and_db_scale():
    ax = record(Noise(seed=1)).psd()
    assert ax.get_xlabel() == "f / Hz" and ax.get_ylabel() == "PSD / (dB/Hz)"
    _, P = record(Noise(seed=1)).welch()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 10 * np.log10(P + 1e-30))


# integration and theory: spectra


def test_welch_real_tone_peak_and_power():
    # theory: a sine of amplitude A has power A^2 / 2 = the integral of its PSD
    sc = record(Sine(125.0, amp=2.0))
    f, P = sc.welch(window="boxcar")
    assert f[np.argmax(P)] == 125.0
    np.testing.assert_allclose(np.sum(P) * (f[1] - f[0]), 2.0, rtol=1e-6)


def test_welch_complex_is_two_sided_and_centred():
    sc = record(Sine(-125.0, complex=True))
    f, P = sc.welch(window="boxcar")
    assert f[0] < 0 < f[-1] and np.all(np.diff(f) > 0)
    assert f[np.argmax(P)] == -125.0


def test_welch_keeps_dc():
    # a carrier mixed to 0 Hz must not be removed by detrending
    s = Sum()
    Sine(0.0) >> s["x0"]
    Noise(0.01, seed=2) >> s["x1"]
    f, P = record(s).welch()
    assert f[np.argmax(P)] == 0.0


def test_white_noise_level():
    # theory: one-sided PSD of white noise with variance s^2 is 2 s^2 / fs
    std = 0.5
    f, P = record(Noise(std, seed=3), n=2**17).welch(nperseg=1024)
    np.testing.assert_allclose(np.mean(P[1:-1]), 2 * std**2 / FS, rtol=0.03)


# comparison: welch equals scipy on the raw data


def test_welch_equals_scipy():
    sc = record(Noise(seed=4))
    f, P = sc.welch(nperseg=256)
    f2, P2 = sig.welch(sc.data, fs=FS, nperseg=256, detrend=False)
    np.testing.assert_allclose(P, P2)
