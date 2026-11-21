import matplotlib
import numpy as np
import pytest
import scipy.signal as sig

from sigline import Diagram, FromArray, Noise, PowerMeter, Sine, SpectrumAnalyzer, Sum

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def feed(block, x, fs=1.0, chunk=1000):
    src = FromArray(x)
    src >> block
    d = Diagram(block, fs=fs)
    d.run(n=len(x), chunk=chunk)
    return d


# spectrum analyzer


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("chunk", [1, 777, 20_000])
def test_spectrum_equals_scipy_welch_for_any_chunking(cplx, chunk):
    # comparison with scipy one-shot; chunk sizes 1 .. whole record
    n = np.arange(20_000)
    x = rng.standard_normal(len(n))
    x = (
        x + 1j * rng.standard_normal(len(n)) + np.exp(0.2j * np.pi * n)
        if cplx
        else x + np.cos(0.3 * n)
    )
    sa = SpectrumAnalyzer(nfft=1024, overlap=0.5, waterfall=5)
    feed(sa, x, fs=1000.0, chunk=chunk)
    f, P = sig.welch(x, fs=1000.0, nperseg=1024, noverlap=512, detrend=False,
                     return_onesided=not cplx)  # fmt: skip
    if cplx:
        f, P = np.fft.fftshift(f), np.fft.fftshift(P)
    np.testing.assert_allclose(sa.f, f)
    np.testing.assert_allclose(sa.psd, P, rtol=1e-9)
    assert sa.frames == 1 + (20_000 - 1024) // 512
    assert np.all(sa.max_hold >= sa.psd * (1 - 1e-12))
    assert sa.waterfall.shape == (5, len(f))


def test_white_noise_level_and_tone_power():
    # theory: white noise PSD 2 s^2 / fs one-sided; a sine's PSD integrates to A^2/2
    fs, std, A = 1000.0, 0.5, 2.0
    sa = SpectrumAnalyzer(nfft=1024, window="boxcar")
    feed(sa, std * rng.standard_normal(2**17), fs=fs)
    np.testing.assert_allclose(np.mean(sa.psd[1:-1]), 2 * std**2 / fs, rtol=0.02)
    tone = SpectrumAnalyzer(nfft=1000, window="boxcar")
    feed(tone, A * np.cos(2 * np.pi * 125.0 * np.arange(10_000) / fs), fs=fs)
    np.testing.assert_allclose(np.sum(tone.psd) * fs / 1000, A**2 / 2, rtol=1e-9)
    assert tone.peak()[0] == 125.0


def test_exponential_average_tracks_a_change():
    x = np.r_[np.zeros(50_000), np.ones(50_000)] + 0.01 * rng.standard_normal(100_000)
    sa = SpectrumAnalyzer(nfft=256, average="exp", alpha=0.2)
    feed(sa, x, chunk=4096)
    assert sa.peak()[0] == 0.0 and sa.psd[0] > 100 * np.median(sa.psd)


def test_skip_errors_and_info():
    sa = SpectrumAnalyzer(nfft=64, skip=100)
    feed(sa, np.ones(150), chunk=7)
    with pytest.raises(RuntimeError, match="no complete frame"):
        _ = sa.psd
    assert sa.info() == "64-pt, mean"
    with pytest.raises(ValueError, match="average"):
        SpectrumAnalyzer(average="median")
    with pytest.raises(ValueError, match="one component"):
        SpectrumAnalyzer(fs=1.0)(np.ones((10, 2)))


def test_as_a_probe_with_plots():
    s, add = Sine(1e3), Sum()
    s >> add["x0"]
    Noise(0.1, seed=1) >> add["x1"]
    d = Diagram(add, fs=8000.0)
    sa = d.probe(add, sink=SpectrumAnalyzer(nfft=512, waterfall=20, label="rx"))
    d.run(1.0)
    assert sa.fs == 8000.0 and abs(sa.peak()[0] - 1e3) <= 8000 / 512
    ax = sa.plot()
    assert ax.get_xlabel() == "f / Hz" and ax.get_ylabel() == "PSD / (dB/Hz)"
    ax2 = sa.plot_waterfall(ax=plt.figure().gca())
    assert ax2.get_ylabel() == "t / s"


# power meter


@pytest.mark.parametrize("chunk", [1, 333, 2000])
def test_power_meter_readings_for_any_chunking(chunk):
    x = np.r_[np.ones(1000) * 0.1, np.ones(1000) * 1.0]  # 0.01 W, then 1 W
    pm = PowerMeter(interval=0.5, unit="dBm")
    feed(pm, x, fs=1000.0, chunk=chunk)
    np.testing.assert_allclose(pm.readings, [10, 10, 30, 30], atol=1e-9)
    np.testing.assert_allclose(pm.readings_t, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(pm.level, 10 * np.log10(0.505 / 1e-3), atol=1e-9)


def test_power_meter_full_scale_and_vector_signals():
    # theory: a full-scale sine is 0 dBFS (real), so is a full-scale exponential
    fsine = PowerMeter(unit="dBFS", full_scale=2.0)
    feed(fsine, 2.0 * np.sin(0.1 * np.arange(100_000)))
    assert abs(fsine.level) < 0.01
    fc = PowerMeter(unit="dBFS", full_scale=2.0)
    feed(fc, 2.0 * np.exp(0.1j * np.arange(1000)))
    assert abs(fc.level) < 1e-9
    pv = PowerMeter()
    feed(pv, np.ones((100, 2)))
    assert pv.power == 2.0  # channels add


def test_power_meter_matches_noise_variance_and_validates_unit():
    pm = PowerMeter()
    feed(pm, 0.3 * rng.standard_normal(200_000))
    assert abs(pm.power / 0.09 - 1) < 5 * np.sqrt(2 / 200_000)
    assert pm.info() == "dB" and PowerMeter(0.1, "dBm").info() == "dBm every 100 ms"
    with pytest.raises(ValueError, match="unit"):
        PowerMeter(unit="W")
