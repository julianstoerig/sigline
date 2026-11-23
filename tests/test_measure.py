import matplotlib
import numpy as np
import pytest
import scipy.signal as sig
from scipy.special import erfc

from sigline import (
    BERCounter,
    ConstellationScope,
    Diagram,
    EyeDiagram,
    FromArray,
    FuncChunk,
    Noise,
    PowerMeter,
    Signal,
    Sine,
    SNREstimator,
    Spectrogram,
    SpectrumAnalyzer,
    Sum,
    qam_ambiguities,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(0)
QPSK = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
LEVELS = np.arange(-7, 8, 2) / np.sqrt(42)
QAM64 = np.array([a + 1j * b for a in LEVELS for b in LEVELS])


def noisy(const, snr_db, n=100_000):
    a = const[rng.integers(0, len(const), n)]
    w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return a + np.sqrt(10 ** (-snr_db / 10) / 2) * w


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
    sa = SpectrumAnalyzer(nfft=1024, overlap=0.5)
    feed(sa, x, fs=1000.0, chunk=chunk)
    f, P = sig.welch(x, fs=1000.0, nperseg=1024, noverlap=512, detrend=False,
                     return_onesided=not cplx)  # fmt: skip
    if cplx:
        f, P = np.fft.fftshift(f), np.fft.fftshift(P)
    np.testing.assert_allclose(sa.f, f)
    np.testing.assert_allclose(sa.psd, P, rtol=1e-9)
    assert sa.frames == 1 + (20_000 - 1024) // 512
    assert np.all(sa.max_hold >= sa.psd * (1 - 1e-12))


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
    sa = d.probe(add, sink=SpectrumAnalyzer(nfft=512, name="rx"))
    d.run(1.0)
    assert sa.fs == 8000.0 and abs(sa.peak()[0] - 1e3) <= 8000 / 512
    ax = sa.plot()
    assert ax.get_xlabel() == "f / Hz" and ax.get_ylabel() == "PSD / (dB/Hz)"
    assert ax.get_legend().get_texts()[0].get_text() == "rx"


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


# constellation


@pytest.mark.parametrize(("const", "snr_db"), [(QPSK, 15.0), (QAM64, 25.0)])
def test_mer_and_evm_match_the_snr(const, snr_db):
    # theory: at high SNR, MER = SNR and EVM = 100 / sqrt(SNR) %
    cs = ConstellationScope(const, keep=500)
    feed(cs, noisy(const, snr_db), chunk=4096)
    assert abs(cs.mer_db - snr_db) < 0.2
    assert abs(cs.evm_percent - 100 * 10 ** (-snr_db / 20)) < 1
    assert len(cs.points) == 500


@pytest.mark.parametrize("chunk", [1, 999])
def test_constellation_any_chunking_and_skip(chunk):
    y = noisy(QPSK, 12.0, n=5000)
    ref = ConstellationScope(QPSK, keep=100, skip=1000)
    feed(ref, y, chunk=5000)
    cs = ConstellationScope(QPSK, keep=100, skip=1000)
    feed(cs, y, chunk=chunk)
    assert cs.count == 4000 and cs.mer_db == pytest.approx(ref.mer_db, abs=1e-9)
    np.testing.assert_array_equal(cs.points, y[-100:])


def test_constellation_plot_and_without_reference():
    cs = ConstellationScope(QPSK, name="rx")
    feed(cs, noisy(QPSK, 20.0, n=1000))
    ax = cs.plot()
    assert "MER" in ax.get_title() and (ax.get_xlabel(), ax.get_ylabel()) == ("I", "Q")
    bare = ConstellationScope(keep=10)
    feed(bare, noisy(QPSK, 20.0, n=100))
    assert bare.info() == "last 10" and np.isnan(bare.evm_percent)


# eye diagram


def test_eye_keeps_all_traces_or_the_newest():
    sps = 8
    x = np.repeat(rng.choice([-1.0, 1.0], 2000), sps)  # NRZ
    n_tr = (len(x) - (2 * sps + 1)) // sps + 1
    every = EyeDiagram(sps, span=2)  # default: all, as a Scope
    feed(every, x, chunk=37)
    assert every.traces_seen == n_tr and every.traces.shape == (n_tr, every.length)
    np.testing.assert_allclose(every.traces[0], x[: every.length])
    newest = EyeDiagram(sps, span=2, keep=10)
    feed(newest, x, chunk=37)
    assert newest.traces.shape == (10, newest.length)
    np.testing.assert_allclose(newest.traces, every.traces[-10:])


def test_eye_is_fully_open_at_the_sampling_instant():
    # theory: NRZ through an integrate-and-dump filter of one symbol is exactly
    # +-1 at the end of each symbol, i.e. at trace positions 0, sps, 2 sps
    sps = 8
    x = sig.lfilter(np.ones(sps) / sps, 1, np.repeat(rng.choice([-1.0, 1.0], 500), sps))
    eye = EyeDiagram(sps, offset=sps - 1, keep=1000)
    feed(eye, x, chunk=100)
    tr = eye.traces
    for col in (0, sps, 2 * sps):
        np.testing.assert_allclose(np.abs(tr[:, col]), 1.0, atol=1e-12)
    assert len(np.unique(np.round(tr[:, 1], 9))) > 2  # between instants it closes


def test_eye_traces_are_chunk_invariant():
    x = sig.lfilter(np.ones(8) / 8, 1, np.repeat(rng.choice([-1.0, 1.0], 500), 8))
    traces = []
    for chunk in (1, 100, 4000):
        e = EyeDiagram(8)
        feed(e, x, chunk=chunk)
        traces.append(e.traces)
    np.testing.assert_array_equal(traces[0], traces[1])
    np.testing.assert_array_equal(traces[0], traces[2])


def test_eye_density_counts_each_trace_once_per_position():
    sps = 8
    x = sig.lfilter(np.ones(sps) / sps, 1, np.repeat(rng.choice([-1.0, 1.0], 500), sps))
    eye = EyeDiagram(sps, offset=sps - 1)
    feed(eye, x, chunk=100)
    t, edges, counts = eye.density(bins=60, interp=4)
    assert counts.shape == (len(t), 60) and len(edges) == 61
    np.testing.assert_array_equal(counts.sum(axis=1), eye.traces_seen)
    # theory: integrate-and-dump NRZ is exactly +-1 at the symbol instants
    for col in (0, 4 * sps, 8 * sps):  # t = 0, 1, 2 symbols
        hit = edges[np.flatnonzero(counts[col])]
        assert len(hit) == 2 and hit[0] <= -1 < hit[0] + edges[1] - edges[0]
    assert np.count_nonzero(counts[2]) > 2  # between instants the eye closes


def test_eye_plots_a_map_that_suits_light_and_dark_backgrounds():
    eye = EyeDiagram(4, component="abs", name="rx")
    feed(eye, np.exp(1j * np.arange(2000.0)) * (1 + 0.1 * rng.standard_normal(2000)))
    for style in ("default", "dark_background"):
        with plt.style.context(style):
            ax = eye.plot(ax=plt.figure().gca())
            c0 = matplotlib.colors.to_rgb("C0")  # the style's first colour
        (img,) = ax.images
        rgba = np.asarray(img.get_array())
        assert rgba.shape[-1] == 4 and ax.get_xlabel() == "t / T" and not ax.lines
        np.testing.assert_allclose(
            rgba[..., :3], np.broadcast_to(c0, rgba[..., :3].shape)
        )
        alpha = rgba[..., 3]
        assert alpha.max() == 1.0 and (alpha == 0).mean() > 0.3  # empty: see-through
    red = np.asarray(eye.plot(ax=plt.figure().gca(), color="r").images[0].get_array())
    assert np.all(red[..., :3] == (1.0, 0.0, 0.0))
    assert ax.get_title() == f"rx: {eye.traces_seen} traces"
    assert EyeDiagram(4).plot(ax=plt.figure().gca()).images  # empty: no error
    assert eye.info() == "4 sps, 2 UI"
    with pytest.raises(ValueError, match="component"):
        EyeDiagram(4, component="phase")


# snr estimator


@pytest.mark.parametrize(("const", "snr_db"), [(QPSK, 15.0), (QAM64, 25.0)])
def test_snr_estimators_match_the_snr(const, snr_db):
    y = noisy(const, snr_db)
    m2 = SNREstimator("m2m4", constellation=const)
    dd = SNREstimator("dd", constellation=const)
    src = FromArray(y)
    src >> m2
    src >> dd
    Diagram(m2, fs=1.0).run(n=len(y), chunk=4096)
    assert abs(m2.snr_db - snr_db) < 0.5
    assert abs(dd.snr_db - snr_db) < 0.3


def test_known_limits_of_both_methods():
    # dd counts wrong decisions as signal: biased upward at low SNR
    # (measured: 1.76 dB instead of 0 dB for QPSK)
    dd = SNREstimator("dd", constellation=QPSK)
    feed(dd, noisy(QPSK, 0.0))
    assert dd.snr_db > 1.0
    # m2m4 needs no decisions and stays unbiased there
    m2 = SNREstimator("m2m4", kurtosis=1.0)
    feed(m2, noisy(QPSK, 0.0))
    assert abs(m2.snr_db) < 0.3


@pytest.mark.parametrize("chunk", [1, 777])
def test_snr_any_chunking_and_skip(chunk):
    y = np.r_[np.zeros(1000), noisy(QPSK, 10.0, n=4000)]
    a, b = (
        SNREstimator("dd", constellation=QPSK, skip=1000),
        SNREstimator("dd", constellation=QPSK, skip=1000),
    )
    feed(a, y, chunk=chunk)
    feed(b, y, chunk=len(y))
    assert a.count == 4000 and a.snr == pytest.approx(b.snr, rel=1e-12)


def test_snr_estimator_validates_arguments():
    with pytest.raises(ValueError, match="method"):
        SNREstimator("ml")
    with pytest.raises(ValueError, match="needs constellation= or kurtosis="):
        SNREstimator("m2m4")
    with pytest.raises(ValueError, match="dd needs"):
        SNREstimator("dd", kurtosis=1.0)
    assert np.isnan(SNREstimator(kurtosis=1.0).snr)


# ber counter


def ber_setup(n, delay, flips, chunk=999, **kw):
    desired = rng.integers(0, 2, n)
    received = np.r_[rng.integers(0, 2, delay), desired[: n - delay]].copy()
    pos = rng.choice(
        np.arange(2000 + delay, n), flips, replace=False
    )  # after skip+sync
    received[pos] ^= 1
    ber = BERCounter(skip=1000, sync=1000, **kw)
    FromArray(desired.astype(float)) >> ber["desired"]
    FromArray(received.astype(float)) >> ber["received"]
    d = Diagram(ber, fs=1.0)
    d.run(n=n, chunk=chunk)
    return ber, d


@pytest.mark.parametrize("chunk", [1, 999, 50_000])
def test_ber_exact_errors_and_delay(chunk):
    ber, _ = ber_setup(50_000, delay=17, flips=123, chunk=chunk)
    assert ber.locked and ber.delay == 17
    assert ber.errors == 123 and ber.bits == 50_000 - 2000
    lo, hi = ber.confidence()
    assert lo < ber.ber < hi
    assert "delay 17 bits" in ber.summary()


def test_ber_accepts_bool_and_soft_bits():
    n = 20_000
    desired = rng.integers(0, 2, n).astype(bool)
    received = desired.copy()
    received[5000:5010] ^= True
    ber = BERCounter(sync=500)
    FromArray(desired) >> ber["desired"]
    FromArray(received.astype(float) * 0.9 + 0.05) >> ber["received"]
    Diagram(ber, fs=1.0).run(n=n, chunk=3000)
    assert ber.delay == 0 and ber.errors == 10


def test_ber_skips_garbage_and_stops_after_errors():
    n = 200_000
    desired = rng.integers(0, 2, n)
    received = desired.copy()
    received[:5000] = rng.integers(0, 2, 5000)  # receiver not converged yet
    received[np.arange(10_000, n, 50)] ^= 1  # then 2 % errors
    ber = BERCounter(sync=1000, stop_after_errors=100, min_bits=1000)
    FromArray(desired.astype(float)) >> ber["desired"]
    FromArray(received.astype(float)) >> ber["received"]
    d = Diagram(ber, fs=1.0)
    d.run(n=n, chunk=2000)
    assert ber.locked and ber.lock_attempts >= 5 and ber.delay == 0
    assert d.stopped and ber.errors >= 100 and d.n < n
    assert abs(ber.ber - 0.02) < 0.01
    assert BERCounter().summary().endswith("not locked (0 attempts)")


def test_bpsk_ber_matches_theory():
    # theory: BPSK in AWGN has BER = erfc(sqrt(Eb/N0)) / 2; here Eb/N0 = 6 dB
    n, ebn0 = 300_000, 10**0.6
    bits = FromArray(rng.integers(0, 2, n).astype(float))
    mod, add = FuncChunk(lambda b: 2.0 * b - 1.0), Sum()
    det = FuncChunk(lambda v: (v > 0).astype(float))
    ber = BERCounter(sync=500)
    bits >> mod >> add["x0"]
    Noise(np.sqrt(1 / (2 * ebn0)), seed=3) >> add["x1"]
    add >> det >> ber["received"]
    bits >> ber["desired"]
    Diagram(ber, fs=1.0).run(n=n, chunk=8192)
    theory = 0.5 * erfc(np.sqrt(ebn0))
    lo, hi = ber.confidence(0.999)
    assert ber.delay == 0 and lo < theory < hi and abs(ber.ber / theory - 1) < 0.1


def test_qam_ambiguity_tables():
    amb = qam_ambiguities(QAM64)
    np.testing.assert_array_equal(amb[0], np.arange(64))
    w = rng.integers(0, 64, 100)
    np.testing.assert_allclose(QAM64[amb[1][w]], QAM64[w] * 1j, atol=1e-12)
    np.testing.assert_array_equal(amb[2][amb[2]], np.arange(64))  # 180 twice = 0


def test_spectrum_plot_before_the_first_frame_is_empty():
    # a live plot may draw before nfft samples have arrived
    sa = SpectrumAnalyzer(nfft=256, name="rx", fs=1000.0)
    sa(np.ones(100))
    ax = sa.plot()
    assert not ax.lines and "waiting for 256 samples" in ax.get_title()
    assert ax.get_xlabel() == "f / Hz"


# spectrogram


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("chunk", [1, 333, 10_000])
def test_spectrogram_equals_scipy_for_any_chunking(cplx, chunk):
    n = np.arange(10_000)
    x = rng.standard_normal(len(n)) + np.cos(0.3 * n)
    if cplx:
        x = x + 1j * rng.standard_normal(len(n))
    sg = Spectrogram(nfft=256, overlap=0.5, keep=10_000)
    feed(sg, x, fs=1000.0, chunk=chunk)
    f, t, S = sig.spectrogram(
        x, fs=1000.0, window="hann", nperseg=256, noverlap=128, detrend=False,
        return_onesided=not cplx,
    )  # fmt: skip
    if cplx:
        f, S = np.fft.fftshift(f), np.fft.fftshift(S, axes=0)
    np.testing.assert_allclose(sg.f, f)
    np.testing.assert_allclose(sg.t, t)
    np.testing.assert_allclose(sg.psd, S.T, rtol=1e-9)


def test_spectrogram_keeps_the_newest_frames_with_absolute_times():
    sg = Spectrogram(nfft=100, overlap=0.0, keep=5)
    feed(sg, rng.standard_normal(2000), fs=100.0, chunk=77)
    assert sg.frames == 20 and sg.psd.shape == (5, 51)
    np.testing.assert_allclose(sg.t, (np.arange(15, 20) * 100 + 50) / 100.0)


def test_spectrogram_follows_a_chirp():
    # theory: the peak of each frame sits at the chirp's frequency at that time
    fs, f0, k = 8000.0, 500.0, 1500.0  # f(t) = f0 + k t
    sg, src = (
        Spectrogram(nfft=256, overlap=0.75),
        Signal(lambda t: np.cos(2 * np.pi * (f0 * t + k * t**2 / 2))),
    )
    src >> sg
    Diagram(sg, fs=fs).run(2.0, chunk=500)
    peaks = sg.f[np.argmax(sg.psd, axis=1)]
    np.testing.assert_allclose(peaks, f0 + k * sg.t, atol=fs / 256)


def test_spectrogram_plot_and_probe():
    s = Sine(1e3)
    d = Diagram(s, fs=8000.0)
    sg = d.probe(s, sink=Spectrogram(nfft=128, name="rx"))
    ax = sg.plot()
    assert "waiting" in ax.get_title()  # live plots may ask before the first frame
    d.run(0.5)
    ax = sg.plot(ax=plt.figure().gca())
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("t / s", "f / Hz")
    assert len(ax.figure.axes) == 2  # with the colour bar
    assert sg.info() == "128-pt, 400 frames"


# sinks read from other threads


def test_reads_during_a_background_run_see_whole_updates():
    # a reader polls while a background run updates; every reading is whole
    src = Noise(1.0, seed=4, complex=True)
    cs = ConstellationScope(QPSK, keep=100)
    pm = PowerMeter()
    src >> cs
    src >> pm
    d = Diagram(cs, pm, fs=1e5)
    d.start(n=400_000, chunk=1000, realtime=False)
    seen = []
    while d.running:
        seen.append((len(cs.points), pm.power))
    d.wait(timeout=30)
    assert all(n <= 100 for n, _ in seen)
    assert all(np.isnan(p) or 0.5 < p < 2.0 for _, p in seen)
    assert abs(pm.power - 1.0) < 0.01
