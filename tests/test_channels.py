import math

import numpy as np
import pytest
import scipy.signal as sig
from scipy.special import erfc

import sigline as sl
from sigline.channels import BOLTZMANN
from sigline.testing import assert_chunk_invariant

N = 200_000


def run(part, x, fs, chunk=8192):
    sc = sl.Scope()
    sl.FromArray(x) >> part >> sc
    sl.Diagram(sc, fs=fs).run(n=len(x), chunk=chunk)
    return sc.data


def power(y):
    return np.mean(np.abs(y) ** 2)


def Q(x):
    return 0.5 * erfc(x / math.sqrt(2))


# awgn: unit


def test_awgn_members_and_arguments():
    ch = sl.awgn(snr_db=10.0, signal_power=2.0)
    assert ch.type == "awgn" and ch.inputs == ("x",) and ch.outputs == ("y",)
    assert isinstance(ch.noise, sl.Noise) and isinstance(ch.add, sl.Sum)
    assert ch.noise.std == pytest.approx(math.sqrt(0.2))
    assert ch.info() == "P = 0.2"
    with pytest.raises(ValueError, match="exactly one"):
        sl.awgn(snr_db=10.0, power=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        sl.awgn()
    with pytest.raises(ValueError, match="need fs"):
        sl.awgn(snr_db=10.0, bandwidth=3e3)
    with pytest.raises(ValueError, match="need fs"):
        sl.awgn(temperature=290.0)


def test_awgn_refuses_noise_of_the_other_kind_or_width():
    with pytest.raises(TypeError, match="add.*all real or all complex"):
        run(sl.awgn(power=1.0), np.ones(10, complex), 1.0)
    with pytest.raises(TypeError, match="all real or all complex"):
        run(sl.awgn(power=1.0, complex=True), np.ones(10), 1.0)
    with pytest.raises(ValueError, match="input widths differ"):
        run(sl.awgn(power=1.0), np.ones((10, 2)), 1.0)
    assert run(sl.awgn(power=1.0, width=2), np.ones((10, 2)), 1.0).shape == (10, 2)


@pytest.mark.parametrize("cplx", [False, True])
def test_awgn_chunk_invariant(cplx):
    x = np.ones(500, complex if cplx else float)
    assert_chunk_invariant(lambda: sl.awgn(power=0.3, complex=cplx, seed=4), x)


# integration: a noisy tone measured by the power meter


def test_noisy_tone_in_a_diagram():
    tone, meter = sl.Sine(1e3, complex=True), sl.PowerMeter()
    tone >> sl.awgn(snr_db=3.0, complex=True) >> meter
    sl.Diagram(meter, fs=48e3).run(n=N, chunk=4096)
    pn = 10**-0.3
    # |s + n|^2 has mean 1 + pn and variance 2 pn + pn^2
    tol = 5 * math.sqrt((2 * pn + pn**2) / N) / (1 + pn)
    assert abs(meter.power / (1 + pn) - 1) < tol


# comparison: the same as Noise and Sum wired by hand


def test_awgn_equals_noise_and_sum_by_hand():
    x = np.random.default_rng(1).standard_normal(3000)
    add, sc = sl.Sum(), sl.Scope()
    sl.FromArray(x) >> add["x0"]
    sl.Noise(0.5, seed=7) >> add["x1"]
    add >> sc
    sl.Diagram(sc, fs=1.0).run(n=3000, chunk=100)
    np.testing.assert_array_equal(run(sl.awgn(power=0.25, seed=7), x, 1.0), sc.data)


# theory


@pytest.mark.parametrize("cplx", [False, True])
def test_awgn_power_from_snr(cplx):
    x = np.zeros(N, complex if cplx else float)
    y = run(sl.awgn(snr_db=10.0, signal_power=2.0, complex=cplx), x, 1.0)
    assert np.iscomplexobj(y) == cplx
    assert abs(power(y) / 0.2 - 1) < 5 * math.sqrt(2 / N)


def test_awgn_snr_in_a_bandwidth():
    # snr in 3 kHz at fs = 48 kHz: 16 times the noise over the whole band
    ch = sl.awgn(snr_db=10.0, bandwidth=3e3, complex=True, fs=48e3)
    y = run(ch, np.zeros(N, complex), 48e3)
    assert abs(power(y) / 1.6 - 1) < 5 * math.sqrt(1 / N)


@pytest.mark.parametrize("cplx", [False, True])
def test_thermal_noise_is_k_t_b(cplx):
    fs = 1e6
    x = np.zeros(N, complex if cplx else float)
    y = run(sl.awgn(temperature=290.0, complex=cplx, fs=fs), x, fs)
    ktb = BOLTZMANN * 290.0 * (fs if cplx else fs / 2)
    assert abs(power(y) / ktb - 1) < 5 * math.sqrt(2 / N)


def test_bpsk_bit_error_rate_matches_theory():
    bits = np.random.default_rng(1).integers(0, 2, 400_000)
    ebn0 = 6.0
    # real bpsk: snr = 2 Eb/N0
    y = run(sl.awgn(snr_db=ebn0 + 10 * math.log10(2)), 2.0 * bits - 1, 1.0)
    ber = np.mean((y > 0) != bits.astype(bool))
    theory = Q(math.sqrt(2 * 10 ** (ebn0 / 10)))
    # about 970 errors: 5 standard errors are 16 %
    assert abs(ber / theory - 1) < 0.16


# colored noise


def test_colored_noise_members_and_colors():
    ch = sl.colored_noise(2.0, "brown")
    assert isinstance(ch.shape, sl.IIR) and ch.info() == "brown, P = 2"
    with pytest.raises(ValueError, match="color must be"):
        sl.colored_noise(1.0, "green")


@pytest.mark.parametrize("cplx", [False, True])
def test_colored_noise_chunk_invariant(cplx):
    x = np.zeros(500, complex if cplx else float)
    assert_chunk_invariant(
        lambda: sl.colored_noise(1.0, complex=cplx, width=2), x[:, None] * [1, 1]
    )


def test_colored_noise_equals_filtered_awgn_by_hand():
    # the same noise stream, shaped by lfilter in one go
    b, a = sl.channels.COLORS["pink"]
    y = run(sl.colored_noise(1.0, "pink", seed=3), np.zeros(5000), 1.0, chunk=333)
    w = run(sl.awgn(power=1.0, seed=3), np.zeros(5000), 1.0)
    want = sig.lfilter(b, a, w) / math.sqrt(sl.channels._noise_gain(b, a))
    np.testing.assert_allclose(y, want, atol=1e-12)


@pytest.mark.parametrize(
    ("color", "slope", "band"),
    [
        ("pink", -1.0, (1e-3, 1e-1)),
        ("brown", -2.0, (3e-3, 5e-2)),
        ("violet", 2.0, (1e-3, 2e-2)),
    ],
)
def test_colored_noise_slope_and_power(color, slope, band):
    y = run(sl.colored_noise(1.0, color, seed=1), np.zeros(2**21), 1.0, chunk=2**16)
    assert abs(power(y[5000:]) - 1) < 0.1
    f, P = sig.welch(y, fs=1.0, nperseg=2**16)
    m = (f > band[0]) & (f < band[1])
    k = np.polyfit(np.log10(f[m]), np.log10(P[m]), 1)[0]
    assert abs(k - slope) < 0.15, k


# phase noise


def phase_run(part, n, fs, chunk=8192):
    sy, sp = sl.Scope(), sl.Scope()
    sl.FromArray(np.ones(n, complex)) >> part
    part["y"] >> sy
    part["phi"] >> sp
    sl.Diagram(sy, fs=fs).run(n=n, chunk=chunk)
    return sy.data, sp.data


def test_phase_noise_members_and_arguments():
    pn = sl.phase_noise(100.0, fs=1e5)
    assert pn.outputs == ("y", "phi") and isinstance(pn.phase, sl.IIR)
    assert pn.info() == "Δν 100 Hz"
    level = sl.phase_noise(
        dbc_hz=10 * math.log10(100 / (2 * math.pi * 1e6)), offset=1e3, fs=1e5
    )
    assert level.info() == "Δν 100 Hz"
    with pytest.raises(ValueError, match="linewidth"):
        sl.phase_noise(dbc_hz=-80.0, fs=1e5)
    real, sc = sl.phase_noise(1.0, fs=1.0), sl.Scope()
    sl.FromArray(np.ones(10)) >> real
    real["y"] >> sc
    with pytest.raises(TypeError, match="all real or all complex"):
        sl.Diagram(sc, fs=1.0).run(n=10)


def test_phase_noise_chunk_invariant():
    assert_chunk_invariant(
        lambda: sl.phase_noise(50.0, seed=2, fs=1.0), np.ones(400, complex)
    )


def test_phase_noise_equals_cumulated_noise():
    fs, lw = 1e4, 30.0
    y, phi = phase_run(sl.phase_noise(lw, seed=5, fs=fs), 5000, fs, chunk=777)
    steps = math.sqrt(2 * math.pi * lw / fs) * np.random.default_rng(5).standard_normal(
        5000
    )
    np.testing.assert_allclose(phi, np.cumsum(steps), atol=1e-12)
    np.testing.assert_allclose(y, np.exp(1j * phi), atol=1e-12)


def test_phase_noise_step_variance_and_level():
    fs, lw, n = 100e3, 100.0, 2**20
    y, phi = phase_run(sl.phase_noise(lw, seed=3, fs=fs), n, fs)
    assert abs(np.var(np.diff(phi)) / (2 * np.pi * lw / fs) - 1) < 5 * math.sqrt(2 / n)
    f, P = sig.welch(y, fs=fs, nperseg=2**14, return_onesided=False, detrend=False)
    for off in (1e3, 3e3):
        level = 10 * np.log10(np.mean(P[np.abs(np.abs(f) - off) < 50]))
        # lorentzian line of full width lw, normalised to unit power
        theory = 10 * np.log10(lw / (2 * np.pi) / (off**2 + (lw / 2) ** 2))
        assert abs(level - theory) < 1.0, (off, level, theory)


# impulsive noise


def test_impulsive_noise_members():
    ch = sl.impulsive_noise(4.0, 50.0, duration=1e-3)
    assert isinstance(ch.pulses, sl.RandomPulses) and isinstance(ch.gate, sl.Product)
    assert ch.info() == "50/s, 1 ms, P = 4"
    assert ch.pulses.seed == ch.noise.seed + 1


@pytest.mark.parametrize("cplx", [False, True])
def test_impulsive_noise_chunk_invariant(cplx):
    x = np.zeros(600, complex if cplx else float)
    assert_chunk_invariant(
        lambda: sl.impulsive_noise(2.0, p=0.02, duration=5.0, complex=cplx), x
    )


def test_impulsive_noise_is_noise_times_pulses():
    y = run(
        sl.impulsive_noise(4.0, p=0.01, duration=20.0, seed=3),
        np.zeros(5000),
        1.0,
        chunk=99,
    )
    noise = run(sl.awgn(power=4.0, seed=3), np.zeros(5000), 1.0)
    pulses = sl.RandomPulses(p=0.01, duration=20.0, seed=4, fs=1.0)(5000)
    np.testing.assert_array_equal(y, noise * pulses)


def test_impulsive_noise_occupancy_and_power():
    fs, rate, n = 1e5, 50.0, 100  # 50 pulses/s of 100 samples
    y = run(
        sl.impulsive_noise(4.0, rate, duration=n / fs, seed=2), np.zeros(2_000_000), fs
    )
    on = y != 0
    assert abs(on.mean() / (1 - (1 - rate / fs) ** n) - 1) < 0.1
    assert abs(power(y[on]) / 4.0 - 1) < 5 * math.sqrt(2 / on.sum())


# burst noise


def burst_run(part, n, chunk=8192):
    sy, sb = sl.Scope(), sl.Scope()
    sl.FromArray(np.zeros(n)) >> part
    part["y"] >> sy
    part["bad"] >> sb
    sl.Diagram(sy, fs=1.0).run(n=n, chunk=chunk)
    return sy.data, sb.data


def test_burst_noise_members():
    ch = sl.burst_noise(1e-3, 2e-2, power_bad=1.0)
    assert ch.outputs == ("y", "bad") and isinstance(ch.state, sl.MarkovChain)
    assert ch.info() == "bursts ~50 samples"
    np.testing.assert_allclose(ch.level.params["points"], [0.0, 1.0])


def test_burst_noise_chunk_invariant():
    x = np.zeros(800)
    assert_chunk_invariant(lambda: sl.burst_noise(0.01, 0.1, 1.0, 0.01, seed=1), x)


def test_burst_noise_equals_its_parts():
    y, bad = burst_run(sl.burst_noise(0.01, 0.1, 4.0, 0.25, seed=2), 5000, chunk=123)
    state = sl.MarkovChain([[0.99, 0.01], [0.1, 0.9]], seed=3, fs=1.0)(5000)
    np.testing.assert_array_equal(bad, state)
    w = np.random.default_rng(2).standard_normal(5000)
    np.testing.assert_allclose(y, np.where(state == 1, 2.0, 0.5) * w, atol=1e-15)


def test_burst_noise_power_per_state():
    p_gb, p_bg = 1e-3, 2e-2
    y, bad = burst_run(sl.burst_noise(p_gb, p_bg, 9.0, 0.01, seed=4), 2_000_000)
    on = bad == 1
    assert abs(on.mean() / (p_gb / (p_gb + p_bg)) - 1) < 0.1
    assert abs(power(y[on]) / 9.0 - 1) < 5 * math.sqrt(2 / on.sum())
    assert abs(power(y[~on]) / 0.01 - 1) < 5 * math.sqrt(2 / (~on).sum())


# interferers


@pytest.mark.parametrize("cplx", [False, True])
def test_tone_interferer_power_and_frequency(cplx):
    fs, n = 1e4, 2**16
    ch = sl.tone_interferer(1234.0, 0.5, complex=cplx)
    assert ch.info().startswith("CW 1.234 kHz")
    y = run(ch, np.zeros(n, complex if cplx else float), fs)
    assert abs(power(y) - 0.5) < 1e-3
    f, P = sig.welch(y, fs=fs, nperseg=4096, return_onesided=False)
    assert abs(f[np.argmax(P)] - 1234.0) < fs / 4096


def test_tone_interferer_drifts_like_a_chirp():
    fs = 1e4
    y = run(
        sl.tone_interferer(100.0, 1.0, drift=50.0, complex=True),
        np.zeros(5000, complex),
        fs,
    )
    t = np.arange(5000) / fs
    np.testing.assert_allclose(y, np.exp(2j * np.pi * (100 * t + 25 * t**2)), atol=1e-8)


def test_chirp_interferer_stays_in_its_band():
    fs = 1e4
    y = run(
        sl.chirp_interferer(-2e3, 3e3, 0.05, 1.0, complex=True),
        np.zeros(2**17, complex),
        fs,
    )
    assert abs(power(y) - 1.0) < 1e-9
    f, P = sig.welch(y, fs=fs, nperseg=4096, return_onesided=False)
    assert np.sum(P[(f > -2.05e3) & (f < 3.05e3)]) / np.sum(P) > 0.97


def test_pulsed_interferer_duty_cycle_and_peak_power():
    fs = 1e4
    ch = sl.pulsed_interferer(500.0, 10.0, width=1e-3, period=20e-3, complex=True)
    assert ch.info() == "500 Hz, 1 ms / 20 ms"
    y = run(ch, np.zeros(200_000, complex), fs)
    on = np.abs(y) > 0
    assert on.mean() == pytest.approx(0.05)
    np.testing.assert_allclose(np.abs(y[on]) ** 2, 10.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: sl.tone_interferer(0.01, 1.0, drift=1e-5, complex=True),
        lambda: sl.chirp_interferer(-0.1, 0.2, 77.0, 1.0, complex=True),
        lambda: sl.pulsed_interferer(0.05, 2.0, 5.0, 31.0, 3.0, complex=True),
    ],
)
def test_interferers_chunk_invariant(make):
    assert_chunk_invariant(make, np.zeros(600, complex))


def test_pulsed_interferer_equals_sine_times_pulses():
    fs, n = 1e4, 3000
    y = run(
        sl.pulsed_interferer(700.0, 2.0, 2e-3, 9e-3, 1e-3), np.zeros(n), fs, chunk=101
    )
    want = sl.Sine(700.0, amp=2.0, fs=fs)(n) * sl.Pulses(2e-3, 9e-3, 1e-3, fs=fs)(n)
    np.testing.assert_allclose(y, want, atol=1e-12)


# modulated interferer


def test_modulated_interferer_members():
    ch = sl.modulated_interferer(6e3, 2.0, 8, fs=48e3)
    assert ch.info() == "4 points, 8 sps at 6 kHz"
    assert isinstance(ch.bits, sl.RandomBits) and ch.bits.fs == 12e3
    assert ch.shape.type == "pulse_shaper" and ch.map.type == "mapper"
    assert ch.level.g == pytest.approx(2.0 * math.sqrt(8))  # real: sqrt(2 P sps)


def test_modulated_interferer_chunk_invariant():
    def make():
        return sl.modulated_interferer(0.1, 1.0, 4, sl.qam(16), complex=True, fs=1.0)

    # bits run at 4 / 4 = 1 per sample, symbols at 1/4: chunks of 4
    assert_chunk_invariant(make, np.zeros(800, complex), chunks=(4, 28, 64))


def test_modulated_interferer_equals_a_transmitter():
    fs, sps, n = 1.0, 4, 4000
    ch = sl.modulated_interferer(0.1, 3.0, sps, sl.qam(16), complex=True, seed=5, fs=fs)
    y = run(ch, np.zeros(n, complex), fs, chunk=400)
    bits, sc = sl.RandomBits(seed=5, fs=fs), sl.Scope()
    tx = bits >> sl.mapper(sl.qam(16)) >> sl.pulse_shaper(sps) >> sl.mixer(0.1, "exp")
    tx >> sl.Gain(math.sqrt(3.0 * sps)) >> sc
    sl.Diagram(sc, fs=fs).run(n=n, chunk=400)
    np.testing.assert_allclose(y, sc.data, atol=1e-12)


@pytest.mark.parametrize("cplx", [False, True])
def test_modulated_interferer_power_and_band(cplx):
    fs, sps, beta, f0 = 48e3, 10, 0.35, 6e3
    ch = sl.modulated_interferer(f0, 2.0, sps, beta=beta, complex=cplx, fs=fs)
    y = run(ch, np.zeros(520_000, complex if cplx else float), fs, chunk=8000)
    assert abs(power(y[1000:]) / 2.0 - 1) < 0.05
    f, P = sig.welch(y, fs=fs, nperseg=8192, return_onesided=False)
    o = np.argsort(f)
    f, P = f[o], P[o]
    if not cplx:  # the upper half of a real signal's symmetric spectrum
        f, P = f[f >= 0], P[f >= 0]
    c = np.cumsum(P) / np.sum(P)
    bw = f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)]
    assert abs(np.sum(f * P) / np.sum(P) - f0) < 50
    assert abs(bw / ((1 + beta) * fs / sps) - 1) < 0.15, bw


# integration: noise after a rate change takes the rate of the signal


def test_awgn_after_pulse_shaping_runs_at_the_sample_rate():
    sc = sl.Scope()
    ch = sl.awgn(power=0.5, complex=True)
    sl.RandomBits() >> sl.mapper(sl.psk(4)) >> sl.pulse_shaper(8) >> ch >> sc
    d = sl.Diagram(sc, fs=2.0)  # bits at 2 Hz, symbols at 1 Hz, samples at 8 Hz
    assert ch.noise.fs == 8.0
    d.run(n=200, chunk=20)
    assert len(sc.data) == 800


# each channel subsystem is typed after its function, not a helper


@pytest.mark.parametrize(
    "make",
    [
        lambda: sl.awgn(power=1.0),
        lambda: sl.colored_noise(1.0),
        lambda: sl.phase_noise(1.0, fs=8.0),
        lambda: sl.impulsive_noise(1.0, 1.0),
        lambda: sl.burst_noise(0.1, 0.1, 1.0),
        lambda: sl.tone_interferer(1.0, 1.0),
        lambda: sl.chirp_interferer(0.0, 1.0, 1.0, 1.0),
        lambda: sl.pulsed_interferer(1.0, 1.0, 1.0, 2.0),
        lambda: sl.modulated_interferer(1.0, 1.0, 4, fs=8.0),
    ],
)
def test_channel_types_name_their_function(make):
    part = make()
    assert part.type == make.__code__.co_names[-1]  # the function called
