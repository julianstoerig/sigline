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
