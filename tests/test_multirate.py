import numpy as np
import pytest
import scipy.signal as sig

import sigline._numba as nb
from sigline import (
    FIR,
    Decimate,
    Delay,
    Diagram,
    Downsample,
    FromArray,
    Gain,
    Interpolate,
    Repeat,
    Scope,
    Sine,
    Sum,
    Upsample,
)
from sigline.testing import assert_chunk_invariant

rng = np.random.default_rng(9)
X = rng.standard_normal(600) + 1j * rng.standard_normal(600)


def run(blocks, x, fs=1.0, chunk=97):
    src, sc = FromArray(x, fs=fs), Scope()
    node = src
    for b in blocks:
        node = node >> b
    node >> sc
    d = Diagram(sc, fs=fs)
    n = len(x) // d.granularity * d.granularity
    d.run(n=n, chunk=chunk)
    return sc.data, d


# upsample


def test_upsample_zero_stuffing_phase_and_rates():
    np.testing.assert_array_equal(
        Upsample(3, phase=1)(np.array([1.0, 2.0])), [0, 1, 0, 0, 2, 0]
    )
    up = Upsample(4)
    y, d = run([up], X, fs=1000.0)
    np.testing.assert_array_equal(y[::4], X)
    assert d.rates()[up.name] == (1000.0, 4000.0)


def test_upsample_chunk_invariant():
    assert_chunk_invariant(lambda: Upsample(3, phase=2), X)


def test_upsample_then_fir_equals_upfirdn():
    # comparison with scipy's one-shot upfirdn
    h = sig.firwin(31, 0.25)
    y, _ = run([Upsample(4), FIR(h)], X, chunk=50)
    np.testing.assert_allclose(y, sig.upfirdn(h, X, up=4)[: 4 * len(X)], atol=1e-12)


def test_upsampled_sine_spectrum_has_images():
    # theory: zero-stuffing by L repeats the spectrum L times: images at
    # k fs_in +- f0 of the new rate, each with 1/L of the amplitude
    fs, f0, L, n = 1000.0, 125.0, 4, 1000
    s, up, sc = Sine(f0), Upsample(L), Scope()
    s >> up >> sc
    Diagram(sc, fs=fs).run(n=n)
    Y = np.abs(np.fft.rfft(sc.data)) / (L * n / 2)
    f = np.fft.rfftfreq(L * n, 1 / (L * fs))
    for k in (f0, fs - f0, fs + f0):
        assert Y[np.argmin(np.abs(f - k))] == pytest.approx(1 / L, rel=1e-9)


# repeat


def test_repeat_holds_samples():
    np.testing.assert_array_equal(Repeat(3)(np.array([1.0, 2.0])), [1, 1, 1, 2, 2, 2])
    v = Repeat(2)(np.arange(4.0).reshape(2, 2))
    np.testing.assert_array_equal(v, [[0, 1], [0, 1], [2, 3], [2, 3]])


def test_repeat_equals_upsample_then_hold_filter():
    # comparison: a zero-order hold is zero-stuffing followed by L ones
    a, _ = run([Repeat(4)], X)
    b, _ = run([Upsample(4), FIR(np.ones(4))], X)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_repeat_chunk_invariant():
    assert_chunk_invariant(lambda: Repeat(5), X)


def test_hold_response_is_a_sinc():
    # theory: |H(f)| of an L-sample hold is |sin(pi f L / fs) / sin(pi f / fs)|
    L, fs = 4, 1.0
    f, H = FIR(np.ones(L), fs=fs).freqz(n=256)
    x = np.pi * f[1:] / fs
    np.testing.assert_allclose(
        np.abs(H[1:]), np.abs(np.sin(L * x) / np.sin(x)), atol=1e-12
    )


# downsample


@pytest.mark.parametrize("chunk", [1, 7, 600])
def test_up_then_down_is_identity_for_any_chunking(chunk):
    y, d = run([Upsample(3), Downsample(3)], X, chunk=chunk)
    np.testing.assert_array_equal(y, X)
    assert d.granularity == 1


def test_downsample_phase_and_granularity():
    np.testing.assert_array_equal(Downsample(3, phase=1)(np.arange(9.0)), [1, 4, 7])
    y, d = run([Downsample(4, phase=2)], np.arange(40.0), chunk=6)
    assert d.granularity == 4
    np.testing.assert_array_equal(y, np.arange(2, 40, 4))


def test_downsample_chunk_invariant():
    assert_chunk_invariant(lambda: Downsample(4, phase=3), X)


def test_aliasing_of_an_unfiltered_downsample():
    # theory: keeping every M-th sample aliases f to |f - k fs/M|: a 700 Hz
    # tone at 1 kHz, kept every 2nd sample (500 Hz), shows up at 200 Hz
    fs, n = 1000.0, 2000
    s, dn, sc = Sine(700.0), Downsample(2), Scope()
    s >> dn >> sc
    Diagram(sc, fs=fs).run(n=n)
    f = np.fft.rfftfreq(n // 2, 2 / fs)
    assert f[np.argmax(np.abs(np.fft.rfft(sc.data)))] == pytest.approx(200.0)


def test_feedback_at_a_lower_rate():
    # integration: a loop through a delay at half the source rate
    src, dn, add, z, sc = (
        FromArray(np.ones(64), fs=2.0),
        Downsample(2),
        Sum(),
        Delay(4),
        Scope(),
    )
    src >> dn >> add["x0"]
    add >> z >> Gain(0.5) >> add["x1"]
    add >> sc
    Diagram(sc, fs=2.0).run(n=64, chunk=8)
    # theory: y[k] = 1 + 0.5 y[k - 4] -> 1, 1.5, 1.75, ... per group of 4
    k = np.arange(32)
    np.testing.assert_allclose(sc.data, 2 - 0.5 ** (k // 4), atol=1e-12)


# interpolate (polyphase)


@pytest.mark.parametrize("chunk", [1, 61, 600])
def test_interpolate_equals_upfirdn(chunk):
    ip = Interpolate(4)
    y, _ = run([ip], X, chunk=chunk)
    np.testing.assert_allclose(
        y, sig.upfirdn(ip.taps, X, up=4)[: 4 * len(X)], atol=1e-12
    )


def test_numba_path_equals_reference_path(monkeypatch):
    # comparison: the compiled kernel and the lfilter reference agree
    a, _ = run([Interpolate(3)], X, chunk=37)
    monkeypatch.setattr(nb, "HAVE_NUMBA", False)
    b, _ = run([Interpolate(3)], X, chunk=37)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_interpolate_vector_signals_and_chunk_invariance():
    v = np.stack([X.real, X.imag], axis=1)
    y, _ = run([Interpolate(2)], v, chunk=50)
    np.testing.assert_allclose(
        y[:, 1], run([Interpolate(2)], X.imag, chunk=50)[0], atol=1e-12
    )
    assert_chunk_invariant(lambda: Interpolate(3), X)


def test_interpolated_tone_keeps_amplitude_and_drops_images():
    # theory: the default filter passes the tone with gain 1 and suppresses
    # the images at fs_in -+ f0 by more than 60 dB
    fs, f0, L, n = 1000.0, 125.0, 4, 4000
    s, ip, sc = Sine(f0), Interpolate(L), Scope()
    s >> ip >> sc
    Diagram(sc, fs=fs).run(n=n)
    y = sc.data[-3200:]  # after the transient; 100 periods: tones on bins
    Y = np.abs(np.fft.rfft(y)) / (len(y) / 2)
    f = np.fft.rfftfreq(len(y), 1 / (L * fs))
    level = lambda k: Y[np.argmin(np.abs(f - k))]  # noqa: E731
    assert level(f0) == pytest.approx(1.0, abs=0.01)
    assert 20 * np.log10(max(level(fs - f0), level(fs + f0))) < -60


# decimate


@pytest.mark.parametrize("chunk", [4, 64, 600])
def test_decimate_equals_lfilter_then_keep(chunk):
    dec = Decimate(4)
    y, _ = run([dec], X, chunk=chunk)
    np.testing.assert_allclose(y, sig.lfilter(dec.taps, 1, X)[::4], atol=1e-12)


def test_decimate_paths_agree_and_chunk_invariant(monkeypatch):
    a, _ = run([Decimate(3)], X, chunk=33)
    monkeypatch.setattr(nb, "HAVE_NUMBA", False)
    b, _ = run([Decimate(3)], X, chunk=33)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert_chunk_invariant(lambda: Decimate(3), X)


def test_decimate_removes_what_would_alias():
    # theory: unlike Downsample, the anti-alias filter removes a 700 Hz tone
    # before keeping every 2nd sample of 1 kHz, instead of folding it to 200 Hz
    fs, n = 1000.0, 4000
    for blk, expect in ((Downsample(2), 1.0), (Decimate(2), 0.0)):
        s, sc = Sine(700.0), Scope()
        s >> blk >> sc
        Diagram(sc, fs=fs).run(n=n)
        y = sc.data[200:]
        assert np.sqrt(2 * np.mean(y**2)) == pytest.approx(expect, abs=0.01)
