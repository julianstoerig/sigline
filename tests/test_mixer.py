import numpy as np
import pytest
import scipy.signal as sig

from sigline import FIR, Diagram, FromArray, Func, Mixer, Product, Scope, Sine
from sigline.testing import assert_chunk_invariant

FS = 1e6
X = np.random.default_rng(3).standard_normal(1000) + 0j


# unit


@pytest.mark.parametrize(
    ("lo", "fn"),
    [
        ("cos", np.cos),
        ("sin", np.sin),
        ("exp", lambda p: np.exp(1j * p)),
        ("exp-", lambda p: np.exp(-1j * p)),
    ],
)
def test_lo_kinds(lo, fn):
    ph = 2 * np.pi * 0.01 * np.arange(1000) + 0.3
    np.testing.assert_allclose(
        Mixer(0.01, lo, phase=0.3, fs=1.0)(X), X * fn(ph), atol=1e-12
    )


def test_vector_signals_share_the_lo():
    v = np.ones((5, 2))
    y = Mixer(0.1, fs=1.0)(v)
    np.testing.assert_allclose(y[:, 0], y[:, 1])


def test_frequency_change_is_phase_continuous():
    m, sc = Mixer(1000.0), Scope()
    FromArray(np.ones(2000)) >> m >> sc
    d = Diagram(sc, fs=48e3)
    d.run(n=1000, chunk=333)
    m.f = 2000.0
    d.run(n=1000, chunk=333)
    assert np.max(np.abs(np.diff(sc.data))) <= 2 * np.pi * 2000 / 48e3 * 1.0001


@pytest.mark.parametrize("lo", ["cos", "exp-"])
def test_chunk_invariant(lo):
    assert_chunk_invariant(lambda: Mixer(0.11, lo), X)


# comparison: Mixer(f) equals Product with Sine(f)


def test_equals_product_with_sine_source():
    a, b = Scope(), Scope()
    FromArray(X.real) >> Mixer(0.07) >> a
    p = Product()
    FromArray(X.real) >> p["x0"]
    Sine(0.07) >> p["x1"]
    p >> b
    Diagram(a, fs=1.0).run(n=1000, chunk=77)
    Diagram(b, fs=1.0).run(n=1000, chunk=77)
    np.testing.assert_allclose(a.data, b.data, atol=1e-12)


# theory: i/q down-conversion of 110 kHz with a 100 kHz LO gives a 10 kHz tone


def test_iq_downconversion_moves_tone_to_difference_frequency():
    x = Sine(110e3)
    i_mix, q_mix = Mixer(100e3, "cos"), Mixer(100e3, "sin")
    taps = sig.firwin(129, 30e3, fs=FS)
    iq, sc = Func(lambda i, q: 2 * (i - 1j * q)), Scope()
    x >> i_mix >> FIR(taps) >> iq["i"]
    x >> q_mix >> FIR(taps) >> iq["q"]
    iq >> sc
    Diagram(sc, fs=FS).run(n=20_000, chunk=4096)
    z = sc.data[200:]
    spec = np.abs(np.fft.fft(z * np.hanning(len(z))))
    f = np.fft.fftfreq(len(z), 1 / FS)
    assert abs(f[np.argmax(spec)] - 10e3) < FS / len(z)
    # magnitude = filter gain at 10 kHz, +- the image at 210 kHz leaking through
    _, H = sig.freqz(taps, 1, worN=[10e3, 210e3], fs=FS)
    np.testing.assert_allclose(np.abs(z), np.abs(H[0]), atol=np.abs(H[1]) + 1e-9)
