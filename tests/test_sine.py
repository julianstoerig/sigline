import numpy as np
import pytest

from sigline import Diagram, Scope, Signal, Sine
from sigline.testing import assert_chunk_invariant

FS = 8000.0


def run(src, n=800, chunk=97):
    sc = Scope()
    src >> sc
    d = Diagram(sc, fs=FS)
    d.run(n=n, chunk=chunk)
    return d, sc


# unit


def test_amplitude_phase_and_first_sample():
    y = Sine(1000.0, amp=2.0, phase=np.pi / 2, fs=FS)(3)
    np.testing.assert_allclose(
        y, 2 * np.cos(np.pi / 2 + 2 * np.pi / 8 * np.arange(3)), atol=1e-12
    )


def test_complex_has_unit_magnitude_times_amp():
    z = Sine(500.0, amp=3.0, complex=True, fs=FS)(100)
    assert z.dtype == complex
    np.testing.assert_allclose(np.abs(z), 3.0)


def test_reset_restarts_phase():
    s = Sine(123.0, fs=FS)
    a = s(50)
    s(17)
    s.reset()
    np.testing.assert_array_equal(s(50), a)


def test_chunk_invariant():
    assert_chunk_invariant(lambda: Sine(0.05, complex=True), 500)
    assert_chunk_invariant(lambda: Sine(0.013, amp=0.5, phase=1.0), 500)


# integration: frequency change between runs is phase-continuous


def test_frequency_change_keeps_phase_continuous():
    s = Sine(1000.0)
    d, sc = run(s, n=1000, chunk=333)
    s.f = 2000.0
    d.run(n=1000, chunk=333)
    # largest step of cos(phi) is bounded by the phase increment: no jump
    assert np.max(np.abs(np.diff(sc.data))) <= 2 * np.pi * 2000 / FS * 1.0001
    k = np.arange(1000)
    ph1 = 2 * np.pi * 1000 * 1000 / FS
    np.testing.assert_allclose(
        sc.data[1000:], np.cos(ph1 + 2 * np.pi * 2000 * k / FS), atol=1e-9
    )


# comparison: Sine equals Signal(cos) and Signal(exp)


@pytest.mark.parametrize("cplx", [False, True])
def test_equals_signal_of_time(cplx):
    f, ph = 1234.5, 0.3
    fn = (
        (lambda t: np.exp(1j * (2 * np.pi * f * t + ph)))
        if cplx
        else (lambda t: np.cos(2 * np.pi * f * t + ph))
    )
    _, a = run(Sine(f, phase=ph, complex=cplx))
    _, b = run(Signal(fn))
    np.testing.assert_allclose(a.data, b.data, atol=1e-9)


# theory: an FFT of an integer number of periods has a single line at f


def test_spectrum_is_one_line_at_f():
    n, k = 800, 37  # 37 periods in 800 samples
    _, sc = run(Sine(k * FS / n, complex=True), n=n)
    X = np.abs(np.fft.fft(sc.data)) / n
    assert np.argmax(X) == k
    np.testing.assert_allclose(X[k], 1.0, atol=1e-9)
    assert np.max(np.delete(X, k)) < 1e-9
