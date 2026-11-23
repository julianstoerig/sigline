import numpy as np
import pytest

from sigline import Chirp, Diagram, Scope, Sine
from sigline.testing import assert_chunk_invariant


def record(src, n, fs=1.0, chunk=1000):
    sc = Scope()
    src >> sc
    Diagram(sc, fs=fs).run(n=n, chunk=chunk)
    return sc.data


# unit


def test_info_and_amplitude():
    assert Chirp(1e3, 2e3, 0.5, complex=True).info() == "1 kHz, 2 kHz/s per 500 ms cplx"
    assert Chirp(50.0).info() == "50 Hz"
    np.testing.assert_allclose(
        np.abs(Chirp(3.0, 1.0, amp=2.0, complex=True, fs=10.0)(50)), 2.0
    )


@pytest.mark.parametrize("period", [None, 37.0])
def test_chunk_invariant(period):
    assert_chunk_invariant(lambda: Chirp(0.01, 1e-4, period, complex=True), 600)


def test_rate_change_between_runs_keeps_the_phase_continuous():
    c = Chirp(0.05, complex=True, fs=1.0)
    a = c(100)
    c.rate_hz = 1e-3
    b = c(100)
    # no jump: the step into the new run is the old frequency, the next one
    # the new sweep at the middle of the interval
    assert np.angle(b[0] / a[-1]) == pytest.approx(2 * np.pi * 0.05)
    assert np.angle(b[1] / b[0]) == pytest.approx(2 * np.pi * (0.05 + 1e-3 * 100.5))


# comparison: without a sweep it is a sine


def test_without_sweep_equals_sine():
    np.testing.assert_allclose(
        record(Chirp(0.013, phase=0.4), 3000, chunk=77),
        record(Sine(0.013, phase=0.4), 3000, chunk=77),
        atol=1e-9,
    )


# theory: a linear sweep has phase 2 pi (f0 t + rate t^2 / 2)


def test_linear_sweep_equals_closed_form():
    fs, f0, rate = 1e4, 100.0, 2e3
    z = record(Chirp(f0, rate, complex=True), 20_000, fs=fs, chunk=999)
    t = np.arange(20_000) / fs
    np.testing.assert_allclose(
        z, np.exp(2j * np.pi * (f0 * t + rate * t**2 / 2)), atol=1e-8
    )


def test_sawtooth_restarts_each_period():
    fs, period = 1e4, 0.05
    z = record(Chirp(-2e3, 1e5, period, complex=True), 4000, fs=fs)
    f = np.angle(z[1:] * np.conj(z[:-1])) * fs / (2 * np.pi)  # mid-interval frequency
    n = np.arange(1, 4000) - 0.5
    want = -2e3 + 1e5 * np.mod(n / fs, period)
    np.testing.assert_allclose(f, want, atol=1e-6)
