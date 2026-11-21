import numpy as np
import pytest
from scipy.special import jv

from sigline import (
    AMDemod,
    AMMod,
    Diagram,
    FMDemod,
    FMMod,
    FromArray,
    PMDemod,
    PMMod,
    Scope,
    Sine,
)
from sigline.testing import assert_chunk_invariant

FS = 48_000.0
rng = np.random.default_rng(3)
MSG = np.clip(rng.standard_normal(2000) * 0.3, -1, 1)  # a message within [-1, 1]


def through(blocks, x, fs=FS, chunk=97):
    src, sc = FromArray(x), Scope()
    node = src
    for b in blocks:
        node = node >> b
    node >> sc
    Diagram(sc, fs=fs).run(n=len(x), chunk=chunk)
    return sc.data


# am


def test_am_mod_values():
    np.testing.assert_allclose(AMMod(0.5)(MSG), 1 + 0.5 * MSG)
    z = AMMod(0.8, carrier=False)(MSG)
    assert z.dtype == complex
    np.testing.assert_allclose(z, 0.8 * MSG)


def test_am_demod_validates():
    with pytest.raises(ValueError, match="needs the carrier"):
        AMDemod(carrier=False)
    with pytest.raises(ValueError, match="method"):
        AMDemod(method="square-law")
    with pytest.raises(ValueError, match="channel_width"):
        through([AMDemod(channel_width=1e6)], np.ones(10) + 0j)


@pytest.mark.parametrize(
    ("mod", "demod"),
    [
        (AMMod(0.5), AMDemod(0.5)),
        (AMMod(0.5), AMDemod(0.5, method="coherent")),
        (AMMod(0.7, carrier=False), AMDemod(0.7, carrier=False, method="coherent")),
    ],
    ids=["envelope", "coherent", "dsb-sc"],
)
def test_am_demod_inverts_mod(mod, demod):
    np.testing.assert_allclose(through([mod, demod], MSG), MSG, atol=1e-12)


def test_am_chunk_invariant():
    assert_chunk_invariant(lambda: AMMod(0.5), MSG, fs=FS)
    z = AMMod(0.5)(MSG) * np.exp(1j * 0.3)
    assert_chunk_invariant(lambda: AMDemod(0.5, channel_width=10e3), z, fs=FS)


def test_am_power_split_between_carrier_and_sidebands():
    # theory: a tone at full scale gives mean |z|^2 = 1 + m^2 / 2
    m = 0.6
    s, mod, sc = Sine(1e3), AMMod(m), Scope()
    s >> mod >> sc
    Diagram(sc, fs=FS).run(n=48_000, chunk=1024)
    np.testing.assert_allclose(np.mean(np.abs(sc.data) ** 2), 1 + m**2 / 2, rtol=1e-9)


def test_envelope_ignores_carrier_phase_coherent_does_not():
    # theory: a phase error theta scales the coherent output's envelope by cos(theta)
    theta = 0.4
    z = AMMod(0.5)(MSG) * np.exp(1j * theta)
    np.testing.assert_allclose(through([AMDemod(0.5)], z), MSG, atol=1e-12)
    coherent = through([AMDemod(0.5, method="coherent")], z)
    np.testing.assert_allclose(
        coherent, (np.cos(theta) * (1 + 0.5 * MSG) - 1) / 0.5, atol=1e-12
    )


def test_channel_filter_rejects_an_adjacent_signal():
    # a 1 kHz message plus an interferer 15 kHz away: with a 10 kHz channel
    # filter the message comes back, delayed by the filter's group delay
    n, taps = 9600, 201
    msg = np.cos(2 * np.pi * 1e3 * np.arange(n) / FS)
    z = AMMod(0.5)(msg) + 0.5 * np.exp(2j * np.pi * 15e3 * np.arange(n) / FS)
    y = through([AMDemod(0.5, channel_width=10e3, taps=taps)], z)
    delay = (taps - 1) // 2
    np.testing.assert_allclose(y[taps:], msg[taps - delay : n - delay], atol=0.01)
    without = through([AMDemod(0.5)], z)
    assert np.max(np.abs(without[taps:] - msg[taps:])) > 0.5  # unfiltered fails


# fm


def test_fm_mod_has_unit_envelope_and_the_right_frequency():
    # theory: a constant message x gives a tone at x * deviation
    z = FMMod(1000.0, fs=FS)(np.full(480, 0.5))
    np.testing.assert_allclose(np.abs(z), 1.0)
    f = np.angle(z[1:] * np.conj(z[:-1])) * FS / (2 * np.pi)
    np.testing.assert_allclose(f, 500.0)


@pytest.mark.parametrize("chunk", [1, 97, 2000])
def test_fm_demod_inverts_mod_for_any_chunking(chunk):
    y = through([FMMod(5e3), FMDemod(5e3)], MSG, chunk=chunk)
    np.testing.assert_allclose(y, MSG, atol=1e-9)


def test_fm_chunk_invariant():
    assert_chunk_invariant(lambda: FMMod(3e3), MSG, fs=FS)
    z = FMMod(3e3, fs=FS)(MSG)
    assert_chunk_invariant(lambda: FMDemod(3e3, channel_width=20e3), z, fs=FS)


def test_fm_carrier_vanishes_at_the_first_bessel_zero():
    # theory: a tone at modulation index beta leaves a carrier of J0(beta);
    # J0 has its first zero at beta = 2.405
    fm, n = 1000.0, 48_000
    for beta in (1.0, 2.405):
        s, mod, sc = Sine(fm), FMMod(beta * fm), Scope()
        s >> mod >> sc
        Diagram(sc, fs=FS).run(n=n, chunk=1024)
        carrier = abs(np.mean(sc.data))
        assert carrier == pytest.approx(abs(jv(0, beta)), abs=1e-3)


def test_carsons_rule_holds_98_percent_of_the_power():
    # theory: 2 (deviation + fm) holds about 98 % of an FM signal's power
    fm, dev, n = 1000.0, 5000.0, 48_000
    s, mod, sc = Sine(fm), FMMod(dev), Scope()
    s >> mod >> sc
    Diagram(sc, fs=FS).run(n=n, chunk=1024)
    P = np.abs(np.fft.fft(sc.data)) ** 2
    f = np.fft.fftfreq(n, 1 / FS)
    inside = P[np.abs(f) <= dev + fm].sum() / P.sum()
    assert inside > 0.98


def test_channel_filter_lets_fm_through_and_rejects_a_neighbour():
    # carson width is 2 (5 + 1) kHz = 12 kHz, but a channel that narrow clips
    # sidebands (measured: 6 % error at 14 kHz); 20 kHz leaves 0.3 %, and a
    # neighbour 20 kHz away is still removed
    n, taps = 19_200, 201
    msg = np.cos(2 * np.pi * 1e3 * np.arange(n) / FS)
    z = FMMod(5e3, fs=FS)(msg) + 0.5 * np.exp(2j * np.pi * 20e3 * np.arange(n) / FS)
    y = through([FMDemod(5e3, channel_width=20e3, taps=taps)], z)
    delay = (taps - 1) // 2
    np.testing.assert_allclose(y[taps:], msg[taps - delay : n - delay], atol=0.005)
    bad = through([FMDemod(5e3)], z)
    assert np.max(np.abs(bad[taps:] - msg[taps:])) > 0.5


# pm


def test_pm_mod_values():
    np.testing.assert_allclose(PMMod(2.0)(MSG), np.exp(2j * MSG))


@pytest.mark.parametrize("chunk", [1, 97, 2000])
def test_pm_demod_unwraps_many_turns_for_any_chunking(chunk):
    # theory: a slow ramp to 10 turns of phase comes back exactly, wraps and all
    x = np.linspace(0, 1, 2000)
    index = 10 * 2 * np.pi
    y = through([PMMod(index), PMDemod(index)], x, chunk=chunk)
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_pm_chunk_invariant():
    assert_chunk_invariant(lambda: PMMod(1.5), MSG, fs=FS)
    z = PMMod(1.5)(MSG)
    assert_chunk_invariant(lambda: PMDemod(1.5, channel_width=20e3), z, fs=FS)


def test_pm_equals_fm_of_the_first_difference():
    # comparison: FM accumulates its message, so PM of x equals FM of
    # x[n] - x[n-1], scaled by fs / (2 pi deviation), exactly
    b, dev = 1.2, 1000.0
    x = np.cos(2 * np.pi * 1000.0 * np.arange(4800) / FS)
    pm = PMMod(b)(x)
    dx = np.diff(x, prepend=0.0) * b * FS / (2 * np.pi * dev)
    fm = FMMod(dev, fs=FS)(dx)
    np.testing.assert_allclose(fm, pm, atol=1e-9)
