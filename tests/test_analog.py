import numpy as np
import pytest
from scipy.special import jv

from sigline import (
    Diagram,
    FromArray,
    Scope,
    Sine,
    am_demod,
    am_mod,
    fm_demod,
    fm_mod,
    hilbert_taps,
    pm_demod,
    pm_mod,
    ssb_demod,
    ssb_mod,
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
    mod = am_mod(0.5, fs=FS)
    assert mod.type == "am_mod" and mod.info() == "m=0.5"
    np.testing.assert_allclose(mod(MSG), 1 + 0.5 * MSG)
    mod.index.g = 0.6  # parameters stay attributes, on the inner blocks
    assert mod.info() == "m=0.6"
    z = am_mod(0.8, carrier=False, fs=FS)(MSG)
    assert z.dtype == float  # the complex envelope of AM is real
    np.testing.assert_allclose(z, 0.8 * MSG)


def test_am_demod_validates():
    with pytest.raises(ValueError, match="needs the carrier"):
        am_demod(carrier=False)
    with pytest.raises(ValueError, match="method"):
        am_demod(method="square-law")
    with pytest.raises(ValueError, match="channel_width must be"):
        am_demod(channel_width=1e6, fs=FS)
    with pytest.raises(ValueError, match="needs fs"):
        am_demod(channel_width=10e3)


@pytest.mark.parametrize(
    ("mod", "demod"),
    [
        (am_mod(0.5), am_demod(0.5)),
        (am_mod(0.5), am_demod(0.5, method="coherent")),
        (am_mod(0.7, carrier=False), am_demod(0.7, carrier=False, method="coherent")),
    ],
    ids=["envelope", "coherent", "dsb-sc"],
)
def test_am_demod_inverts_mod(mod, demod):
    np.testing.assert_allclose(through([mod, demod], MSG), MSG, atol=1e-12)


def test_am_chunk_invariant():
    assert_chunk_invariant(lambda: am_mod(0.5), MSG, fs=FS)
    z = am_mod(0.5, fs=FS)(MSG) * np.exp(1j * 0.3)
    assert_chunk_invariant(lambda: am_demod(0.5, channel_width=10e3, fs=FS), z, fs=FS)


def test_am_power_split_between_carrier_and_sidebands():
    # theory: a tone at full scale gives mean |z|^2 = 1 + m^2 / 2
    m = 0.6
    s, mod, sc = Sine(1e3), am_mod(m), Scope()
    s >> mod >> sc
    Diagram(sc, fs=FS).run(n=48_000, chunk=1024)
    np.testing.assert_allclose(np.mean(np.abs(sc.data) ** 2), 1 + m**2 / 2, rtol=1e-9)


def test_envelope_ignores_carrier_phase_coherent_does_not():
    # theory: a phase error theta scales the coherent output's envelope by cos(theta)
    theta = 0.4
    z = am_mod(0.5, fs=FS)(MSG) * np.exp(1j * theta)
    np.testing.assert_allclose(through([am_demod(0.5)], z), MSG, atol=1e-12)
    coherent = through([am_demod(0.5, method="coherent")], z)
    np.testing.assert_allclose(
        coherent, (np.cos(theta) * (1 + 0.5 * MSG) - 1) / 0.5, atol=1e-12
    )


def test_channel_filter_rejects_an_adjacent_signal():
    # a 1 kHz message plus an interferer 15 kHz away: with a 10 kHz channel
    # filter the message comes back, delayed by the filter's group delay
    n, taps = 9600, 201
    msg = np.cos(2 * np.pi * 1e3 * np.arange(n) / FS)
    z = am_mod(0.5, fs=FS)(msg) + 0.5 * np.exp(2j * np.pi * 15e3 * np.arange(n) / FS)
    y = through([am_demod(0.5, channel_width=10e3, taps=taps, fs=FS)], z)
    delay = (taps - 1) // 2
    np.testing.assert_allclose(y[taps:], msg[taps - delay : n - delay], atol=0.01)
    without = through([am_demod(0.5)], z)
    assert np.max(np.abs(without[taps:] - msg[taps:])) > 0.5  # unfiltered fails


# fm


def test_fm_mod_has_unit_envelope_and_the_right_frequency():
    # theory: a constant message x gives a tone at x * deviation
    z = fm_mod(1000.0, fs=FS)(np.full(480, 0.5))
    np.testing.assert_allclose(np.abs(z), 1.0)
    f = np.angle(z[1:] * np.conj(z[:-1])) * FS / (2 * np.pi)
    np.testing.assert_allclose(f, 500.0)


@pytest.mark.parametrize("chunk", [1, 97, 2000])
def test_fm_demod_inverts_mod_for_any_chunking(chunk):
    # the phase integrates the message up to the sample before: one sample late
    y = through([fm_mod(5e3, fs=FS), fm_demod(5e3, fs=FS)], MSG, chunk=chunk)
    np.testing.assert_allclose(y[1:], MSG[:-1], atol=1e-9)
    assert y[0] == 0.0


def test_fm_chunk_invariant():
    assert_chunk_invariant(lambda: fm_mod(3e3, fs=FS), MSG, fs=FS)
    z = fm_mod(3e3, fs=FS)(MSG)
    assert_chunk_invariant(lambda: fm_demod(3e3, channel_width=20e3, fs=FS), z, fs=FS)


def test_fm_carrier_vanishes_at_the_first_bessel_zero():
    # theory: a tone at modulation index beta leaves a carrier of J0(beta);
    # J0 has its first zero at beta = 2.405
    fm, n = 1000.0, 48_000
    for beta in (1.0, 2.405):
        s, mod, sc = Sine(fm), fm_mod(beta * fm, fs=FS), Scope()
        s >> mod >> sc
        Diagram(sc, fs=FS).run(n=n, chunk=1024)
        carrier = abs(np.mean(sc.data))
        assert carrier == pytest.approx(abs(jv(0, beta)), abs=1e-3)


def test_carsons_rule_holds_98_percent_of_the_power():
    # theory: 2 (deviation + fm) holds about 98 % of an FM signal's power
    fm, dev, n = 1000.0, 5000.0, 48_000
    s, mod, sc = Sine(fm), fm_mod(dev, fs=FS), Scope()
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
    z = fm_mod(5e3, fs=FS)(msg) + 0.5 * np.exp(2j * np.pi * 20e3 * np.arange(n) / FS)
    y = through([fm_demod(5e3, channel_width=20e3, taps=taps, fs=FS)], z)
    delay = (taps - 1) // 2 + 1  # filter, and fm_mod's one sample
    np.testing.assert_allclose(y[taps:], msg[taps - delay : n - delay], atol=0.005)
    bad = through([fm_demod(5e3, fs=FS)], z)
    assert np.max(np.abs(bad[taps:] - msg[taps:])) > 0.5


# pm


def test_pm_mod_values():
    np.testing.assert_allclose(pm_mod(2.0, fs=FS)(MSG), np.exp(2j * MSG))


@pytest.mark.parametrize("chunk", [1, 97, 2000])
def test_pm_demod_unwraps_many_turns_for_any_chunking(chunk):
    # theory: a slow ramp to 10 turns of phase comes back exactly, wraps and all
    x = np.linspace(0, 1, 2000)
    index = 10 * 2 * np.pi
    y = through([pm_mod(index), pm_demod(index)], x, chunk=chunk)
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_pm_chunk_invariant():
    assert_chunk_invariant(lambda: pm_mod(1.5), MSG, fs=FS)
    z = pm_mod(1.5, fs=FS)(MSG)
    assert_chunk_invariant(lambda: pm_demod(1.5, channel_width=20e3, fs=FS), z, fs=FS)


def test_pm_equals_fm_of_the_first_difference():
    # comparison: FM accumulates its message up to the sample before, so PM
    # of x equals FM of x[n] - x[n-1], scaled by fs / (2 pi deviation), one
    # sample later
    b, dev = 1.2, 1000.0
    x = np.cos(2 * np.pi * 1000.0 * np.arange(4800) / FS)
    pm = pm_mod(b, fs=FS)(x)
    dx = np.diff(x, prepend=0.0) * b * FS / (2 * np.pi * dev)
    fm = fm_mod(dev, fs=FS)(dx)
    np.testing.assert_allclose(fm[1:], pm[:-1], atol=1e-9)


# ssb


def tone_level(z, f):
    # magnitude of the component at f / Hz (windowed, so leakage stays small)
    w = np.hanning(len(z))
    t = np.arange(len(z)) / FS
    return abs(np.sum(w * z * np.exp(-2j * np.pi * f * t)) / np.sum(w))


@pytest.mark.parametrize(("side", "f_want"), [("upper", 1000.0), ("lower", -1000.0)])
def test_ssb_keeps_one_sideband(side, f_want):
    # theory: H{cos} = sin, so x + j H{x} of a cosine is exp(+j w t)
    z = through(
        [ssb_mod(side, taps=201)], np.cos(2 * np.pi * 1e3 * np.arange(9600) / FS)
    )
    z = z[201:]
    wanted, image = tone_level(z, f_want), tone_level(z, -f_want)
    assert wanted == pytest.approx(1.0, abs=0.01)
    assert 20 * np.log10(image / wanted) < -40


def test_ssb_demod_recovers_the_delayed_message():
    taps = 101
    y = through([ssb_mod(taps=taps), ssb_demod()], MSG)
    d = (taps - 1) // 2
    np.testing.assert_allclose(y[d:], MSG[: len(MSG) - d], atol=1e-12)


def test_one_sided_filter_rejects_the_other_sideband():
    # a USB voice tone at +1 kHz and an interferer in the LSB at -1.5 kHz
    n, taps = 19_200, 201
    t = np.arange(n) / FS
    z = np.exp(2j * np.pi * 1e3 * t) + 0.8 * np.exp(-2j * np.pi * 1.5e3 * t)
    y = through([ssb_demod("upper", channel_width=3e3, taps=taps, fs=FS)], z)[taps:]
    assert tone_level(y + 0j, 1e3) == pytest.approx(
        0.5, abs=0.01
    )  # cos -> 0.5 per side
    assert tone_level(y + 0j, 1.5e3) < 0.01
    plain = through([ssb_demod("upper")], z)[taps:]
    assert tone_level(plain + 0j, 1.5e3) > 0.3


def test_ssb_validation_and_chunk_invariance():
    with pytest.raises(ValueError, match="side"):
        ssb_mod("both")
    with pytest.raises(ValueError, match="odd"):
        hilbert_taps(100)
    assert_chunk_invariant(lambda: ssb_mod("lower"), MSG, fs=FS)
    z = ssb_mod(fs=FS)(MSG)
    assert_chunk_invariant(lambda: ssb_demod(channel_width=3e3, fs=FS), z, fs=FS)


def test_pm_start_up_transient_leaves_no_offset():
    # the channel filter starts empty; its transient must not slip the phase
    # (before the fix: 13 slips of 2 pi, an offset of 54 in this setting)
    msg = 0.5 * Sine(500.0) + 0.4 * Sine(1300.0)
    rx = (msg >> pm_mod(1.5)) + 0.3 * Sine(12e3, complex=True)
    demod, sc = pm_demod(1.5, channel_width=14e3, taps=201, fs=FS), Scope()
    rx >> demod >> sc
    Diagram(sc, fs=FS).run(0.5)
    y = sc.data[201:]
    assert abs(np.mean(y)) < 0.01 and np.max(np.abs(np.diff(y))) < 0.5


# subsystems of library blocks


def test_each_is_a_subsystem_of_named_library_blocks():
    shapes = {
        am_mod(): ["index", "carrier", "sum"],
        am_demod(channel_width=10e3, fs=FS): [
            "channel",
            "detector",
            "scale",
            "sum",
            "carrier",
        ],
        fm_mod(5e3, fs=FS): ["scale", "phase", "exp"],
        fm_demod(5e3, fs=FS): ["previous", "discriminator", "scale"],
        pm_mod(1.0): ["index", "exp"],
        pm_demod(1.0): ["angle", "scale", "unwrap"],
        ssb_mod(): ["delay", "hilbert", "j", "sum"],
        ssb_demod(channel_width=3e3, fs=FS): ["real", "channel"],
    }
    for sub, names in shapes.items():
        assert sorted(m.name.split("/")[1] for m in sub.members) == sorted(names)
        assert sub.inputs == ("x",) and sub.outputs == ("y",)


def test_fm_needs_the_sample_rate():
    with pytest.raises(TypeError, match="fs"):
        fm_mod(5e3)
    with pytest.raises(TypeError, match="fs"):
        fm_demod(5e3)
