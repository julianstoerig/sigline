import numpy as np
import pytest

from sigline import AMDemod, AMMod, Diagram, FromArray, Scope, Sine
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
