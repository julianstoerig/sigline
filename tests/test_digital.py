from fractions import Fraction

import numpy as np
import pytest
from scipy.special import erfc

from sigline import (
    BERCounter,
    BitsToWords,
    ConstellationScope,
    Diagram,
    Downsample,
    FromArray,
    FSKDemod,
    Noise,
    Scope,
    WordsToBits,
    demapper,
    fsk_mod,
    mapper,
    matched_filter,
    pam,
    psk,
    pulse_shaper,
    qam,
    rrc_taps,
)
from sigline.testing import assert_chunk_invariant

rng = np.random.default_rng(12)
BITS = rng.integers(0, 2, 1200).astype(float)


def run(blocks, x, fs=1.0, chunk=97):
    src, sc = FromArray(x, fs=fs), Scope()
    node = src
    for b in blocks:
        node = node >> b
    node >> sc
    d = Diagram(sc, fs=fs)
    d.run(n=len(x) // d.granularity * d.granularity, chunk=chunk)
    return sc.data


# words and bits


def test_words_to_bits_msb_first():
    np.testing.assert_array_equal(
        WordsToBits(4)(np.array([1.0, 8.0, 15.0])), [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    )


def test_bits_to_words_thresholds_and_rates():
    np.testing.assert_array_equal(
        BitsToWords(2)(np.array([0.9, 0.1, 0.2, 0.7])), [2, 1]
    )
    assert BitsToWords(3).resampling_ratio == Fraction(1, 3)
    assert WordsToBits(3).info() == "3 bits/word"


@pytest.mark.parametrize("chunk", [6, 60, 1200])
def test_round_trip_for_any_chunking(chunk):
    # comparison: bits -> words -> bits gives the bits back
    np.testing.assert_array_equal(
        run([BitsToWords(6), WordsToBits(6)], BITS, fs=6.0, chunk=chunk), BITS
    )


def test_words_match_numpy_packbits():
    # comparison with numpy's packbits for 8-bit words
    w = run([BitsToWords(8)], BITS, chunk=40)
    np.testing.assert_array_equal(w, np.packbits(BITS.astype(np.uint8)))


def test_chunk_invariant():
    assert_chunk_invariant(lambda: BitsToWords(4), BITS)
    assert_chunk_invariant(
        lambda: WordsToBits(4), rng.integers(0, 16, 300).astype(float)
    )


# constellations

ALL = [
    pam(2),
    pam(4),
    pam(8),
    psk(2),
    psk(4),
    psk(8),
    psk(16),
    qam(4),
    qam(16),
    qam(64),
]
IDS = [
    "2pam",
    "4pam",
    "8pam",
    "bpsk",
    "qpsk",
    "8psk",
    "16psk",
    "4qam",
    "16qam",
    "64qam",
]


@pytest.mark.parametrize("points", ALL, ids=IDS)
def test_unit_average_energy_and_distinct_points(points):
    assert np.mean(np.abs(points) ** 2) == pytest.approx(1.0)
    assert len(np.unique(np.round(points, 9))) == len(points)


@pytest.mark.parametrize("points", ALL, ids=IDS)
def test_gray_coding_nearest_neighbours_differ_by_one_bit(points):
    d = np.abs(points[:, None] - points[None, :])
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    for a, b in zip(*np.nonzero(np.isclose(d, dmin)), strict=True):
        assert bin(int(a) ^ int(b)).count("1") == 1


def test_known_points_and_minimum_distances():
    # theory: BPSK +-1; QPSK at 45 degrees; dmin of square M-QAM with unit
    # energy is sqrt(6 / (M - 1)), of M-PSK 2 sin(pi / M)
    np.testing.assert_allclose(psk(2), [1, -1])
    np.testing.assert_allclose(np.angle(psk(4)) % (np.pi / 2), np.pi / 4)
    for M in (16, 64):
        d = np.abs(qam(M)[:, None] - qam(M)[None, :])
        assert d[d > 1e-9].min() == pytest.approx(np.sqrt(6 / (M - 1)))
    d = np.abs(psk(8)[:, None] - psk(8)[None, :])
    assert d[d > 1e-9].min() == pytest.approx(2 * np.sin(np.pi / 8))


def test_constellation_validation():
    with pytest.raises(ValueError, match="power of 2"):
        pam(6)
    with pytest.raises(ValueError, match="even number of bits"):
        qam(32)


# mapper and demapper


def test_mapper_picks_points_msb_first():
    m = mapper(qam(16), fs=1.0)
    np.testing.assert_allclose(
        m(np.array([0, 0, 0, 1, 1, 1, 1, 1.0])), qam(16)[[1, 15]]
    )
    assert m.words.bits == 4 and m.words.resampling_ratio == Fraction(1, 4)
    assert m.type == "mapper" and m.info() == "16 points"
    np.testing.assert_array_equal(m.table.params["points"], qam(16))


@pytest.mark.parametrize(
    "points", [pam(8), psk(8), qam(64)], ids=["8pam", "8psk", "64qam"]
)
@pytest.mark.parametrize("chunk", [6, 600])
def test_demapper_inverts_mapper(points, chunk):
    np.testing.assert_array_equal(
        run([mapper(points), demapper(points)], BITS, fs=6.0, chunk=chunk), BITS
    )


def test_mapper_chunk_invariant():
    assert_chunk_invariant(lambda: mapper(qam(16)), BITS[:1200])
    sym = mapper(qam(16), fs=1.0)(BITS) + 0.1 * (
        rng.standard_normal(300) + 1j * rng.standard_normal(300)
    )
    assert_chunk_invariant(lambda: demapper(qam(16)), sym)


def ber(points, ebn0_db, n_bits=240_000, seed=1):
    k = int(np.log2(len(points)))
    esn0 = k * 10 ** (ebn0_db / 10)
    bits = FromArray(np.random.default_rng(seed).integers(0, 2, n_bits).astype(float))
    meter = BERCounter(sync=600)
    # unit-energy symbols plus complex noise of variance N0 = 1 / (Es/N0)
    sym = bits >> mapper(points)
    rx = sym + Noise(
        np.sqrt(1 / esn0), seed=seed + 1, complex=True, fs=1.0
    )  # symbol rate
    rx >> demapper(points) >> meter["received"]
    bits >> meter["desired"]
    Diagram(meter, fs=float(k)).run(n=n_bits, chunk=6000)
    return meter


def test_qpsk_ber_matches_theory():
    # theory, Gray QPSK: Pb = erfc(sqrt(Eb/N0)) / 2
    m = ber(psk(4), 6.0)
    theory = 0.5 * erfc(np.sqrt(10**0.6))
    lo, hi = m.confidence(0.999)
    assert m.delay == 0 and lo < theory < hi


def test_16qam_ber_matches_theory():
    # theory, Gray square M-QAM: Pb ~ 4/k (1 - 1/sqrt(M)) Q(sqrt(3 k Eb/N0 / (M - 1)))
    M, k, ebn0 = 16, 4, 10**1.0
    m = ber(qam(M), 10.0)
    q = 0.5 * erfc(np.sqrt(3 * k * ebn0 / (M - 1)) / np.sqrt(2))
    theory = 4 / k * (1 - 1 / np.sqrt(M)) * q
    lo, hi = m.confidence(0.999)
    assert lo < theory < hi


# pulse shaping


def test_rrc_taps_shape():
    h = rrc_taps(0.35, 8, 10)
    assert len(h) == 81 and np.sum(h**2) == pytest.approx(1.0)
    np.testing.assert_allclose(h, h[::-1], atol=1e-15)


@pytest.mark.parametrize("beta", [0.2, 0.35, 0.5, 1.0])
def test_rrc_twice_is_a_nyquist_pulse(beta):
    # theory: rrc * rrc = raised cosine, zero at every other symbol instant;
    # truncating the tails leaves interference measured 40 to 71 dB down
    sps, span = 8, 16
    rc = np.convolve(rrc_taps(beta, sps, span), rrc_taps(beta, sps, span))
    c = len(rc) // 2
    others = np.delete(rc[c % sps :: sps], c // sps)
    assert rc[c] == pytest.approx(1.0)
    assert 10 * np.log10(np.sum(others**2)) < -38


def test_shaper_and_matched_filter_give_the_symbols_back():
    # integration: span symbols of delay, then the symbols; after the start-up
    # transient, MER is limited by the filters' truncation (41 dB at span 10)
    sps, span, ref = 8, 10, qam(16)
    sym = ref[rng.integers(0, 16, 2000)]
    src, cs = FromArray(sym, fs=1.0), ConstellationScope(ref, skip=2 * span)
    rx = src >> pulse_shaper(sps, span=span) >> matched_filter(sps, span=span)
    rx >> Downsample(sps) >> cs
    Diagram(cs, fs=1.0).run(n=2000, chunk=100)
    got, want = cs.points[-1000:], sym[1000 - span : 2000 - span]
    np.testing.assert_allclose(got, want, atol=0.03)
    assert cs.mer_db > 38


def test_spectrum_stays_within_the_rolloff_bandwidth():
    # theory: rrc-shaped symbols occupy |f| <= (1 + beta) Rs / 2
    sps, beta = 8, 0.35
    sym = psk(4)[rng.integers(0, 4, 20_000)]
    y = run([pulse_shaper(sps, beta=beta, span=16)], sym, chunk=1000)
    f = np.fft.fftfreq(len(y), 1 / sps)  # in units of Rs
    P = np.abs(np.fft.fft(y)) ** 2
    assert P[np.abs(f) > (1 + beta) / 2 * 1.05].sum() / P.sum() < 1e-3


def test_shaped_qpsk_link_ber_matches_theory():
    # theory: with unit-energy rrc filters, noise of variance N0 per sample
    # reaches the decision with variance N0, so Pb = erfc(sqrt(Eb/N0)) / 2
    sps, ebn0_db, n_bits = 4, 6.0, 200_000
    esn0 = 2 * 10 ** (ebn0_db / 10)
    bits = FromArray(np.random.default_rng(5).integers(0, 2, n_bits).astype(float))
    meter = BERCounter(sync=600)
    tx = bits >> mapper(psk(4)) >> pulse_shaper(sps)
    rx = tx + Noise(np.sqrt(1 / esn0), seed=6, complex=True, fs=1.0 * sps)
    (
        rx
        >> matched_filter(sps)
        >> Downsample(sps)
        >> demapper(psk(4))
        >> meter["received"]
    )
    bits >> meter["desired"]
    Diagram(meter, fs=2.0).run(n=n_bits, chunk=8000)
    lo, hi = meter.confidence(0.999)
    assert meter.delay == 20 and lo < 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10))) < hi


def test_pulse_shaping_chunk_invariant():
    sym = psk(4)[rng.integers(0, 4, 300)]
    assert_chunk_invariant(lambda: pulse_shaper(4), sym)
    assert_chunk_invariant(lambda: matched_filter(4), np.repeat(sym, 4))
    assert pulse_shaper(4).info() == "4 sps, RRC β=0.35"


# fsk


@pytest.mark.parametrize(
    ("M", "h"), [(2, 1.0), (4, 1.0), (2, 0.5)], ids=["bfsk", "4fsk", "msk"]
)
def test_fsk_demod_inverts_mod_for_any_chunking(M, h):
    sym = rng.integers(0, M, 600).astype(float)
    for chunk in (1, 77, 600):
        y = run([fsk_mod(M, 8, h, fs=1.0), FSKDemod(M, 8, h)], sym, chunk=chunk)
        np.testing.assert_array_equal(y, sym)


def test_fsk_tones_and_continuous_phase():
    # theory: 2-FSK with h = 0.5 (MSK) at Rs = 1 sends -1/4 and +1/4 Rs,
    # with unit envelope and phase steps of +-pi/16 at 8 samples per symbol
    z = fsk_mod(2, 8, 0.5, fs=1.0)(np.array([0.0, 1.0, 1.0, 0.0]))
    np.testing.assert_allclose(np.abs(z), 1.0)
    steps = np.angle(z[1:] * np.conj(z[:-1]))
    np.testing.assert_allclose(np.abs(steps), np.pi / 16, atol=1e-12)


def test_fsk_chunk_invariant():
    sym = rng.integers(0, 4, 300).astype(float)
    assert_chunk_invariant(lambda: fsk_mod(4, 4, fs=1.0), sym)
    assert_chunk_invariant(lambda: FSKDemod(4, 4), fsk_mod(4, 4, fs=1.0)(sym), fs=4.0)


def test_noncoherent_bfsk_ber_matches_theory():
    # theory, noncoherent orthogonal BFSK: Pb = exp(-Eb / (2 N0)) / 2; a
    # symbol has energy sps (unit amplitude), noise variance N0 per sample
    sps, ebn0_db, n_bits = 8, 9.0, 200_000
    ebn0 = 10 ** (ebn0_db / 10)
    bits = FromArray(np.random.default_rng(7).integers(0, 2, n_bits).astype(float))
    meter = BERCounter(sync=600)
    tx = bits >> fsk_mod(2, sps, fs=1.0)
    rx = tx + Noise(np.sqrt(sps / ebn0), seed=8, complex=True, fs=float(sps))
    rx >> FSKDemod(2, sps) >> meter["received"]
    bits >> meter["desired"]
    Diagram(meter, fs=1.0).run(n=n_bits, chunk=8000)
    lo, hi = meter.confidence(0.999)
    assert meter.delay == 0 and lo < 0.5 * np.exp(-ebn0 / 2) < hi


def test_fsk_is_fm_of_held_levels():
    # comparison: fsk_mod equals fm_mod of the tone levels held for sps samples
    from sigline import Repeat, fm_mod

    sym = rng.integers(0, 4, 200).astype(float)
    z = fsk_mod(4, 8, 0.7, fs=1.0)(sym)
    held = Repeat(8)(sym - 1.5)
    np.testing.assert_allclose(z, fm_mod(0.7, fs=8.0)(held), atol=1e-12)
