from fractions import Fraction

import numpy as np
import pytest
from scipy.special import erfc

from sigline import (
    BERCounter,
    BitsToWords,
    Demapper,
    Diagram,
    FromArray,
    Mapper,
    Noise,
    Scope,
    WordsToBits,
    pam,
    psk,
    qam,
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
    m = Mapper(qam(16))
    np.testing.assert_allclose(
        m(np.array([0, 0, 0, 1, 1, 1, 1, 1.0])), qam(16)[[1, 15]]
    )
    assert m.k == 4 and m.resampling_ratio == Fraction(1, 4)


@pytest.mark.parametrize(
    "points", [pam(8), psk(8), qam(64)], ids=["8pam", "8psk", "64qam"]
)
@pytest.mark.parametrize("chunk", [6, 600])
def test_demapper_inverts_mapper(points, chunk):
    np.testing.assert_array_equal(
        run([Mapper(points), Demapper(points)], BITS, fs=6.0, chunk=chunk), BITS
    )


def test_mapper_chunk_invariant():
    assert_chunk_invariant(lambda: Mapper(qam(16)), BITS[:1200])
    sym = Mapper(qam(16))(BITS) + 0.1 * (
        rng.standard_normal(300) + 1j * rng.standard_normal(300)
    )
    assert_chunk_invariant(lambda: Demapper(qam(16)), sym)


def ber(points, ebn0_db, n_bits=240_000, seed=1):
    k = int(np.log2(len(points)))
    esn0 = k * 10 ** (ebn0_db / 10)
    bits = FromArray(np.random.default_rng(seed).integers(0, 2, n_bits).astype(float))
    meter = BERCounter(sync=600)
    # unit-energy symbols plus complex noise of variance N0 = 1 / (Es/N0)
    sym = bits >> Mapper(points)
    rx = sym + Noise(
        np.sqrt(1 / esn0), seed=seed + 1, complex=True, fs=1.0
    )  # symbol rate
    rx >> Demapper(points) >> meter["received"]
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
