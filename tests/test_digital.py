from fractions import Fraction

import numpy as np
import pytest

from sigline import BitsToWords, Diagram, FromArray, Scope, WordsToBits, pam, psk, qam
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
