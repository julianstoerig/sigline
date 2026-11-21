from fractions import Fraction

import numpy as np
import pytest

from sigline import BitsToWords, Diagram, FromArray, Scope, WordsToBits
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
