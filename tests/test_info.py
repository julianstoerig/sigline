import numpy as np
import pytest
import scipy.signal as sig

from sigline import (
    FIR,
    IIR,
    Block,
    Const,
    Delay,
    FromArray,
    Func,
    Gain,
    Mixer,
    Noise,
    Product,
    RtlSdrSource,
    Scope,
    Signal,
    Sine,
    Step,
    Sum,
)
from sigline._format import si


@pytest.mark.parametrize(
    ("v", "unit", "text"),
    [
        (1500, "Hz", "1.5 kHz"),
        (145.8e6, "Hz", "145.8 MHz"),
        (2.4e9, "Hz", "2.4 GHz"),
        (2e-3, "s", "2 ms"),
        (5e-7, "s", "0.5 µs"),
        (-48000, "Hz", "-48 kHz"),
        (0, "V", "0 V"),
        (12.5, "", "12.5"),
    ],
)
def test_si_prefixes(v, unit, text):
    assert si(v, unit) == text


def test_block_info_default_is_empty():
    assert Block().info() == ""


@pytest.mark.parametrize(
    ("block", "text"),
    [
        (Const(2.5), "2.5"),
        (Step(1e-3, after=2.0), "0→2 @ 1 ms"),
        (Signal(np.sin), "sin"),
        (Signal(lambda t: t), ""),
        (FromArray(np.zeros(10), loop=True), "10 samples loop"),
        (Sine(1e3), "1 kHz"),
        (Sine(2.5e6, amp=0.5, complex=True), "2.5 MHz cplx ×0.5"),
        (Noise(0.3, complex=True), "σ=0.3 cplx"),
        (Gain(-2), "×-2"),
        (Gain(np.ones(2)), "×g"),
        (Sum("+-+"), "+ - +"),
        (Product(), "×"),
        (Func(np.abs), ""),
        (Func(np.abs, name="mag"), "absolute"),
        (Delay(4), "z^-4"),
        (Mixer(100e3, "exp-"), "100 kHz exp-"),
        (FIR(sig.firwin(31, 0.2)), "31 taps"),
        (IIR(sos=sig.butter(4, 0.2, output="sos")), "2 SOS"),
        (IIR(b=[1], a=[1, -0.5]), "order 1"),
        (Scope(keep=100, every=4), "keep 100, every 4"),
        (Scope(), ""),
        (RtlSdrSource(145.8e6, gain=30), "145.8 MHz, 30 dB"),
        (RtlSdrSource(100e6), "100 MHz, auto"),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Block) else None,
)
def test_info_summaries(block, text):
    assert block.info() == text
