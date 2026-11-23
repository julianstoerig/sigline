"""Channel models: noise added to or multiplied onto a signal, as subsystems.

Each function wires library blocks and returns them as a :class:`Subsystem`
with input ``x`` and output ``y``; inner blocks have names, so parameters
stay attributes (``ch.noise.std = 0.1``).

Power is ``mean(|x|^2)``: watts into 1 ohm, or plain units squared. Complex
noise of power ``P`` has ``P / 2`` in each of I and Q. A source cannot see
the signal, so give ``complex=True`` for complex baseband; the inner ``Sum``
checks on the first chunk that signal and noise are both real or both
complex. ``width`` gives vector signals independent noise per channel.
Every random part takes a ``seed``; any chunking gives the same result.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
from scipy.constants import k as BOLTZMANN

from ._format import si
from .analog import expj, mixer
from .blocks import FuncChunk, Gain, Product, Sum
from .digital import mapper, psk, pulse_shaper, table
from .filters import IIR
from .sources import (
    Chirp,
    MarkovChain,
    Noise,
    Pulses,
    RandomBits,
    RandomPulses,
    Sine,
)
from .subsystem import Subsystem

__all__ = [
    "BOLTZMANN",
    "COLORS",
    "awgn",
    "burst_noise",
    "chirp_interferer",
    "colored_noise",
    "impulsive_noise",
    "modulated_interferer",
    "phase_noise",
    "pulsed_interferer",
    "tone_interferer",
]


def _adder(fs):
    # x + noise; signal and noise must be of one kind and width
    add = Sum(fs=fs, name="add")
    add.same_kind_inputs = True
    add.expands_scalars = False
    return add


def awgn(
    snr_db=None,
    *,
    power=None,
    temperature=None,
    signal_power=1.0,
    bandwidth=None,
    complex=False,
    width=None,
    seed=0,
    fs=None,
    name=None,
):
    """Add white Gaussian noise: ``Noise`` and ``Sum``.

    Give exactly one of ``snr_db``, ``power`` or ``temperature``.

    Parameters
    ----------
    snr_db : float, optional
        Signal-to-noise ratio / dB, for a signal of ``signal_power``.
    power : float, optional
        Noise power over the whole simulated band.
    temperature : float, optional
        Noise temperature / K: power ``k T B`` with ``B = fs`` for complex
        baseband, ``fs / 2`` for real signals. Needs ``fs``.
    signal_power : float, default 1.0
    bandwidth : float, optional
        Bandwidth / Hz that ``snr_db`` refers to, e.g. 3 kHz for voice at
        ``fs = 48 kHz``; the noise in the whole band is ``fs / bandwidth``
        times larger. Needs ``fs``. Default: the whole band.
    complex : bool, default False
        Circular complex noise, for complex baseband.
    width : int, optional
        Channels of a vector signal.
    seed : int, default 0
    fs : float, optional
        Sample rate / Hz, for ``bandwidth`` or ``temperature``, or to call
        it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``noise`` (``Noise``), ``add``
        (``Sum``).

    Raises
    ------
    ValueError
        Not exactly one of ``snr_db``, ``power``, ``temperature``, or no
        ``fs`` where it is needed.

    Examples
    --------
    >>> import numpy as np
    >>> import sigline as sl
    >>> ch = sl.awgn(snr_db=20.0, complex=True, fs=1.0)
    >>> y = ch(np.ones(100_000, complex))
    >>> round(float(np.mean(np.abs(y - 1) ** 2)), 3)
    0.01
    """
    given = [v is not None for v in (snr_db, power, temperature)]
    if sum(given) != 1:
        raise ValueError("give exactly one of snr_db=, power= or temperature=")
    if (bandwidth is not None and snr_db is not None or temperature is not None) and (
        fs is None
    ):
        raise ValueError("bandwidth= and temperature= need fs=")
    if snr_db is not None:
        power = signal_power / 10 ** (snr_db / 10)
        if bandwidth is not None:
            power *= fs / bandwidth
    elif temperature is not None:
        power = BOLTZMANN * temperature * (fs if complex else fs / 2)
    noise = Noise(np.sqrt(power), seed, complex, width, fs=fs, name="noise")
    add = _adder(fs)
    noise >> add["x1"]
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: f"P = {noise.std**2:.3g}",
    )


# (b, a) shaping white noise; slopes of the power spectral density
COLORS = {
    # paul kellet's three poles approximating -10 dB/decade (1/f)
    "pink": (
        [0.049922035, -0.095993537, 0.050612699, -0.004408786],
        [1.0, -2.494956002, 2.017265875, -0.522189400],
    ),
    "brown": ([1.0], [1.0, -0.999]),  # 1/f^2, a leaky integrator
    "blue": ([1.0, -0.5], [1.0]),  # rising, about +f at low frequencies
    "violet": ([1.0, -1.0], [1.0]),  # +f^2, a difference
}


def _noise_gain(b, a):
    # sum of |h|^2: output power for white input of unit power
    h = sig.lfilter(b, a, np.r_[1.0, np.zeros(2**16 - 1)])
    return float(np.sum(h**2))


def colored_noise(
    power, color="pink", *, complex=False, width=None, seed=0, fs=None, name=None
):
    """Add coloured Gaussian noise: ``Noise``, a shaping ``IIR`` and ``Sum``.

    Parameters
    ----------
    power : float
        Noise power over the whole band, in steady state; the filter starts
        empty, so ``brown`` noise needs about a thousand samples to settle.
    color : {"pink", "brown", "blue", "violet"}, default "pink"
        Spectral slope: ``1/f``, ``1/f^2``, rising, ``f^2``; filters in
        :data:`COLORS`. I and Q are shaped alike and independently.
    complex : bool, default False
    width : int, optional
        Channels of a vector signal.
    seed : int, default 0
    fs : float, optional
        Sample rate / Hz, to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``noise`` (``Noise``), ``shape``
        (``IIR``), ``add`` (``Sum``).

    Raises
    ------
    ValueError
        Unknown ``color``.
    """
    if color not in COLORS:
        raise ValueError(f"color must be one of {sorted(COLORS)}, not {color!r}")
    b, a = COLORS[color]
    std = np.sqrt(power / _noise_gain(b, a))
    noise = Noise(std, seed, complex, width, fs=fs, name="noise")
    shape = IIR(b=b, a=a, fs=fs, name="shape")
    add = _adder(fs)
    noise >> shape >> add["x1"]
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: f"{color}, P = {power:.3g}",
    )


def phase_noise(linewidth=None, *, dbc_hz=None, offset=None, seed=0, fs, name=None):
    """Oscillator phase noise: a random-walk phase multiplied onto the signal.

    The phase is a Wiener process, as of a free-running oscillator: steps
    ``N(0, 2 pi linewidth / fs)`` per sample, summed. Its spectrum is a
    Lorentzian line of full width ``linewidth``, falling as ``1/f^2``:
    ``L(f) = linewidth / (2 pi f^2)`` well outside the line.

    Parameters
    ----------
    linewidth : float, optional
        Line width / Hz.
    dbc_hz, offset : float, optional
        Instead of ``linewidth``: the level ``L(offset)`` / (dBc/Hz) at
        ``offset`` / Hz.
    seed : int, default 0
    fs : float
        Sample rate / Hz; the step size depends on it.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x`` (complex baseband), outputs ``y`` and ``phi`` (the
        phase / rad); members ``noise`` (``Noise``), ``phase`` (``IIR``
        summing the steps), ``rotate`` (``FuncChunk``), ``mix`` (``Product``).

    Raises
    ------
    ValueError
        Neither ``linewidth`` nor ``dbc_hz`` with ``offset``.
    """
    if linewidth is None:
        if dbc_hz is None or offset is None:
            raise ValueError("give linewidth= or dbc_hz= with offset=")
        linewidth = 2 * np.pi * offset**2 * 10 ** (dbc_hz / 10)
    noise = Noise(np.sqrt(2 * np.pi * linewidth / fs), seed, fs=fs, name="noise")
    phase = IIR(b=[1.0], a=[1.0, -1.0], fs=fs, name="phase")
    rotate = FuncChunk(expj, fs=fs, name="rotate")
    mix = Product(fs=fs, name="mix")
    mix.same_kind_inputs = True  # a real signal would silently turn complex
    noise >> phase >> rotate >> mix["x1"]
    return Subsystem(
        inputs={"x": mix["x0"]},
        outputs={"y": mix, "phi": phase},
        name=name,
        info=lambda: f"Δν {si(noise.std**2 * fs / (2 * np.pi), 'Hz')}",
    )


def impulsive_noise(
    power,
    rate=None,
    *,
    p=None,
    duration=0.0,
    complex=False,
    width=None,
    seed=0,
    fs=None,
    name=None,
):
    """Add noise in bursts starting at random: ignition, switching, lightning.

    Gaussian noise of ``power`` while a pulse is on (Bernoulli-Gaussian).

    Parameters
    ----------
    power : float
        Noise power while on; the mean is ``power`` times the fraction of
        time on, ``1 - (1 - p)^n`` for pulses of ``n`` samples.
    rate : float, optional
        Mean pulse starts per second.
    p : float, optional
        Start probability per sample, instead of ``rate``.
    duration : float, default 0.0
        Pulse length / s; at least one sample.
    complex : bool, default False
    width : int, optional
        Channels of a vector signal; all share the pulses.
    seed : int, default 0
        The noise uses ``seed``, the pulses ``seed + 1``.
    fs : float, optional
        Sample rate / Hz, to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``noise`` (``Noise``), ``pulses``
        (``RandomPulses``), ``gate`` (``Product``), ``add`` (``Sum``).
    """
    noise = Noise(np.sqrt(power), seed, complex, width, fs=fs, name="noise")
    pulses = RandomPulses(rate, p, duration, seed + 1, fs=fs, name="pulses")
    gate = Product(fs=fs, name="gate")
    add = _adder(fs)
    noise >> gate["x0"]
    pulses >> gate["x1"]
    gate >> add["x1"]
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: f"{pulses.info()}, P = {noise.std**2:.3g}",
    )


def burst_noise(
    p_gb,
    p_bg,
    power_bad,
    power_good=0.0,
    *,
    complex=False,
    width=None,
    seed=0,
    fs=None,
    name=None,
):
    """Add noise whose power switches between a good and a bad state.

    The Gilbert-Elliott model: per sample the state goes good to bad with
    ``p_gb``, bad to good with ``p_bg``. Bursts last ``1 / p_bg`` samples
    on average; the time in bursts is ``p_gb / (p_gb + p_bg)``.

    Parameters
    ----------
    p_gb, p_bg : float
        Transition probabilities per sample.
    power_bad : float
        Noise power in the bad state.
    power_good : float, default 0.0
        Noise power in the good state.
    complex : bool, default False
    width : int, optional
        Channels of a vector signal; all share the state.
    seed : int, default 0
        The noise uses ``seed``, the state ``seed + 1``.
    fs : float, optional
        Sample rate / Hz, to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, outputs ``y`` and ``bad`` (the state, 0 or 1); members
        ``state`` (``MarkovChain``), ``level`` (``FuncChunk``: the standard
        deviation per state), ``noise`` (``Noise`` of unit power), ``gate``
        (``Product``), ``add`` (``Sum``).
    """
    P = [[1 - p_gb, p_gb], [p_bg, 1 - p_bg]]
    state = MarkovChain(P, seed=seed + 1, fs=fs, name="state")
    stds = np.sqrt([power_good, power_bad])
    level = FuncChunk(table, params={"points": stds}, fs=fs, name="level")
    noise = Noise(1.0, seed, complex, width, fs=fs, name="noise")
    gate = Product(fs=fs, name="gate")
    add = _adder(fs)
    state >> level >> gate["x1"]
    noise >> gate["x0"]
    gate >> add["x1"]
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add, "bad": state},
        name=name,
        info=lambda: f"bursts ~{1 / state.P[1, 0]:.0f} samples",
    )


def _amp(power, complex):
    # amplitude of a carrier of this power: |a|^2 complex, a^2 / 2 real
    return np.sqrt(power if complex else 2 * power)


def _added(src, fs):
    # a sum adding src to the input; the caller makes the subsystem, so it
    # takes the caller's name as its type
    add = _adder(fs)
    src >> add["x1"]
    return add


def tone_interferer(
    f, power, drift=0.0, phase=0.0, *, complex=False, fs=None, name=None
):
    """Add a carrier: a CW interferer, optionally drifting.

    Parameters
    ----------
    f : float
        Frequency / Hz; for complex baseband the offset from the centre.
    power : float
        Carrier power.
    drift : float, default 0.0
        Frequency drift / (Hz/s).
    phase : float, default 0.0
        Start phase / rad.
    complex : bool, default False
        A complex exponential, else a cosine.
    fs : float, optional
        Sample rate / Hz, to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``tone`` (``Chirp``), ``add``
        (``Sum``).
    """
    tone = Chirp(
        f, drift, None, _amp(power, complex), phase, complex, fs=fs, name="tone"
    )
    add = _added(tone, fs)
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: f"CW {tone.info()}",
    )


def chirp_interferer(f0, f1, period, power, *, complex=False, fs=None, name=None):
    """Add a swept jammer: from ``f0`` to ``f1`` every ``period``, then again.

    Parameters
    ----------
    f0, f1 : float
        Start and end frequency / Hz.
    period : float
        Sweep time / s.
    power : float
    complex : bool, default False
    fs : float, optional
        Sample rate / Hz, to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``sweep`` (``Chirp``), ``add``
        (``Sum``).
    """
    rate = (f1 - f0) / period
    amp = _amp(power, complex)
    sweep = Chirp(f0, rate, period, amp, complex=complex, fs=fs, name="sweep")
    add = _added(sweep, fs)
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: sweep.info(),
    )


def pulsed_interferer(
    f, power, width, period, delay=0.0, *, complex=False, fs=None, name=None
):
    """Add a pulsed carrier, as from a radar: on for ``width`` every ``period``.

    Parameters
    ----------
    f : float
        Carrier frequency / Hz.
    power : float
        Peak power, while on; the mean is ``power width / period``.
    width, period : float
        Pulse length and repetition time / s.
    delay : float, default 0.0
        Start of the first pulse / s.
    complex : bool, default False
    fs : float, optional
        Sample rate / Hz, to call it alone.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``carrier`` (``Sine``), ``pulses``
        (``Pulses``), ``gate`` (``Product``), ``add`` (``Sum``).
    """
    carrier = Sine(f, _amp(power, complex), complex=complex, fs=fs, name="carrier")
    pulses = Pulses(width, period, delay, fs=fs, name="pulses")
    gate = Product(fs=fs, name="gate")
    carrier >> gate["x0"]
    pulses >> gate["x1"]
    add = _added(gate, fs)
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: f"{si(carrier.f, 'Hz')}, {pulses.info()}",
    )


def modulated_interferer(
    f,
    power,
    sps,
    points=None,
    beta=0.35,
    span=10,
    *,
    complex=False,
    seed=0,
    fs,
    name=None,
):
    """Add another transmitter's signal: co- or adjacent-channel traffic.

    Built like a wanted signal: random bits, ``mapper``, ``pulse_shaper``
    (root-raised cosine) and a ``mixer`` to ``f``, at ``fs / sps`` symbols
    per second.

    Parameters
    ----------
    f : float
        Centre frequency / Hz.
    power : float
    sps : int
        Samples per symbol.
    points : array_like, optional
        Constellation; default QPSK, ``psk(4)``.
    beta : float, default 0.35
        Roll-off; the signal is ``(1 + beta) fs / sps`` wide.
    span : int, default 10
        Pulse length / symbols.
    complex : bool, default False
        Complex baseband, else the real part (of the same power).
    seed : int, default 0
    fs : float
        Sample rate / Hz of the signal it is added to.
    name : str, optional

    Returns
    -------
    Subsystem
        Input ``x``, output ``y``; members ``bits`` (``RandomBits``), ``map``
        (``mapper``), ``shape`` (``pulse_shaper``), ``shift`` (``mixer``),
        ``real`` (``FuncChunk``, real signals only), ``level`` (``Gain``),
        ``add`` (``Sum``).
    """
    points = psk(4) if points is None else np.asarray(points)
    k = int(np.log2(len(points)))
    bits = RandomBits(seed, fs=fs / sps * k, name="bits")
    shape = pulse_shaper(sps, beta, span, fs=fs / sps, name="shape")
    shift = mixer(f, "exp", fs=fs, name="shift")
    # unit-energy pulses every sps samples: power 1 / sps per unit symbol
    level = Gain(_amp(power, complex) * np.sqrt(sps), fs=fs, name="level")
    bits >> mapper(points, fs=fs / sps * k, name="map") >> shape >> shift
    if complex:
        shift >> level
    else:
        shift >> FuncChunk(np.real, fs=fs, name="real") >> level
    add = _added(level, fs)
    return Subsystem(
        inputs={"x": add["x0"]},
        outputs={"y": add},
        name=name,
        info=lambda: f"{len(points)} points, {sps} sps at {si(shift.lo.f, 'Hz')}",
    )
