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

from .blocks import Sum
from .filters import IIR
from .sources import Noise
from .subsystem import Subsystem

__all__ = ["BOLTZMANN", "COLORS", "awgn", "colored_noise"]


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
