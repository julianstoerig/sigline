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
from scipy.constants import k as BOLTZMANN

from .blocks import Sum
from .sources import Noise
from .subsystem import Subsystem

__all__ = ["BOLTZMANN", "awgn"]


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
