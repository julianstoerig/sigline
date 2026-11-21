"""Analog modulation: AM, FM, PM and SSB.

Modulators turn a real message into complex baseband; for a real passband
signal follow with ``Mixer(fc, "exp")`` and take the real part. Demodulators
take complex baseband and can low-pass it first to ``channel_width``.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig

from ._format import si
from .core import Block
from .filters import FIR

__all__ = ["AMDemod", "AMMod", "FMDemod", "FMMod"]


class _ChannelFilter:
    # optional low-pass of a two-sided width, built once fs is known

    def _setup_channel(self, channel_width, taps):
        self.channel_width, self.taps = channel_width, int(taps)

    def _reset_channel(self):
        self._lp = None

    def _channel(self, z):
        if self.channel_width is None:
            return z
        if self._lp is None:
            cutoff = self.channel_width / 2
            if not 0 < cutoff < self.fs / 2:
                raise ValueError(
                    f"{self.name}: channel_width must be between 0 and "
                    f"fs = {self.fs:g} Hz"
                )
            self._lp = FIR(sig.firwin(self.taps, cutoff, fs=self.fs))
        return self._lp(z)


class AMMod(Block):
    """Amplitude modulation: complex baseband ``1 + index * x``.

    Parameters
    ----------
    index : float, default 0.5
        Modulation index for a message within ``[-1, 1]``.
    carrier : bool, default True
        ``False`` gives double sideband with suppressed carrier,
        ``index * x``.
    """

    real_inputs = True

    def __init__(self, index=0.5, carrier=True, **kw):
        self.index, self.carrier = index, carrier
        super().__init__(**kw)

    def step(self, x):
        """Return the complex envelope."""
        return (float(self.carrier) + self.index * x).astype(complex)

    def info(self):
        """Return index and kind."""
        return f"m={self.index:g}" + ("" if self.carrier else " DSB-SC")


class AMDemod(_ChannelFilter, Block):
    """Amplitude demodulation of complex baseband.

    ``envelope`` takes ``|z|`` (needs the carrier, phase does not matter);
    ``coherent`` takes ``Re(z)`` (needs carrier phase 0, works without
    carrier). With a carrier, it is removed and the index divided out.

    Parameters
    ----------
    index : float, default 0.5
    carrier : bool, default True
    method : {"envelope", "coherent"}, default "envelope"
    channel_width : float, optional
        Low-pass to this two-sided width / Hz first.
    taps : int, default 101
        Of the channel filter.

    Raises
    ------
    ValueError
        Unknown method, or envelope detection without carrier.
    """

    def __init__(
        self,
        index=0.5,
        carrier=True,
        method="envelope",
        channel_width=None,
        taps=101,
        **kw,
    ):
        if method not in ("envelope", "coherent"):
            raise ValueError("method must be 'envelope' or 'coherent'")
        if method == "envelope" and not carrier:
            raise ValueError("envelope detection needs the carrier; use 'coherent'")
        self.index, self.carrier, self.method = index, carrier, method
        self._setup_channel(channel_width, taps)
        super().__init__(**kw)

    def reset(self):
        """Clear the channel filter."""
        self._reset_channel()

    def step(self, x):
        """Return the message."""
        z = self._channel(x)
        a = np.abs(z) if self.method == "envelope" else np.real(z)
        return (a - float(self.carrier)) / self.index

    def info(self):
        """Return method and channel width."""
        width = f", {si(self.channel_width, 'Hz')}" if self.channel_width else ""
        return self.method + width


TWO_PI = 2 * np.pi


class FMMod(Block):
    """Frequency modulation: ``exp(j phi)``, ``phi`` the integral of the message.

    ``phi[n] = phi[n-1] + 2 pi deviation x[n] / fs``: a message of 1 shifts
    the frequency by ``deviation``.

    Parameters
    ----------
    deviation : float
        Peak frequency deviation / Hz for ``|x| = 1``.
    """

    real_inputs = True

    def __init__(self, deviation, **kw):
        self.deviation = deviation
        super().__init__(**kw)

    def reset(self):
        """Restart at phase 0."""
        self.phi = 0.0

    def step(self, x):
        """Return the complex FM signal."""
        ph = self.phi + np.cumsum(TWO_PI * self.deviation / self.fs * x)
        self.phi = float(ph[-1]) % TWO_PI if len(ph) else self.phi
        return np.exp(1j * ph)

    def info(self):
        """Return the deviation."""
        return f"Δf={si(self.deviation, 'Hz')}"


class FMDemod(_ChannelFilter, Block):
    """Frequency demodulation: quadrature discriminator.

    ``y[n] = angle(z[n] conj(z[n-1])) fs / (2 pi deviation)``, so
    ``FMDemod(FMMod(x)) == x`` while the phase step stays below pi.

    Parameters
    ----------
    deviation : float
        Frequency / Hz that maps to an output of 1.
    channel_width : float, optional
        Low-pass to this two-sided width / Hz first. Carson's rule,
        ``2 (deviation + highest message frequency)``, holds 98 % of the
        power; leave a margin above it, or clipped sidebands distort.
    taps : int, default 101
        Of the channel filter.
    """

    def __init__(self, deviation, channel_width=None, taps=101, **kw):
        self.deviation = deviation
        self._setup_channel(channel_width, taps)
        super().__init__(**kw)

    def reset(self):
        """Clear the channel filter and the previous sample."""
        self._reset_channel()
        self._prev = 1.0 + 0j

    def step(self, x):
        """Return the message."""
        z = self._channel(x)
        if not len(z):
            return np.zeros(0)
        prev = np.concatenate([[self._prev], z[:-1]])
        self._prev = z[-1]
        return np.angle(z * np.conj(prev)) * self.fs / (TWO_PI * self.deviation)

    def info(self):
        """Return deviation and channel width."""
        width = f", {si(self.channel_width, 'Hz')}" if self.channel_width else ""
        return f"Δf={si(self.deviation, 'Hz')}" + width
