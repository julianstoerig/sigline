"""Hardware sources and sinks: RTL-SDR and sound cards.

Drivers are optional extras, imported when a diagram opens the block:
``pip install "sigline[rtlsdr]"`` (pyrtlsdr) and ``pip install
"sigline[audio]"`` (sounddevice). The sample rate is the block's ``fs``.
"""

from __future__ import annotations

import importlib
import threading
import time

import numpy as np

from .streams import FromQueue, ToQueue

__all__ = ["AudioIn", "AudioOut", "RtlSdrSource"]


def _driver(module, extra):
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"{module} is missing; install it with pip install 'sigline[{extra}]'"
        ) from err


class RtlSdrSource(FromQueue):
    """Complex baseband samples from an RTL-SDR receiver.

    ``center_freq`` and ``gain`` apply to the open device at once; change
    them with ``Diagram.set`` during a background run.

    Parameters
    ----------
    center_freq : float
        Tuning frequency / Hz.
    gain : float or "auto", default "auto"
        Tuner gain / dB.
    ppm : int, default 0
        Frequency correction / ppm.
    device_index : int, default 0
    read_size : int, default 2**16
        Samples per driver callback.
    timeout : float or None, default None
        See :class:`FromQueue`; ``None`` lets the receiver pace the diagram.
    capacity : int, default 2**22
        Buffered samples.
    """

    def __init__(
        self,
        center_freq,
        gain="auto",
        ppm=0,
        device_index=0,
        read_size=2**16,
        timeout=None,
        capacity=2**22,
        **kw,
    ):
        self._sdr = None
        self._center_freq, self._gain = center_freq, gain
        self.ppm, self.device_index, self.read_size = ppm, device_index, read_size
        super().__init__(timeout=timeout, capacity=capacity, dtype=complex, **kw)

    @property
    def center_freq(self) -> float:
        """Tuning frequency / Hz."""
        return self._center_freq

    @center_freq.setter
    def center_freq(self, f):
        self._center_freq = f
        if self._sdr is not None:
            self._sdr.center_freq = f

    @property
    def gain(self):
        """Tuner gain / dB, or ``"auto"``."""
        return self._gain

    @gain.setter
    def gain(self, g):
        self._gain = g
        if self._sdr is not None:
            self._sdr.gain = g

    def open(self):
        """Open the receiver and start reading in a driver thread."""
        super().open()
        rtlsdr = _driver("rtlsdr", "rtlsdr")
        sdr = rtlsdr.RtlSdr(device_index=self.device_index)
        sdr.sample_rate = self.fs
        sdr.center_freq = self._center_freq
        sdr.gain = self._gain
        if self.ppm:
            sdr.freq_correction = self.ppm
        self._sdr = sdr

        def read():
            sdr.read_samples_async(lambda s, ctx: self.push(s), self.read_size)

        self._reader = threading.Thread(target=read, name="sigline-rtlsdr", daemon=True)
        self._reader.start()

    def close(self):
        """Stop reading and release the receiver."""
        sdr, self._sdr = self._sdr, None
        if sdr is not None:
            sdr.cancel_read_async()
            self._reader.join(timeout=2.0)
            sdr.close()


class AudioIn(FromQueue):
    """Samples from a sound card input.

    Parameters
    ----------
    channels : int, default 1
        One channel gives a 1-D signal, more a vector signal.
    device : int or str, optional
        Sound device; default: the system default.
    blocksize : int, default 0
        Frames per callback; 0 lets the driver choose.
    timeout : float or None, default None
        See :class:`FromQueue`; ``None`` lets the card pace the diagram.
    capacity : int, default 2**18

    Attributes
    ----------
    device_overflows : int
        Callbacks in which the card reported lost input.
    """

    def __init__(
        self, channels=1, device=None, blocksize=0, timeout=None, capacity=2**18, **kw
    ):
        self.channels, self.device, self.blocksize = channels, device, blocksize
        self._stream = None
        self.device_overflows = 0
        super().__init__(timeout=timeout, capacity=capacity, **kw)

    def open(self):
        """Open and start the input stream."""
        super().open()
        sd = _driver("sounddevice", "audio")

        def callback(indata, frames, time, status):
            if status and status.input_overflow:
                self.device_overflows += 1
            self.push(indata[:, 0] if self.channels == 1 else indata)

        self._stream = sd.InputStream(
            samplerate=self.fs,
            channels=self.channels,
            device=self.device,
            blocksize=self.blocksize,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def close(self):
        """Stop and close the input stream."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()


class AudioOut(ToQueue):
    """Play the input on a sound card; the card paces the diagram.

    Parameters
    ----------
    channels : int, default 1
        A 1-D signal is played on all channels.
    device : int or str, optional
    blocksize : int, default 0
    capacity : int, default 8192
        Buffered samples, i.e. the most output latency.
    prefill : int, optional
        Play silence until this many samples are buffered, so jitter does not
        starve the card; default ``capacity // 2``. Keep it above one diagram
        chunk plus one card block, or the card underruns between chunks.

    Attributes
    ----------
    device_underflows : int
        Callbacks in which the card reported missing output.
    """

    def __init__(
        self, channels=1, device=None, blocksize=0, capacity=8192, prefill=None, **kw
    ):
        self.channels, self.device, self.blocksize = channels, device, blocksize
        self.prefill = capacity // 2 if prefill is None else prefill
        self._stream, self._draining = None, False
        self.device_underflows = 0
        super().__init__(capacity=capacity, block=True, **kw)

    def open(self):
        """Open and start the output stream."""
        super().open()
        sd = _driver("sounddevice", "audio")

        primed = False
        self._draining = False

        def callback(outdata, frames, time, status):
            nonlocal primed
            if status and status.output_underflow:
                self.device_underflows += 1
            primed = primed or self._draining or len(self.buffer) >= self.prefill
            outdata[:] = 0
            if not primed:
                return
            # at the end, silence after the last sample is not an underrun
            k = min(frames, len(self.buffer)) if self._draining else frames
            y = np.real(self.pull(k))
            outdata[:k] = (y[:, None] if y.ndim == 1 else y).astype(outdata.dtype)

        self._stream = sd.OutputStream(
            samplerate=self.fs,
            channels=self.channels,
            device=self.device,
            blocksize=self.blocksize,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def close(self):
        """Play what is buffered, then stop and close the output stream."""
        stream, self._stream = self._stream, None
        if stream is not None:
            self._draining = True
            deadline = time.monotonic() + 1.0 + self.buffer.capacity / self.fs
            while len(self.buffer) and time.monotonic() < deadline:
                time.sleep(0.005)
            stream.stop()
            stream.close()
