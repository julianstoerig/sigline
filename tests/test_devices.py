"""device blocks against fake drivers that behave like pyrtlsdr / sounddevice."""

import sys
import threading
import time
import types

import numpy as np
import pytest

from sigline import AudioIn, AudioOut, Diagram, Gain, RtlSdrSource, Scope


def iq(k, n):
    return np.exp(2j * np.pi * 0.01 * np.arange(k, k + n))


class FakeRtlSdr:
    instances = []

    def __init__(self, device_index=0):
        self.device_index = device_index
        self.sample_rate = self.center_freq = self.gain = None
        self.freq_correction = 0
        self.cancelled = threading.Event()
        self.closed = False
        FakeRtlSdr.instances.append(self)

    def read_samples_async(self, callback, num_samples):
        k = 0
        while not self.cancelled.is_set():
            callback(iq(k, num_samples), None)
            k += num_samples
            time.sleep(num_samples / self.sample_rate)

    def cancel_read_async(self):
        self.cancelled.set()

    def close(self):
        self.closed = True


class FakeStream:
    def __init__(self, samplerate, channels, device, blocksize, dtype, callback):
        self.fs, self.channels, self.callback = samplerate, channels, callback
        self.blocksize = blocksize or 256
        self.running = threading.Event()
        self.closed = False
        self.k = 0

    def start(self):
        self.running.set()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running.clear()

    def close(self):
        self.closed = True


class FakeInputStream(FakeStream):
    def _loop(self):
        while self.running.is_set():
            n = np.arange(self.k, self.k + self.blocksize)
            data = np.stack(
                [np.sin(0.01 * n + c) for c in range(self.channels)], axis=1
            )
            self.callback(data.astype("float32"), self.blocksize, None, None)
            self.k += self.blocksize
            time.sleep(self.blocksize / self.fs)


class FakeOutputStream(FakeStream):
    played = []

    def _loop(self):
        while self.running.is_set():
            out = np.zeros((self.blocksize, self.channels), dtype="float32")
            self.callback(out, self.blocksize, None, None)
            FakeOutputStream.played.append(out.copy())
            time.sleep(self.blocksize / self.fs)


@pytest.fixture
def drivers(monkeypatch):
    FakeRtlSdr.instances.clear()
    FakeOutputStream.played.clear()
    monkeypatch.setitem(sys.modules, "rtlsdr", types.SimpleNamespace(RtlSdr=FakeRtlSdr))
    sd = types.SimpleNamespace(
        InputStream=FakeInputStream, OutputStream=FakeOutputStream
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sd)


# rtl-sdr


def test_rtlsdr_settings_stream_and_release(drivers):
    src, sc = RtlSdrSource(100e6, gain=20, ppm=3, read_size=1024), Scope()
    src >> sc
    d = Diagram(sc, fs=250_000.0)
    d.start(n=8192, chunk=2048)  # the receiver paces the run
    d.wait(timeout=10)
    dev = FakeRtlSdr.instances[0]
    assert (dev.sample_rate, dev.center_freq, dev.gain, dev.freq_correction) == (
        250_000.0,
        100e6,
        20,
        3,
    )
    np.testing.assert_allclose(sc.data, iq(0, 8192))  # every sample, in order
    assert dev.cancelled.is_set() and dev.closed and src.underruns == 0


def test_rtlsdr_retunes_while_running(drivers):
    src, sc = RtlSdrSource(100e6, read_size=512), Scope()
    src >> sc
    d = Diagram(sc, fs=100_000.0)
    d.start(chunk=1024)
    try:
        time.sleep(0.05)
        d.set(src, center_freq=145.8e6, gain=30)
        dev = FakeRtlSdr.instances[0]
        # applied between chunks: wait for it, slow runners need longer
        deadline = time.monotonic() + 5.0
        while dev.center_freq != 145.8e6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (dev.center_freq, dev.gain) == (145.8e6, 30)
    finally:
        d.stop()
    assert src.center_freq == 145.8e6


def test_missing_driver_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtlsdr", None)
    src, sc = RtlSdrSource(100e6), Scope()
    src >> sc
    with pytest.raises(ImportError, match=r"sigline\[rtlsdr\]"):
        Diagram(sc, fs=1e6).run(n=1)


# sound card


def test_audio_loopback_through_a_diagram(drivers):
    # comparison: what the fake card records reaches the fake card's output,
    # scaled by the gain, sample for sample
    # prefill 2048 > one chunk (512) + one card block (256)
    mic, spk = AudioIn(blocksize=256), AudioOut(blocksize=256, capacity=4096)
    mic >> Gain(0.5) >> spk
    d = Diagram(spk, fs=48_000.0)
    d.start(n=4096, chunk=512)
    d.wait(timeout=10)
    played = np.concatenate(FakeOutputStream.played)[:, 0]
    assert spk.underruns == 0
    # output starts after its latency; the signal starts with sin(0) = 0
    start = int(np.argmax(np.abs(played) > 0)) - 1
    ref = 0.5 * np.sin(0.01 * np.arange(4096)).astype("float32")
    np.testing.assert_allclose(played[start : start + 4096], ref, atol=1e-6)


def test_audio_in_channels_make_vector_signals(drivers):
    mic, sc = AudioIn(channels=2, blocksize=128), Scope()
    mic >> sc
    Diagram(sc, fs=8000.0).start(n=512, chunk=128).wait(timeout=10)
    assert sc.data.shape == (512, 2)
    np.testing.assert_allclose(
        sc.data[:, 1], np.sin(0.01 * np.arange(512) + 1), atol=1e-6
    )
