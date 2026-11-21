from fractions import Fraction

import matplotlib
import numpy as np
import pytest
import scipy.signal as sig

from _blocks import Split
from sigline import FIR, Block, Diagram, FromArray, Gain, Noise, Scope

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAPS = sig.firwin(31, 0.2)


def chain():
    src, g, lp, sc = (
        Noise(1.0, seed=2),
        Gain(2.0, name="g"),
        FIR(TAPS, name="lp"),
        Scope(),
    )
    src >> g >> lp >> sc
    return Diagram(sc, fs=1000.0), sc


# unit: targets and errors


def test_probe_by_block_port_name_and_name_port():
    d, sc = chain()
    probes = [
        d.probe(d["g"]),
        d.probe(d["lp"].out("y")),
        d.probe("lp"),
        d.probe("lp:y"),
    ]
    d.run(n=1000, chunk=128)
    for p in probes[1:]:
        np.testing.assert_array_equal(p.data, sc.data)
    assert probes[0].name == "probe:g:y" and probes[2].name == "probe:lp"
    assert probes[0].fs == 1000.0 and probes[0] not in d.blocks


def test_probe_errors():
    d, _ = chain()
    with pytest.raises(ValueError, match="is an input"):
        d.probe(d["lp"].inp("x"))
    with pytest.raises(ValueError, match="not part of this diagram"):
        d.probe(Gain(1.0))
    with pytest.raises(KeyError):
        d.probe("nope")
    s = Split()
    FromArray(np.ones(4)) >> s
    s.out("re") >> Scope()
    d2 = Diagram(s, fs=1.0)
    with pytest.raises(ValueError, match="use 'name:port'"):
        d2.probe(s)
    p = d2.probe(f"{s.name}:im")
    d2.run(n=4)
    np.testing.assert_array_equal(p.data, np.zeros(4))


def test_reset_and_remove_probe():
    d, _ = chain()
    p = d.probe("g")
    d.run(n=100)
    d.reset()
    assert p.data.size == 0
    d.remove_probe(p)
    d.run(n=100)
    assert p.data.size == 0 and not d._probes


# integration: keep / every, multirate


def test_probe_ring_buffer_and_decimation_keep_absolute_time():
    d, sc = chain()
    p = d.probe("lp", keep=100, every=4)
    d.run(n=1000, chunk=64)
    np.testing.assert_allclose(p.data, sc.data[::4][-100:])
    np.testing.assert_allclose(p.t, np.arange(1000)[::4][-100:] / 1000.0)


def test_probe_rate_follows_the_block():
    class Down2(Block):
        resampling_ratio = Fraction(1, 2)

        def step(self, x):
            return x[::2]

    src, dn, sc = FromArray(np.arange(20.0)), Down2(), Scope()
    src >> dn >> sc
    d = Diagram(sc, fs=10.0)
    p = d.probe(dn)
    d.run(n=20, chunk=4)
    assert p.fs == 5.0
    np.testing.assert_allclose(p.t, np.arange(10) / 5.0)


# comparison: a probe equals a scope on a line; probing does not change results


def test_probe_equals_wired_scope_and_leaves_output_unchanged():
    d, sc = chain()
    d.run(n=500, chunk=77)
    ref = sc.data.copy()
    d2, sc2 = chain()
    p = d2.probe("g")
    d2.run(n=500, chunk=77)
    np.testing.assert_array_equal(sc2.data, ref)
    # theory: filtering the probed signal reproduces the output
    np.testing.assert_allclose(sig.lfilter(TAPS, 1, p.data), sc2.data, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 13, 1000])
def test_probe_recording_is_chunk_invariant(chunk):
    d, _ = chain()
    p = d.probe("g", every=3)
    d.run(n=1000, chunk=chunk)
    ref_d, _ = chain()
    ref = ref_d.probe("g", every=3)
    ref_d.run(n=1000, chunk=1000)
    np.testing.assert_array_equal(p.data, ref.data)


# scopes and plot


def test_scopes_and_plot():
    d, sc = chain()
    sc2 = Scope("second")
    d["g"] >> sc2
    d = Diagram(sc, fs=1000.0)
    assert d.scopes() == [sc, sc2]
    d.run(n=200)
    axes = d.plot()
    assert (
        len(axes) == 2
        and axes[-1].get_xlabel() == "t / s"
        and axes[0].get_xlabel() == ""
    )
    plt.close("all")


# any sink as a probe


def test_any_one_input_sink_can_probe():
    class Peak(Block):
        outputs = ()

        def reset(self):
            self.peak = 0.0

        def step(self, x):
            self.peak = max(self.peak, float(np.max(np.abs(x))))
            if self.peak > 3.0:
                self.request_stop("peak above 3")
            return {}

    d, sc = chain()
    pk = d.probe("g", sink=Peak(name="peak"))
    assert pk.fs == 1000.0 and pk.name == "peak"
    d.run(n=5000, chunk=100)
    # theory: the probe saw exactly the gain output, 2 x unit noise
    assert d.stopped == "peak: peak above 3" and pk.peak > 3.0
    d.reset()
    assert pk.peak == 0.0


def test_probe_sink_must_be_a_one_input_sink():
    d, _ = chain()
    with pytest.raises(ValueError, match="one input and no outputs"):
        d.probe("g", sink=Gain(1.0))
