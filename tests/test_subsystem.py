import matplotlib
import numpy as np
import pytest
import scipy.signal as sig

import sigline as sl
from sigline.testing import assert_chunk_invariant

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FS = 48e3
HP = sig.firwin(101, 10e3, pass_zero=False, fs=FS)


def upconverter(fc=10e3):
    mix = sl.Product(name="mix")
    sl.Sine(fc, name="lo") >> mix["x1"]
    hpf = mix >> sl.FIR(HP, name="hpf")
    return sl.Subsystem(inputs={"x": mix["x0"]}, outputs={"y": hpf})


def comb(g=0.5, n=4):
    add = sl.Sum(name="add")
    add >> sl.Delay(n, name="z") >> sl.Gain(g, name="g") >> add["x1"]
    return sl.Subsystem(inputs={"x": add["x0"]}, outputs={"y": add})


def by_hand(x, filt, fc=10e3, chunk=100):
    src, mix, sc = sl.FromArray(x), sl.Product(), sl.Scope()
    src >> mix["x0"]
    sl.Sine(fc) >> mix["x1"]
    mix >> filt >> sc
    sl.Diagram(sc, fs=FS).run(n=len(x), chunk=chunk)
    return sc.data


@pytest.fixture
def x():
    return np.random.default_rng(0).standard_normal(3000)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# unit


def test_ports_members_and_names():
    up = upconverter()
    assert up.inputs == ("x",) and up.outputs == ("y",)
    assert up.type == "upconverter" and up.name.startswith("upconverter")
    assert [m.name for m in up.members] == [
        f"{up.name}/{s}" for s in ("mix", "lo", "hpf")
    ]
    assert up.lo.f == 10e3 and up["x"].block is up.mix and up["y"].block is up.hpf
    assert up.info() == "3 blocks"
    with pytest.raises(AttributeError, match="no member"):
        _ = up.nope


def test_counter_names_each_copy():
    a, b = upconverter(), upconverter()
    assert a.name != b.name and a.hpf.name != b.hpf.name


def test_invalid_subsystems():
    with pytest.raises(ValueError, match="at least one"):
        sl.Subsystem()
    g = sl.Gain(1.0)
    sl.Const(1.0) >> g
    with pytest.raises(ValueError, match="already connected"):
        sl.Subsystem(inputs={"x": g}, outputs={"y": g})
    up = upconverter()
    with pytest.raises(ValueError, match="is not in"):
        up.replace(sl.Gain(1.0), sl.Gain(2.0))
    used = sl.Gain(1.0)
    sl.Const(1.0) >> used
    with pytest.raises(ValueError, match="already connected"):
        up.replace(up.hpf, used)
    with pytest.raises(ValueError, match="no port"):
        up.replace(up.mix, sl.Gain(1.0))  # one input for two


# integration


def test_feedback_inside_and_loop_around(x):
    src, sc = sl.FromArray(x), sl.Scope()
    src >> comb(0.5, 4) >> sc
    sl.Diagram(sc, fs=1.0).run(n=len(x), chunk=4)
    ref = sig.lfilter([1.0], [1, 0, 0, 0, -0.5], x)
    np.testing.assert_allclose(sc.data, ref, atol=1e-12)

    # a loop through a subsystem, closed outside by a delay
    add, sc2 = sl.Sum(), sl.Scope()
    sl.FromArray(x) >> add["x0"]
    add >> sl.Subsystem(inputs={"x": (g := sl.Gain(0.5))}, outputs={"y": g}) >> sc2
    g >> sl.Delay(1) >> add["x1"]
    sl.Diagram(sc2, fs=1.0).run(n=len(x), chunk=1)
    np.testing.assert_allclose(sc2.data, sig.lfilter([0.5], [1, -0.5], x), atol=1e-12)


def test_nested_subsystems_keep_paths_and_chain(x):
    def two_stage():
        a, b = upconverter(5e3), upconverter(12e3)
        a >> b
        return sl.Subsystem(inputs={"x": a}, outputs={"y": b})

    t, sc = two_stage(), sl.Scope()
    sl.FromArray(x) >> t >> sc
    d = sl.Diagram(sc, fs=FS)
    d.run(n=len(x), chunk=100)
    inner = t.members[0]
    assert inner.hpf.name == f"{t.name}/{inner.name.split('/')[-1]}/hpf"
    assert d[inner.lo.name] is inner.lo
    # equal to the same two stages without subsystems
    src, sc2 = sl.FromArray(x), sl.Scope()
    m1, m2 = sl.Product(), sl.Product()
    src >> m1["x0"]
    sl.Sine(5e3) >> m1["x1"]
    m1 >> sl.FIR(HP) >> m2["x0"]
    sl.Sine(12e3) >> m2["x1"]
    m2 >> sl.FIR(HP) >> sc2
    sl.Diagram(sc2, fs=FS).run(n=len(x), chunk=100)
    np.testing.assert_array_equal(sc.data, sc2.data)


def test_probe_and_diagram_from_a_subsystem(x):
    src, up = sl.FromArray(x), upconverter()
    src >> up >> sl.Scope()
    d = sl.Diagram(up, fs=FS)  # a subsystem stands for its blocks
    p = d.probe(up)
    d.run(n=len(x), chunk=100)
    np.testing.assert_array_equal(p.data, by_hand(x, sl.FIR(HP)))


# comparison


def test_as_one_block_equals_by_hand_and_costs_nothing(x):
    src, up, sc = sl.FromArray(x), upconverter(), sl.Scope()
    src >> up >> sc
    d = sl.Diagram(sc, fs=FS)
    d.run(n=len(x), chunk=100)
    np.testing.assert_array_equal(sc.data, by_hand(x, sl.FIR(HP)))
    assert len(d.blocks) == 5  # source, scope, three inside: nothing added


def test_replace_equals_by_hand_and_keeps_lines(x):
    sos = sig.butter(6, 10e3, "highpass", fs=FS, output="sos")
    up = upconverter()
    old = up.hpf
    new = up.replace("hpf", sl.IIR(sos=sos, name="hpf"))
    assert new is up.hpf and new.name == f"{up.name}/hpf" and up["y"].block is new
    assert not old._sources and not old._sinks
    src, sc = sl.FromArray(x), sl.Scope()
    src >> up >> sc
    sl.Diagram(sc, fs=FS).run(n=len(x), chunk=100)
    np.testing.assert_array_equal(sc.data, by_hand(x, sl.IIR(sos=sos)))


def test_chunk_invariant(x):
    assert_chunk_invariant(upconverter, x, fs=FS)


# theory


def test_upconverted_tone_lands_at_the_sum_frequency():
    n = 48_000
    t = np.arange(n) / FS
    src, up, sc = sl.FromArray(np.cos(2 * np.pi * 3e3 * t)), upconverter(), sl.Scope()
    src >> up >> sc
    sl.Diagram(sc, fs=FS).run(n=n, chunk=1024)
    spec = np.abs(np.fft.rfft(sc.data[200:]))
    f = np.fft.rfftfreq(n - 200, 1 / FS)
    # cos a cos b: sum and difference; the high-pass keeps 13 kHz, not 7 kHz
    assert abs(f[np.argmax(spec)] - 13e3) < 2
    lower = spec[np.argmin(np.abs(f - 7e3))] / spec.max()
    assert lower < 10 ** (-40 / 20)


# drawing


def test_drawn_as_one_box_and_inside():
    src, up, sc = sl.FromArray(np.zeros(10), name="src"), upconverter(), sl.Scope("out")
    src >> up >> sc
    d = sl.Diagram(sc, fs=FS)
    texts = {t.get_text() for t in d.draw().texts}
    assert {up.name, "upconverter", "3 blocks"} <= texts and "Product" not in texts
    inside = {t.get_text() for t in up.draw().texts}
    assert {"mix", "lo", "hpf", "Product"} <= inside and "src" not in inside
    assert up.name in d.to_mermaid() and "hpf" not in d.to_dot()


# inputs feeding several inner ports


def iq_mixer(fc=5e3):
    mi, mq = sl.Product(name="mix i"), sl.Product(name="mix q")
    sl.Sine(fc, name="cos") >> mi["x1"]
    sl.Sine(fc, phase=-np.pi / 2, name="sin") >> mq["x1"]
    return sl.Subsystem(inputs={"x": [mi["x0"], mq["x0"]]}, outputs={"i": mi, "q": mq})


def test_one_input_feeds_several_inner_ports(x):
    src, si, sq = sl.FromArray(x), sl.Scope(), sl.Scope()
    iq = iq_mixer()
    assert src >> iq is iq and isinstance(iq["x"], sl.PortGroup)
    iq["i"] >> si
    iq["q"] >> sq
    sl.Diagram(si, fs=FS).run(n=len(x), chunk=128)
    t = np.arange(len(x)) / FS
    np.testing.assert_allclose(si.data, x * np.cos(2 * np.pi * 5e3 * t), atol=1e-12)
    np.testing.assert_allclose(sq.data, x * np.sin(2 * np.pi * 5e3 * t), atol=1e-12)
    # drawn as one line into x, two lines out
    d = sl.Diagram(si, fs=FS)
    from sigline.drawing import view

    _, edges = view(d)
    assert sorted((e.sport, e.dport) for e in edges) == [
        ("i", "x"),
        ("q", "x"),
        ("y", "x"),
    ]


def test_tuple_feeds_a_two_input_subsystem(x):
    add = sl.Sum(name="add")
    s = sl.Subsystem(inputs={"a": add["x0"], "b": add["x1"]}, outputs={"y": add})
    sc = sl.Scope()
    (sl.FromArray(x), sl.FromArray(2 * x)) >> s >> sc
    sl.Diagram(sc, fs=1.0).run(n=len(x))
    np.testing.assert_allclose(sc.data, 3 * x)


def test_replace_inside_a_fanned_out_input(x):
    iq = iq_mixer()
    new = iq.replace("mix q", sl.Product("*/", name="mix q"))  # divides instead
    assert iq["x"][1].block is new and iq["q"].block is new
    src, sq = sl.FromArray(x), sl.Scope()
    src >> iq
    iq["q"] >> sq
    sl.Diagram(sq, fs=FS).run(n=8, chunk=4)
    t = np.arange(8) / FS
    with np.errstate(divide="ignore"):
        np.testing.assert_allclose(
            sq.data[1:], x[1:8] / np.sin(2 * np.pi * 5e3 * t[1:])
        )


def test_inside_drawing_marks_the_ports():
    ax = iq_mixer().draw()
    marks = [
        t.get_text()
        for t in ax.texts
        if t.get_color() == "#1f4e79" and t.get_weight() == "bold"
    ]
    assert sorted(marks) == ["i", "q", "x", "x"]  # x feeds both products
    ax2 = sl.Diagram(*_wired(iq_mixer()), fs=FS).draw()
    assert not [
        t for t in ax2.texts if t.get_weight() == "bold" and t.get_text() == "x"
    ]


def _wired(sub):
    sl.Sine(1e3) >> sub
    outs = [sl.Scope() for _ in sub.outputs]
    for k, sc in zip(sub.outputs, outs, strict=True):
        sub[k] >> sc
    return outs


# diagrams follow edits made after they were built


def test_replace_after_the_diagram_was_built(x):
    sos = sig.butter(6, 10e3, "highpass", fs=FS, output="sos")
    src, up, sc = sl.FromArray(x), upconverter(), sl.Scope()
    src >> up >> sc
    d = sl.Diagram(sc, fs=FS)
    p = d.probe(up.hpf)
    d.run(n=1000, chunk=100)
    old = up.hpf
    up.replace(up.hpf, sl.IIR(sos=sos, name="hpf"))
    d.reset().run(n=len(x), chunk=100)
    assert up.hpf in d.blocks and old not in d.blocks
    assert (old, "y") not in d._probes  # its probe went with it
    assert p not in [sc for lst in d._probes.values() for sc in lst]
    np.testing.assert_array_equal(sc.data, by_hand(x, sl.IIR(sos=sos)))


def test_blocks_connected_later_join_the_diagram(x):
    src, g, sc = sl.FromArray(x), sl.Gain(2.0), sl.Scope()
    src >> g >> sc
    d = sl.Diagram(sc, fs=1.0)
    late = sl.Scope("late")
    g >> late
    d.run(n=10)
    assert late in d.blocks
    np.testing.assert_array_equal(late.data, 2 * x[:10])


# called alone, like a block


def test_called_alone_equals_a_diagram_and_keeps_state(x):
    up = upconverter()
    y = np.concatenate([up(x[:1000], fs=FS), up(x=x[1000:], fs=FS)])
    np.testing.assert_allclose(y, by_hand(x, sl.FIR(HP)), atol=1e-12)
    assert up.lo.fs is None and not up.hpf._sinks  # nothing left behind


def test_called_alone_takes_the_rate_of_its_blocks():
    mix = sl.Product(name="mix")
    sl.Sine(0.25, fs=1.0, name="lo") >> mix["x1"]
    sub = sl.Subsystem(inputs={"x": mix["x0"]}, outputs={"y": mix})
    np.testing.assert_allclose(sub(np.ones(4)), [1, 0, -1, 0], atol=1e-15)
    with pytest.raises(ValueError, match="no sample rate"):
        upconverter()(np.ones(4))


def test_called_alone_refuses_when_connected(x):
    up = upconverter()
    sl.FromArray(x) >> up
    with pytest.raises(ValueError, match="connected"):
        up(x, fs=FS)


def test_source_subsystem_takes_a_length():
    def two_tones():
        s = sl.Sum(name="sum")
        sl.Sine(1.0, name="a") >> s["x0"]
        sl.Sine(2.0, name="b") >> s["x1"]
        return sl.Subsystem(outputs={"y": s})

    n = np.arange(8)
    ref = np.cos(2 * np.pi * n / 8) + np.cos(4 * np.pi * n / 8)
    np.testing.assert_allclose(two_tones()(8, fs=8.0), ref, atol=1e-12)


def test_info_function_follows_parameters():
    mix, lo = sl.Product(name="mix"), sl.Sine(10e3, name="lo")
    lo >> mix["x1"]
    sub = sl.Subsystem(
        inputs={"x": mix["x0"]}, outputs={"y": mix}, info=lambda: f"{lo.f / 1e3:g} kHz"
    )
    assert sub.info() == "10 kHz"
    sub.lo.f = 12e3
    assert sub.info() == "12 kHz"


def test_text_dot_and_mermaid_of_the_inside():
    up = upconverter()
    text = up.to_text()
    assert text.startswith(f"{up.name} (upconverter)") and "fs =" not in text
    assert "  x ──▶ mix[x0]" in text and "  hpf[y] ──▶ y" in text
    assert str(up) == text
    assert "hpf" in up.to_dot() and "lo -->|x1| mix" in up.to_mermaid()
    assert up._repr_svg_().lstrip().startswith("<?xml")
