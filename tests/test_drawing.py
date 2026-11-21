import os
import shutil
import subprocess
import sys
from fractions import Fraction

import matplotlib
import numpy as np
import pytest

from _blocks import Add, Ramp, Record, Scale, Split
from sigline import Const, Delay, Diagram, FromArray, Gain, Scope, Sine, Sum
from sigline.drawing import layout

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402


def names(cols):
    return [[n.name for n in c] for c in cols]


# layout


def test_chain_gets_one_column_per_block():
    a, b, c, r = Ramp(name="a"), Scale(name="b"), Scale(name="c"), Record(name="r")
    a >> b >> c >> r
    cols, edges = layout(Diagram(r, fs=1.0))
    assert names(cols) == [["a"], ["b"], ["c"], ["r"]]
    assert len(edges) == 3 and not any(e.feedback for e in edges)


def test_feedback_line_is_marked_and_columns_ignore_it():
    src, add, g, dl, sc = (
        Const(name="src"),
        Sum(name="add"),
        Gain(0.5, name="g"),
        Delay(1, name="z"),
        Scope(name="sc"),
    )
    src >> add["x0"]
    add >> g >> sc
    g >> dl >> add["x1"]
    cols, edges = layout(Diagram(sc, fs=1.0))
    fb = [(e.src.name, e.dst.name) for e in edges if e.feedback]
    assert len(fb) == 1
    col = {n.name: i for i, c in enumerate(cols) for n in c}
    assert col["src"] < col["add"] < col["g"] < col["sc"]
    assert col["z"] > col["g"]  # the delay sits after its driver


def test_sources_sit_next_to_their_consumer():
    a, b, c, add = Ramp(name="a"), Scale(name="b"), Scale(name="c"), Add(name="add")
    late = Ramp(name="late")
    a >> b >> c >> add["a"]
    late >> add["b"]
    cols, _ = layout(Diagram(add, fs=1.0))
    col = {n.name: i for i, c in enumerate(cols) for n in c}
    assert col["late"] == col["add"] - 1


def test_parallel_chains_do_not_cross():
    # two chains into a two-input block: barycenter ordering keeps them apart
    s1, s2 = Ramp(name="s1"), Ramp(name="s2")
    g1, g2 = Scale(name="g1"), Scale(name="g2")
    add = Add(name="add")
    s2 >> g2 >> add["b"]
    s1 >> g1 >> add["a"]
    cols, _ = layout(Diagram(add, fs=1.0))
    pos = {n.name: i for c in cols for i, n in enumerate(c)}
    assert (pos["s1"] < pos["s2"]) == (pos["g1"] < pos["g2"])


def test_nodes_carry_ports_kinds_and_rates():
    sp, sc = Split(name="split"), Scope(name="sc")
    FromArray(np.zeros(4) + 0j, name="src") >> sp
    sp["re"] >> sc
    sp["im"] >> Record(name="rec")
    cols, edges = layout(Diagram(sc, fs=48e3))
    node = {n.name: n for c in cols for n in c}
    assert node["split"].outputs == ("re", "im") and node["split"].kind == "block"
    assert node["src"].kind == "source" and node["sc"].kind == "sink"
    assert node["src"].rate_text() == "48 kHz"
    assert {(e.sport, e.dst.name) for e in edges if e.src.name == "split"} == {
        ("re", "sc"),
        ("im", "rec"),
    }
    assert node["src"].info == "4 samples"


def test_state_blocks_have_their_own_kind():
    s, sc = Sine(1.0), Scope()
    s >> Delay(3, name="z") >> sc
    cols, _ = layout(Diagram(sc, fs=10.0))
    assert {n.name: n.kind for c in cols for n in c}["z"] == "state"


# text


def test_text_lists_chains_feedback_probes_and_blocks():
    src, add, g, dl, sc = (
        Sine(1e3, name="tone"),
        Sum(name="add"),
        Gain(0.5, name="g"),
        Delay(1, name="z"),
        Scope(name="sc"),
    )
    src >> add["x0"]
    add >> g >> sc
    g >> dl >> add["x1"]
    d = Diagram(sc, fs=48e3)
    d.probe("g")
    text = str(d)
    assert text == d.text()
    lines = text.splitlines()
    assert lines[0] == "Diagram  fs = 48 kHz  |  5 blocks, 5 lines, 1 feedback"
    assert "  tone ──▶ add[x0]" in lines
    assert "  add ──▶ g" in lines
    assert {"  g ──▶ z", "  g ──▶ sc"} <= set(lines)  # fan-out splits the chain
    assert lines[lines.index("feedback:") + 1] == "  z ══⟲══▶ add[x1]"
    assert "  probe:g  <-  g[y]" in lines
    assert any("tone" in line and "Sine" in line and "1 kHz" in line for line in lines)


def test_text_of_a_multirate_diagram_shows_rates():
    class Down2(Gain):
        resampling_ratio = Fraction(1, 2)

        def step(self, x):
            return x[::2]

    s, sc = Sine(1.0, name="s"), Scope(name="sc")
    s >> Down2(1.0, name="dn") >> sc
    text = Diagram(sc, fs=8e3).text()
    assert "8 kHz → 4 kHz" in text and "4 kHz" in text


# matplotlib


def feedback_diagram():
    src, add, g, dl, sc = (
        Sine(1e3, name="tone"),
        Sum(name="add"),
        Gain(0.5, name="g"),
        Delay(1, name="z"),
        Scope(name="sc"),
    )
    src >> add["x0"]
    add >> g >> sc
    g >> dl >> add["x1"]
    return Diagram(sc, fs=48e3)


def test_draw_boxes_labels_and_feedback_style():
    d = feedback_diagram()
    d.probe("g")
    ax = d.draw()
    boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    lines = [p for p in ax.patches if isinstance(p, PathPatch)]
    assert len(boxes) == 5 and len(lines) == 5
    dashed = [p for p in lines if p.get_linestyle() == "--"]
    assert len(dashed) == 1
    texts = {t.get_text() for t in ax.texts}
    assert {"tone", "Sine", "1 kHz", "x0", "x1", "z^-1", "probe:g"} <= texts
    assert ax.get_title(loc="left") == "fs = 48 kHz"
    plt.close("all")


def test_draw_places_columns_left_to_right():
    d = feedback_diagram()
    ax = d.draw(info=False)
    cols, _ = layout(d)
    x = {}
    for p in ax.patches:
        if isinstance(p, FancyBboxPatch):
            x[round(p.get_x() + p.get_width() / 2, 6)] = True
    assert len(x) == len(cols)  # one box centre per column
    assert "1 kHz" not in {t.get_text() for t in ax.texts}
    plt.close("all")


def test_draw_saves_files(tmp_path):
    d = feedback_diagram()
    for ext in ("svg", "png", "pdf"):
        path = tmp_path / f"diagram.{ext}"
        d.draw(save=path)
        assert path.stat().st_size > 1000
    assert (tmp_path / "diagram.svg").read_text().lstrip().startswith("<?xml")
    plt.close("all")


def test_jupyter_shows_the_drawing_as_svg():
    svg = feedback_diagram()._repr_svg_()
    assert "<svg" in svg and "tone" in svg
    assert not plt.get_fignums()  # the figure is closed again


# dot and mermaid


def test_dot_has_one_node_per_block_ports_and_dashed_feedback():
    dot = feedback_diagram().to_dot()
    assert dot.startswith("digraph sigline {") and dot.endswith("}")
    nodes = [ln for ln in dot.splitlines() if "[label=" in ln]
    edges = [ln for ln in dot.splitlines() if "->" in ln]
    assert len(nodes) == 5 and len(edges) == 5
    assert any("<i_x0> x0|<i_x1> x1" in ln for ln in nodes)
    assert sum("style=dashed" in ln for ln in edges) == 1
    assert "z^-1" in dot and "z^-1" not in feedback_diagram().to_dot(info=False)


def test_dot_escapes_record_characters():
    s, sc = Sine(1.0, name="a|b{c}"), Scope(name="<out>")
    s >> sc
    dot = Diagram(sc, fs=1.0).to_dot()
    assert "a\\|b\\{c\\}" in dot and "\\<out\\>" in dot


def test_dot_renders_with_graphviz(tmp_path):
    # also look next to python: conda puts graphviz there without activation
    exe = shutil.which("dot") or shutil.which(
        "dot", path=os.path.dirname(sys.executable)
    )
    if exe is None:
        pytest.skip("graphviz not installed")
    src = tmp_path / "d.dot"
    src.write_text(feedback_diagram().to_dot())
    out = subprocess.run(
        [exe, "-Tsvg", str(src)], capture_output=True, text=True, check=True
    )
    assert "<svg" in out.stdout and not out.stderr


def test_mermaid_flowchart():
    text = feedback_diagram().mermaid().splitlines()
    assert text[0] == "flowchart LR"
    assert sum("-->" in ln or "-.->" in ln for ln in text) == 5
    assert sum("-.->|x1|" in ln for ln in text) == 1


def test_blocks_sort_by_the_port_they_feed():
    # created in the "wrong" order: the block feeding x1 first
    b_src, a_src, add = Ramp(name="feeds_b"), Ramp(name="feeds_a"), Add(name="add")
    b_src >> add["b"]
    a_src >> add["a"]
    cols, _ = layout(Diagram(add, fs=1.0))
    pos = {n.name: i for c in cols for i, n in enumerate(c)}
    assert pos["feeds_a"] < pos["feeds_b"]  # input a is above input b
