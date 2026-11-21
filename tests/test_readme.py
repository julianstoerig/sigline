"""the self-contained examples in the README must run."""

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = [
    b
    for b in re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    if b.startswith("import")  # complete examples; the rest are fragments
]


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_example_runs(code):
    exec(compile(code, "README.md", "exec"), {})
    plt.close("all")


def test_there_are_runnable_examples():
    assert len(BLOCKS) >= 2
