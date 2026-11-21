"""Execute notebooks in place to refresh their outputs.

Usage: ``python tools/run_notebooks.py [notebook ...]`` (default: all).
Needs the ``notebooks`` dependency group.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def run(path: Path) -> None:
    """Execute the notebook at ``path`` and write it back with its outputs."""
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(nb, path)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(ROOT.glob("notebooks/*.ipynb"))
    for p in paths:
        print(f"running {p.name}")
        run(p)
