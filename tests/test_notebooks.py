"""execute every example notebook; skipped without the notebooks group."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_without_errors(path):
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    errors = [
        o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"
    ]
    assert not errors
