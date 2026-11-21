"""Signal checks run by the diagram.

Opt-in input checks, as class attributes of a block:

``same_shape_inputs = True``
    all inputs have the same vector width.
``real_inputs = True``
    no complex inputs.
"""

from __future__ import annotations

import numpy as np


def describe(inputs) -> str:
    """Return a short ``port: dtype [width]`` summary of a block's inputs."""
    out = []
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            w = "" if v.ndim == 1 else f" width {v.shape[1:]}"
            out.append(f"{k}: {v.dtype}{w}")
    return ", ".join(out)


def check_inputs(b, inputs):
    """Raise if ``inputs`` violate the checks block ``b`` opted in to."""
    if getattr(b, "same_shape_inputs", False) and len(inputs) > 1:
        shapes = {k: np.shape(v)[1:] for k, v in inputs.items()}
        if len(set(shapes.values())) > 1:
            desc = ", ".join(
                f"{k}: {'scalar' if not s else 'width ' + str(s)}"
                for k, s in shapes.items()
            )
            raise ValueError(
                f"{b.name} ({type(b).__name__}): input widths differ ({desc})"
            )
    if getattr(b, "real_inputs", False):
        cplx = [k for k, v in inputs.items() if np.iscomplexobj(v)]
        if cplx:
            raise TypeError(
                f"{b.name} ({type(b).__name__}) expects real signals, but "
                f"{cplx} {'is' if len(cplx) == 1 else 'are'} complex"
            )


def check_output(b, k, v):
    """Raise if output ``k`` of block ``b`` contains NaN or Inf."""
    if v.dtype.kind in "fc" and v.size and not np.all(np.isfinite(v)):
        bad = ~np.isfinite(v.reshape(len(v), -1)).all(axis=1)
        i = int(np.flatnonzero(bad)[0])
        raise FloatingPointError(
            f"{b.name}[{k!r}] ({type(b).__name__}) produced NaN/Inf at sample {i} "
            "of this chunk; its inputs were finite, so this block is the first "
            "one to fail"
        )
