"""Signal checks run by the diagram.

Opt-in input checks, as class attributes of a block:

``same_shape_inputs = True``
    all inputs have the same vector width.
``expands_scalars = True``
    with ``same_shape_inputs``: inputs with one value per sample may meet
    vector inputs; the block repeats them over the channels.
``real_inputs = True``
    no complex inputs.
``same_kind_inputs = True``
    all inputs real or all complex; e.g. on the ``Sum`` that adds noise, so
    real noise is not added to complex baseband by mistake. Set it on an
    instance to check only that one.
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
        widths = set(shapes.values())
        if getattr(b, "expands_scalars", False):
            widths.discard(())
        if len(widths) > 1:
            desc = ", ".join(
                f"{k}: {'scalar' if not s else 'width ' + str(s)}"
                for k, s in shapes.items()
            )
            raise ValueError(
                f"{b.name} ({type(b).__name__}): input widths differ ({desc})"
            )
    if getattr(b, "same_kind_inputs", False):
        kinds = {k: np.iscomplexobj(v) for k, v in inputs.items()}
        if len(set(kinds.values())) > 1:
            desc = ", ".join(
                f"{k}: {'complex' if c else 'real'}" for k, c in kinds.items()
            )
            raise TypeError(
                f"{b.name} ({type(b).__name__}) expects all real or all complex "
                f"signals ({desc})"
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
