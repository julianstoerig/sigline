"""Optional numba: ``njit`` compiles when numba is there, else returns the function.

Set ``SIGLINE_NO_NUMBA=1`` to run the pure NumPy paths with numba installed.
"""

from __future__ import annotations

import os
import types

import numpy as np

HAVE_NUMBA = False
if not os.environ.get("SIGLINE_NO_NUMBA"):
    try:
        import numba

        HAVE_NUMBA = True
    except ImportError:  # pragma: no cover - depends on the environment
        pass


def njit(f):
    """Compile ``f`` with numba if available; return it unchanged otherwise."""
    if HAVE_NUMBA:
        return numba.njit(cache=True, nogil=True)(f)
    return f


def callable_in_numba(f, **options):
    """Return ``f`` in a form compiled code can call; needs numba.

    Builtins (``math.tanh``), NumPy ufuncs and compiled functions are used
    as they are; other functions are compiled with ``options``.
    """
    if isinstance(
        f, types.BuiltinFunctionType | np.ufunc | numba.core.dispatcher.Dispatcher
    ):
        return f
    return numba.njit(**options)(f)
