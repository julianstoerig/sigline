"""Block diagrams for DSP and mixed continuous/discrete systems.

Blocks connected by lines (``>>``), in plain Python. Signals are NumPy arrays
in chunks; blocks own their state, so chunking never changes results.

Examples
--------
>>> import sigline as sl
>>> isinstance(sl.__version__, str)
True
"""

from .core import Block, Port, StateBlock
from .diagram import AlgebraicLoopError, Diagram
from .scope import Scope
from .sources import Const, Source

__version__ = "0.1.0.dev0"

__all__ = [
    "AlgebraicLoopError",
    "Block",
    "Const",
    "Diagram",
    "Port",
    "Scope",
    "Source",
    "StateBlock",
    "__version__",
]
