"""Block-diagram simulation of DSP and mixed continuous/discrete systems.

``sigline`` wires signal-processing blocks into diagrams with ``>>`` while
keeping the code plain Python: scripts, functions, parameters as attributes
and parameter sweeps as ``for`` loops. Signals are NumPy arrays processed in
chunks, and every block owns its state, so any chunking gives the same result.

Examples
--------
>>> import sigline as sl
>>> isinstance(sl.__version__, str)
True
"""

from .core import Block, Port
from .diagram import AlgebraicLoopError, Diagram

__version__ = "0.1.0.dev0"

__all__ = ["AlgebraicLoopError", "Block", "Diagram", "Port", "__version__"]
