"""Block diagrams for DSP and mixed continuous/discrete systems.

Blocks connected by lines (``>>``), in plain Python. Signals are NumPy arrays
in chunks; blocks own their state, so chunking never changes results.

Examples
--------
>>> import sigline as sl
>>> isinstance(sl.__version__, str)
True
"""

from .analog import AMDemod, AMMod
from .blocks import Delay, Func, Gain, Mixer, Product, Sum
from .core import Block, Port, StateBlock
from .devices import AudioIn, AudioOut, RtlSdrSource
from .diagram import AlgebraicLoopError, Diagram
from .filters import FIR, IIR
from .live import watch
from .measure import (
    BERCounter,
    ConstellationScope,
    EyeDiagram,
    PowerMeter,
    SNREstimator,
    SpectrumAnalyzer,
    qam_ambiguities,
)
from .scope import Scope
from .sources import Const, FromArray, Noise, Signal, Sine, Source, Step
from .streams import FromQueue, SampleBuffer, ToQueue

__version__ = "0.1.0.dev0"

__all__ = [
    "AMDemod",
    "AMMod",
    "AlgebraicLoopError",
    "AudioIn",
    "AudioOut",
    "BERCounter",
    "Block",
    "Const",
    "ConstellationScope",
    "Delay",
    "Diagram",
    "EyeDiagram",
    "FIR",
    "FromArray",
    "FromQueue",
    "Func",
    "Gain",
    "IIR",
    "Mixer",
    "Noise",
    "Port",
    "PowerMeter",
    "Product",
    "RtlSdrSource",
    "SNREstimator",
    "SampleBuffer",
    "Scope",
    "Signal",
    "Sine",
    "Source",
    "SpectrumAnalyzer",
    "StateBlock",
    "Step",
    "Sum",
    "ToQueue",
    "qam_ambiguities",
    "watch",
    "__version__",
]
