"""Block diagrams for DSP and mixed continuous/discrete systems.

Blocks connected by lines (``>>``), in plain Python. Signals are NumPy arrays
in chunks; blocks own their state, so chunking never changes results.

Examples
--------
>>> import sigline as sl
>>> isinstance(sl.__version__, str)
True
"""

from ._numba import HAVE_NUMBA
from .analog import (
    AMDemod,
    AMMod,
    FMDemod,
    FMMod,
    PMDemod,
    PMMod,
    SSBDemod,
    SSBMod,
    hilbert_taps,
)
from .blocks import Delay, Func, Gain, Mixer, Product, Sum
from .continuous import ODE, ContinuousLTI
from .core import Block, Port, PortGroup, StateBlock
from .devices import AudioIn, AudioOut, RtlSdrSource
from .diagram import AlgebraicLoopError, Diagram
from .digital import (
    BitsToWords,
    Demapper,
    FSKDemod,
    FSKMod,
    Mapper,
    MatchedFilter,
    PulseShaper,
    WordsToBits,
    pam,
    psk,
    qam,
    rrc_taps,
)
from .filters import FIR, IIR
from .live import watch
from .measure import (
    BERCounter,
    ConstellationScope,
    EyeDiagram,
    PowerMeter,
    SNREstimator,
    Spectrogram,
    SpectrumAnalyzer,
    qam_ambiguities,
)
from .multirate import (
    DDC,
    Decimate,
    Downsample,
    Interpolate,
    Repeat,
    Resample,
    Upsample,
)
from .nonlinear import Nonlinear, state
from .scope import Scope
from .sources import Const, FromArray, Impulse, Noise, Signal, Sine, Source, Step
from .streams import FromQueue, SampleBuffer, ToQueue
from .subsystem import Subsystem

__version__ = "0.1.0.dev0"

__all__ = [
    "AMDemod",
    "AMMod",
    "AlgebraicLoopError",
    "AudioIn",
    "AudioOut",
    "BERCounter",
    "BitsToWords",
    "Block",
    "Const",
    "ConstellationScope",
    "ContinuousLTI",
    "DDC",
    "Decimate",
    "Delay",
    "Demapper",
    "Diagram",
    "Downsample",
    "EyeDiagram",
    "FIR",
    "FMDemod",
    "FMMod",
    "FSKDemod",
    "FSKMod",
    "FromArray",
    "FromQueue",
    "Func",
    "Gain",
    "HAVE_NUMBA",
    "IIR",
    "Impulse",
    "Interpolate",
    "Mapper",
    "MatchedFilter",
    "Mixer",
    "Noise",
    "ODE",
    "Nonlinear",
    "PMDemod",
    "PMMod",
    "Port",
    "PortGroup",
    "PowerMeter",
    "Product",
    "PulseShaper",
    "Repeat",
    "Resample",
    "RtlSdrSource",
    "SNREstimator",
    "SSBDemod",
    "SSBMod",
    "SampleBuffer",
    "Scope",
    "Signal",
    "Sine",
    "Source",
    "Spectrogram",
    "SpectrumAnalyzer",
    "StateBlock",
    "Step",
    "Subsystem",
    "Sum",
    "ToQueue",
    "Upsample",
    "WordsToBits",
    "__version__",
    "hilbert_taps",
    "pam",
    "psk",
    "qam",
    "qam_ambiguities",
    "rrc_taps",
    "state",
    "watch",
]
