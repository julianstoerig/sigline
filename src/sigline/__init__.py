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
    am_demod,
    am_mod,
    fm_demod,
    fm_mod,
    hilbert_taps,
    mixer,
    pm_demod,
    pm_mod,
    ssb_demod,
    ssb_mod,
)
from .blocks import Delay, Func, Gain, Product, Sum, Unwrap
from .channels import awgn, colored_noise, phase_noise
from .continuous import ODE, ContinuousLTI
from .core import Block, Port, PortGroup, StateBlock
from .devices import AudioIn, AudioOut, RtlSdrSource
from .diagram import AlgebraicLoopError, Diagram
from .digital import (
    BitsToWords,
    FSKDemod,
    WordsToBits,
    demapper,
    fsk_mod,
    mapper,
    matched_filter,
    pam,
    psk,
    pulse_shaper,
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
from .sources import (
    Const,
    FromArray,
    Impulse,
    Noise,
    RandomPulses,
    Signal,
    Sine,
    Source,
    Step,
)
from .streams import FromQueue, SampleBuffer, ToQueue
from .subsystem import Subsystem

__version__ = "0.1.0.dev0"

__all__ = [
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
    "Diagram",
    "Downsample",
    "EyeDiagram",
    "FIR",
    "FSKDemod",
    "FromArray",
    "FromQueue",
    "Func",
    "Gain",
    "HAVE_NUMBA",
    "IIR",
    "Impulse",
    "Interpolate",
    "Noise",
    "ODE",
    "Nonlinear",
    "Port",
    "PortGroup",
    "PowerMeter",
    "Product",
    "RandomPulses",
    "Repeat",
    "Resample",
    "RtlSdrSource",
    "SNREstimator",
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
    "Unwrap",
    "Upsample",
    "WordsToBits",
    "__version__",
    "am_demod",
    "am_mod",
    "awgn",
    "colored_noise",
    "demapper",
    "fm_demod",
    "fm_mod",
    "fsk_mod",
    "hilbert_taps",
    "mapper",
    "matched_filter",
    "mixer",
    "pm_demod",
    "pm_mod",
    "pam",
    "phase_noise",
    "psk",
    "pulse_shaper",
    "qam",
    "qam_ambiguities",
    "rrc_taps",
    "ssb_demod",
    "ssb_mod",
    "state",
    "watch",
]
