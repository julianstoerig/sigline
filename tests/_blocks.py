"""small blocks shared by the core tests."""

import numpy as np

from sigline import Block


class Scale(Block):
    def __init__(self, k=2.0, **kw):
        self.k = k
        super().__init__(**kw)

    def step(self, x):
        return self.k * x


class Add(Block):
    inputs = ("a", "b")

    def step(self, a, b):
        return a + b


class Split(Block):
    outputs = ("re", "im")

    def step(self, x):
        return x.real, x.imag


class Accumulate(Block):
    def reset(self):
        self.acc = 0.0

    def step(self, x):
        y = self.acc + np.cumsum(x)
        self.acc = y[-1]
        return y


class Ramp(Block):
    inputs = ()

    def reset(self):
        self.n = 0

    def step_source(self, L):
        y = np.arange(self.n, self.n + L, dtype=float)
        self.n += L
        return y


class Record(Block):
    outputs = ()

    def reset(self):
        self.chunks = []

    def step(self, x):
        self.chunks.append(x.copy())
        return {}

    @property
    def data(self):
        return np.concatenate(self.chunks) if self.chunks else np.empty(0)
