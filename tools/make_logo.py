"""Generate ``docs/logo.svg``: two blocks, and a sine stem plot on the line between.

Step 1 simulates the samples with sigline (source >> recorder); step 2 draws
the SVG from them.

Usage: ``python tools/make_logo.py [out.svg]``
"""

import sys
from pathlib import Path

import numpy as np

import sigline as sl

# signal
N_SAMPLES = 6  # samples per sine period, one stem each
PHASE = 0.5  # in samples; half a sample keeps every dot off the line

# drawing, all lengths are in the svg's own units
OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "logo.svg"
VIEW_BOX = (0, 0, 32, 32)
LINE_Y = 16  # the line between the blocks, also the stems' zero
BLOCK_SIZE = 4.8
BLOCK_RADIUS = 1.2
LEFT_BLOCK_X = 1.6  # the right block mirrors it
LINE_WIDTH = 1.9  # blocks and line
STEM_WIDTH = 1.5
DOT_RADIUS = 1.35
STEM_SPACING = 2.6  # the stems are centred between the blocks
UNITS_PER_AMPLITUDE = 10  # a sample of 1 stands this far above the line
MARGIN = 1.0  # everything stays this far inside the view box

# black on light themes, white on dark ones
STYLE = (
    "@media (prefers-color-scheme: dark) "
    '{ [stroke="#000"] { stroke: #fff; } [fill="#000"] { fill: #fff; } }'
)


# step 1: the signal


class SampledSine(sl.Block):
    """Source of ``sin(2 pi (n + phase) / period)`` for n = 0, 1, 2, ..."""

    inputs = ()

    def __init__(self, period, phase, **kw):
        self.period, self.phase = period, phase
        super().__init__(**kw)

    def reset(self):
        """Start again at sample 0."""
        self.n = 0

    def step_source(self, L):
        """Return the next ``L`` samples."""
        n = np.arange(self.n, self.n + L)
        self.n += L
        return np.sin(2 * np.pi * (n + self.phase) / self.period)


class Recorder(sl.Block):
    """Sink that keeps every sample it receives."""

    outputs = ()

    def reset(self):
        """Forget everything recorded."""
        self.chunks = []

    def step(self, x):
        """Store one chunk."""
        self.chunks.append(x.copy())
        return {}


def simulate_samples() -> np.ndarray:
    """Run source >> recorder for one period and return what the line carried."""
    source, recorder = SampledSine(N_SAMPLES, PHASE), Recorder()
    source >> recorder
    sl.Diagram(recorder, fs=1.0).run(n=N_SAMPLES)
    return np.concatenate(recorder.chunks)


# step 2: the svg


def number(value: float) -> str:
    """Shortest text for a coordinate, rounded to 0.01."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def stroke_attributes(width: float) -> str:
    """Attributes of a black, round-capped, unfilled stroke."""
    return (
        f'stroke="#000" stroke-width="{number(width)}" stroke-linecap="round" '
        'stroke-linejoin="round" fill="none"'
    )


def block_xs() -> tuple[float, float]:
    """Left x of the left and the right block, mirrored about the centre."""
    width = VIEW_BOX[2]
    return LEFT_BLOCK_X, width - LEFT_BLOCK_X - BLOCK_SIZE


def stem_points(samples: np.ndarray) -> list[tuple[float, float]]:
    """Top of each stem, y points down."""
    centre = VIEW_BOX[2] / 2
    xs = centre + STEM_SPACING * (np.arange(len(samples)) - (len(samples) - 1) / 2)
    ys = LINE_Y - UNITS_PER_AMPLITUDE * samples
    return list(zip(xs.tolist(), ys.tolist(), strict=True))


def check_layout(points: list[tuple[float, float]]) -> None:
    """Raise if a dot leaves the view box or touches a block."""
    top = VIEW_BOX[1] + MARGIN + DOT_RADIUS
    bottom = VIEW_BOX[1] + VIEW_BOX[3] - MARGIN - DOT_RADIUS
    left_edge = block_xs()[0] + BLOCK_SIZE + LINE_WIDTH / 2
    right_edge = block_xs()[1] - LINE_WIDTH / 2
    for x, y in points:
        if not top <= y <= bottom:
            raise ValueError(f"y = {y:.1f} is outside {top:.1f} ... {bottom:.1f}")
        if not left_edge + DOT_RADIUS < x < right_edge - DOT_RADIUS:
            raise ValueError(f"the stem at x = {x:.1f} touches a block")


def build_svg(points: list[tuple[float, float]]) -> str:
    """Return the whole logo as text."""
    y_block = LINE_Y - BLOCK_SIZE / 2
    blocks = "".join(
        f'<rect x="{number(x)}" y="{number(y_block)}" width="{number(BLOCK_SIZE)}" '
        f'height="{number(BLOCK_SIZE)}" rx="{number(BLOCK_RADIUS)}" '
        f"{stroke_attributes(LINE_WIDTH)}/>"
        for x in block_xs()
    )
    left, right = block_xs()
    line = (
        f'<path d="M{number(left + BLOCK_SIZE)} {number(LINE_Y)}'
        f'H{number(right)}" {stroke_attributes(LINE_WIDTH)}/>'
    )
    stems = "".join(
        f'<path d="M{number(x)} {number(LINE_Y)}V{number(y)}" '
        f"{stroke_attributes(STEM_WIDTH)}/>"
        f'<circle cx="{number(x)}" cy="{number(y)}" r="{number(DOT_RADIUS)}" '
        'fill="#000"/>'
        for x, y in points
    )
    view_box = " ".join(number(v) for v in VIEW_BOX)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">'
        f"<style>{STYLE}</style>{blocks}{line}{stems}</svg>\n"
    )


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT
    points = stem_points(simulate_samples())
    check_layout(points)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_svg(points))
    print(f"wrote {output}")
