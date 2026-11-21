"""Number formatting for labels."""

from __future__ import annotations

_PREFIXES = ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"))


def si(v, unit="") -> str:
    """Format with an SI prefix: ``si(1500, "Hz") == "1.5 kHz"``.

    Examples
    --------
    >>> si(1500, "Hz"), si(2e-3, "s"), si(0, "V"), si(12.5)
    ('1.5 kHz', '2 ms', '0 V', '12.5')
    """
    v = float(v)
    if v == 0:
        return f"0 {unit}".rstrip()
    for f, p in _PREFIXES:
        if abs(v) >= f or f == 1e-6:
            return f"{v / f:.4g} {p}{unit}".rstrip()
    raise AssertionError("unreachable")
