# sigline

Block-diagram simulation of DSP, communications and mixed continuous/discrete
systems, built on the scientific Python stack.

Blocks are wired with `>>`, but the code stays flat Python: plain scripts,
plain functions, parameters as attributes and parameter sweeps as `for` loops.
Signals are NumPy arrays processed in chunks; every block owns its state, so
any chunking gives the same result.

> **Status:** early development. The API is being built up step by step and is
> not yet released on PyPI.

## Installation

```bash
pip install sigline            # core
pip install "sigline[numba]"   # with numba for compiled per-sample code
```

numba is optional: everything runs without it, only slower.

## Development

```bash
git clone https://github.com/julianstoerig/sigline.git
cd sigline
pip install -e ".[numba]" --group dev
pytest
ruff check . && ruff format --check .
```

## License

MIT, see [LICENSE](LICENSE).
