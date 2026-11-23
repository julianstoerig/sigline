#!/usr/bin/env bash
# Rerun everything after a change: the notebooks (refreshing their saved
# outputs), the tests with and without numba, and lint, as CI does.
#
# usage: tools/check.sh           everything (about 5 minutes)
#        tools/check.sh --quick   without the notebooks (about 3 minutes)
set -euo pipefail
cd "$(dirname "$0")/.."

# the sigline conda environment, unless an environment with sigline is active
if ! python -c "import sigline" 2>/dev/null; then
    PATH="$HOME/miniforge3/envs/sigline/bin:$PATH"
fi

quick=false
[[ "${1:-}" == "--quick" ]] && quick=true

step() { printf '\n== %s\n' "$*"; }

if ! $quick; then
    step "notebooks: run and save their outputs"
    # no MPLBACKEND: with Agg the figures would be missing from the outputs
    env -u MPLBACKEND python tools/run_notebooks.py
    ruff format -q notebooks
fi

step "tests"
python -m pytest -q --ignore=tests/test_notebooks.py

step "tests without numba"
SIGLINE_NO_NUMBA=1 python -m pytest -q --ignore=tests/test_notebooks.py

step "lint"
ruff check .
ruff format --check .

step "all passed"
