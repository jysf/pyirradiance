#!/bin/sh
# Lint, build and run every test on synthetic DNGs (no real-camera corpus).
# This is what CI runs; it also runs locally or in a Linux container.
#
# Usage: tests/ci.sh [synth_dir]
# Env:   VENV        virtualenv with maturin, numpy, torch (default .venv)
#        SKIP_BUILD  1 = the module is already installed in VENV (e.g. a
#                    cross-built wheel); run only the Python tests
set -eu

VENV=${VENV:-.venv}
PY="$VENV/bin/python"
SYN=${1:-$(mktemp -d)}

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    cargo clippy --all-targets -- -D warnings
    cargo test --lib
    VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release
fi

"$PY" tests/synth.py "$SYN"
"$PY" tests/golden.py
for t in smoke hostile_dims api; do
    "$PY" "tests/$t.py" "$SYN/bayer.dng"
done
"$PY" tests/test_dataloader.py "$SYN/bayer.dng" "$SYN/mono.dng"
echo "ci: all passed"
