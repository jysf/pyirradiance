#!/bin/sh
# Run tests/ci.sh inside a Linux container, for cross-arch checks without CI.
#
# Usage: tests/linux_docker.sh [platform]   linux/arm64 or linux/amd64 (default)
#
# Both repos are mounted read-only and copied in (minus target/ and .venv/), so
# the host's macOS build output is never reused or overwritten.
#
# When the platform differs from the host's arch, rustc segfaults under
# Docker's QEMU emulation. So the wheel is cross-built in a native container
# (maturin --zig, after clippy + cargo test there), and only the Python tests
# run emulated, against that wheel.
set -eu

PLATFORM=${1:-linux/amd64}
HERE=$(cd "$(dirname "$0")/.." && pwd)
UPSTREAM=$(cd "$HERE/../../crustimg_redo_plus/irradiance" && pwd)
case "$(uname -m)" in
    arm64 | aarch64) NATIVE=linux/arm64 ;;
    *) NATIVE=linux/amd64 ;;
esac
case "$PLATFORM" in
    linux/amd64) TRIPLE=x86_64-unknown-linux-gnu ;;
    linux/arm64) TRIPLE=aarch64-unknown-linux-gnu ;;
    *) echo "unsupported platform $PLATFORM" >&2; exit 2 ;;
esac

# Shared setup: copy both repos into the Cargo path-dep layout, make a venv.
SETUP='
    mkdir -p /w/RustLibraries /w/crustimg_redo_plus
    tar -C /src -cf - --exclude=target --exclude=.venv pyirradiance | tar -C /w/RustLibraries -xf -
    tar -C /src -cf - --exclude=target --exclude=fuzz/target irradiance | tar -C /w/crustimg_redo_plus -xf -
    python -m venv /venv
    /venv/bin/pip install -q --disable-pip-version-check maturin numpy
    /venv/bin/pip install -q --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
    cd /w/RustLibraries/pyirradiance
'
RUST='
    curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal -c clippy >/dev/null
    . "$HOME/.cargo/env"
'
MOUNTS="-v $HERE:/src/pyirradiance:ro -v $UPSTREAM:/src/irradiance:ro"

if [ "$PLATFORM" = "$NATIVE" ]; then
    # shellcheck disable=SC2086
    docker run --rm --platform "$PLATFORM" $MOUNTS python:3.12 sh -euc "$SETUP $RUST"'
        echo "== $(uname -sm), $(rustc --version)"
        VENV=/venv tests/ci.sh
    '
    exit 0
fi

WHEELS=$(mktemp -d)
trap 'rm -rf "$WHEELS"' EXIT
# shellcheck disable=SC2086
docker run --rm --platform "$NATIVE" $MOUNTS -v "$WHEELS:/out" python:3.12 sh -euc "$SETUP $RUST"'
    echo "== build host $(uname -sm), $(rustc --version), target '"$TRIPLE"'"
    cargo clippy --all-targets -- -D warnings
    cargo test --lib
    rustup target add '"$TRIPLE"' >/dev/null
    /venv/bin/pip install -q --disable-pip-version-check ziglang
    PATH="/venv/bin:$PATH" maturin build --release --zig --target '"$TRIPLE"' -o /out
'
# shellcheck disable=SC2086
docker run --rm --platform "$PLATFORM" $MOUNTS -v "$WHEELS:/wheels:ro" python:3.12 sh -euc "$SETUP"'
    /venv/bin/pip install -q --disable-pip-version-check /wheels/*.whl
    echo "== test host $(uname -sm), wheel $(ls /wheels)"
    VENV=/venv SKIP_BUILD=1 tests/ci.sh
'
