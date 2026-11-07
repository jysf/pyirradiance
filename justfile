# Dev tasks for the irradiance Python bindings. `just` lists them.
# Releasing: see RELEASING.md (`just release X.Y.Z`).

set shell := ["sh", "-eu", "-c"]

venv := ".venv"
py := venv / "bin/python"
corpus := env("IRRADIANCE_CORPUS", home_directory() / "PSeven/experiments/crustimg_redo_plus/images")
dng := corpus / "CANON-EOS-5D-MARK-III/5G4A9394.DNG"

default:
    @just --list

# Create .venv with the build, test and benchmark dependencies
setup:
    uv venv {{venv}}
    VIRTUAL_ENV={{venv}} uv pip install maturin numpy torch rawpy

# Build the extension into .venv (rerun after every Rust change)
build:
    VIRTUAL_ENV={{venv}} {{venv}}/bin/maturin develop --release

# Clippy and the pure-Rust unit tests
check:
    cargo clippy --all-targets -- -D warnings
    cargo test --lib

# What CI runs: lint, build, every test on synthetic DNGs
ci:
    VENV={{venv}} tests/ci.sh

# The test scripts on the real-camera corpus (needs a build)
test-corpus:
    {{py}} tests/smoke.py {{dng}}
    {{py}} tests/hostile_dims.py {{dng}}
    {{py}} tests/api.py {{dng}}
    {{py}} tests/test_dataloader.py {{dng}} {{corpus}}

# tests/ci.sh inside a Linux container (linux/amd64 or linux/arm64)
docker platform="linux/amd64":
    tests/linux_docker.sh {{platform}}

# Regenerate tests/golden.json after an intentional output change (say why in the commit)
golden-update:
    {{py}} tests/golden.py --update

# irradiance vs rawpy: throughput, corrupt files, determinism (see bench/README.md)
bench *args:
    {{py}} bench/rawpy_compare.py {{args}}

# Refuse to release unless the tree, branch, CI and version are all ready
release-check version:
    #!/bin/sh
    set -eu
    v="{{version}}"
    fail() { echo "release: $*" >&2; exit 1; }
    echo "$v" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || fail "version must be X.Y.Z, got '$v'"
    [ "$(git branch --show-current)" = main ] || fail "not on main"
    [ -z "$(git status --porcelain)" ] || fail "working tree is not clean"
    git fetch -q origin
    [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || fail "main is not in sync with origin/main"
    ! git rev-parse -q --verify "refs/tags/v$v" >/dev/null || fail "tag v$v already exists"
    code=$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/irradiance/$v/json")
    [ "$code" = 404 ] || fail "irradiance $v is already on PyPI (versions can't be reused)"
    # Every workflow run on this commit must have succeeded.
    runs=$(gh run list --commit "$(git rev-parse HEAD)" --json workflowName,conclusion \
        -q '.[] | .workflowName + "=" + .conclusion')
    [ -n "$runs" ] || fail "no CI runs found for HEAD; push and wait for CI first"
    echo "$runs" | grep -qv '=success$' && fail "CI is not green on HEAD: $runs"
    echo "release: ready for v$v (CI green: $(echo $runs | tr '\n' ' '))"

# Bump the version, commit, tag vX.Y.Z and push; CI then publishes to PyPI and GitHub
release version: (release-check version)
    #!/bin/sh
    set -eu
    v="{{version}}"
    sed -i.bak "s/^version = \".*\"$/version = \"$v\"/" pyproject.toml Cargo.toml
    rm -f pyproject.toml.bak Cargo.toml.bak
    cargo check -q  # updates Cargo.lock to the new version
    git commit -q -am "Release v$v"
    git tag -a "v$v" -m "v$v"
    git push -q origin main "v$v"
    echo "release: pushed v$v; watch it with: gh run watch \$(gh run list -w Wheels -L1 --json databaseId -q '.[0].databaseId')"
