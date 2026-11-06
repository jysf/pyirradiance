# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

`AGENTS.md` (imported above) holds the goal, current state, ordered gap list, and rules. Treat it as the source of truth for priorities; this file adds the build/test commands and the architecture notes.

## Commands

```bash
# One-time setup
uv venv .venv && uv pip install maturin numpy

# Build + install the extension into .venv (rerun after every Rust change)
.venv/bin/maturin develop --release

# Rust-only check (faster than a full maturin build)
cargo check
cargo clippy

# Tests: real file, hostile inputs, determinism, 4-process agreement; forged dimensions
DNG=~/PSeven/experiments/crustimg_redo_plus/images/CANON-EOS-5D-MARK-III/5G4A9394.DNG
.venv/bin/python tests/smoke.py $DNG
.venv/bin/python tests/hostile_dims.py $DNG
.venv/bin/python tests/api.py $DNG
```

There is no pytest suite yet; `tests/smoke.py`, `tests/hostile_dims.py` and `tests/api.py` are scripts that take the DNG path as `argv[1]`. Run all of them after each change.

## Architecture

- **Naming is split three ways**: the distribution is `pyirradiance` (pyproject), the Rust cdylib is `irradiance_py` (Cargo `[lib]`), and the Python import name is `irradiance` (`[tool.maturin] module-name` and `#[pyo3(name = "irradiance")]`). Keep these consistent when touching packaging.
- **Standalone Cargo workspace**: `Cargo.toml` declares an empty `[workspace]` so this crate doesn't join the parent repo's workspace. Don't remove it.
- **All glue lives in `src/lib.rs`**. Every entry point follows the same shape:
  1. Take `&[u8]` (Python `bytes`).
  2. Do all decoding inside `py.allow_threads(...)` (GIL released) and return `Result<_, irradiance::Error>`.
  3. Map errors to `ValueError` via `err()`; never `unwrap` or index in glue.
  4. Build the NumPy array from a `Vec` with `PyArray1::from_vec(...).reshape(...)`.
- **Pipeline over the upstream crate**: `Container::parse` -> `container.sensor()` (metadata) -> `plane::unpack_into` (u16 Bayer plane) -> `demosaic::demosaic_into` (u16 RGB) -> `tone::to_linear_f32_into` (f32). The `_into` APIs write into caller-allocated buffers, which is what makes the `out=` and allocation-guard gaps possible without changing upstream.
- **Upstream crate is read-only**: `irradiance` is a path dependency at `../../crustimg_redo_plus/irradiance`. If its API blocks you (e.g. no pre-validation size helper), work around it here and record the issue as a note; don't edit it.
- `python/` is currently empty (no pure-Python layer yet).
