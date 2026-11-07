# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

`AGENTS.md` (imported above) holds the goal, current state, ordered gap list, and rules. Treat it as the source of truth for priorities; this file adds the build/test commands and the architecture notes.

## Commands

The `justfile` wraps all of these (`just` lists recipes: `setup`, `build`, `check`, `ci`, `test-corpus`, `docker`, `golden-update`, `bench`, `release`). Releases: `RELEASING.md` (`just release X.Y.Z`; never upload to PyPI by hand).

```bash
# One-time setup
uv venv .venv && uv pip install maturin numpy torch   # torch: dataloader test only

# Build + install the extension into .venv (rerun after every Rust change)
.venv/bin/maturin develop --release

# Rust-only check (faster than a full maturin build)
cargo check
cargo clippy
cargo test --lib   # pure-Rust unit tests (src/pack.rs)

# Tests: real file, hostile inputs, determinism, 4-process agreement; forged dimensions
DNG=~/PSeven/experiments/crustimg_redo_plus/images/CANON-EOS-5D-MARK-III/5G4A9394.DNG
.venv/bin/python tests/smoke.py $DNG
.venv/bin/python tests/hostile_dims.py $DNG
.venv/bin/python tests/api.py $DNG
.venv/bin/python tests/test_dataloader.py ~/PSeven/experiments/crustimg_redo_plus/images
```

`tests/ci.sh` runs everything on synthetic DNGs (no corpus needed); `tests/linux_docker.sh linux/amd64` runs it in a Linux container. After an intentional output change, regenerate goldens with `.venv/bin/python tests/golden.py --update` and say why in the commit.

There is no pytest suite yet; the `tests/` files are scripts. `smoke.py`, `hostile_dims.py` and `api.py` take the DNG path as `argv[1]`; `test_dataloader.py` takes the corpus directory. Run all of them after each change.

## Architecture

- **Naming**: the PyPI distribution and the Python import name are both `irradiance` (pyproject `name`, `[tool.maturin] module-name`, `#[pyo3(name = "irradiance")]`). The Cargo package stays `pyirradiance` (it can't share a name with its `irradiance` dependency) and the cdylib is `irradiance_py` (Cargo `[lib]`). The repo is `jysf/pyirradiance`. Keep these consistent when touching packaging.
- **Standalone Cargo workspace**: `Cargo.toml` declares an empty `[workspace]` so this crate doesn't join the parent repo's workspace. Don't remove it.
- **Pure-Rust kernels live in their own modules** (`src/pack.rs`: Bayer -> 4-channel packing), with no PyO3 types, so `cargo test --lib` covers them.
- **All glue lives in `src/lib.rs`**. Every entry point follows the same shape:
  1. Take `&[u8]` (Python `bytes`) or a numpy array.
  2. Do all decoding inside `py.allow_threads(...)` (GIL released) and return `Result<_, irradiance::Error>` (or `GlueError` for our own checks).
  3. Map errors to `ValueError` (`err()` / `From<GlueError>`); wrong array dtype/rank is `TypeError`. Never `unwrap` or index in glue.
  4. Build the NumPy array from a `Vec` with `PyArray1::from_vec(...).reshape(...)`, or write into `out=` via `writable()`, which enforces C order (numpy's `as_slice` also accepts Fortran order).
- **Allocation guard**: every decode calls `plane_len()` before allocating; keep it that way for any new entry point.
- **Pipeline over the upstream crate**: `Container::parse` -> `container.sensor()` (metadata) -> `plane::unpack_into` (u16 Bayer plane) -> `demosaic::demosaic_into` (u16 RGB), or `color::develop_color_into` (levels + demosaic + colour, u16 linear sRGB) for `develop` -> `tone::to_linear_f32_into` (f32). Both RGB entry points go through `rgb_f32()`. The `_into` APIs write into caller-allocated buffers, which is what makes the `out=` and allocation-guard gaps possible without changing upstream.
- **Upstream crate is read-only**: `irradiance` is a path dependency at `../../crustimg_redo_plus/irradiance`. If its API blocks you (e.g. no pre-validation size helper), work around it here and record the issue as a note; don't edit it.
- `python/` is currently empty (no pure-Python layer yet).
- **CI** clones upstream at the SHA in `.github/irradiance-pin` (shared by `ci.yml` and `wheels.yml`; bump deliberately) into the path the Cargo path dep expects. `ci.yml` also runs weekly against upstream `main` as a canary. `wheels.yml` builds abi3 wheels (manylinux2014 x86_64/aarch64 via zig, macOS universal2), runs the suite against each installed wheel, and on a `vX.Y.Z` tag publishes a GitHub pre-release. `extension-module` is a Cargo feature enabled only by maturin, so `cargo test` links on Linux.
