# pyirradiance — agent brief

Python/NumPy bindings (PyO3 + maturin) over the pure-Rust RAW crate `irradiance`
(`../../crustimg_redo_plus/irradiance`, path dep; READ-ONLY — never edit it from here).

## Goal
A crash-proof, deterministic, GIL-free RAW dataloader primitive for PyTorch/ML:
a truncated or hostile file raises `ValueError`, never kills a worker; output is
bit-identical across OS/arch; wheel is small, no C++ toolchain.

## State (prototype, 1 commit)
- `src/lib.rs`: `unpack_bayer(bytes) -> (uint16[H,W], SensorMeta)`,
  `demosaic(bytes) -> (float32[H,W,3], DEVELOP_VERSION)` (uncorrected camera-space
  RGB — colour correction is upstream STAGE-008, not done), GIL released.
- `tests/smoke.py <dng>`: real file, 83 hostile inputs, determinism, 4-process agreement.
  Passed on `~/PSeven/experiments/crustimg_redo_plus/images/CANON-EOS-5D-MARK-III/5G4A9394.DNG`.
- Build: `uv venv .venv && uv pip install maturin numpy && .venv/bin/maturin develop --release`.

## Known gaps (do in this order)
1. **Allocation guard (blocks the "never crashes" claim).** `unpack()` does
   `vec![0u16; width*height]` before validating; a corrupted header can request a
   huge buffer and abort on OOM (not catchable). Cap pixel count and/or require the
   strip byte total to fit within `data.len()` before allocating. Add hostile tests
   with forged width/height.
2. `unpack_bayer(data, out=None)`: optional preallocated `uint16[H,W]` array (NumPy or
   `torch.from_numpy`/pinned tensor) written via `unpack_into`; shape mismatch -> `ValueError`.
3. `pack_bayer_4ch(plane, meta, out=None)`: RGGB (any of the 4 phases, from `cfa_pattern`)
   -> `[4, H/2, W/2]`, optional black/white-level normalisation. Pure Rust, no unsafe of our own.
4. `meta.cfa_pattern` should be a `list[int]`, not `bytes`.
5. `tests/test_dataloader.py`: torch `DataLoader(num_workers=4)` over the corpus plus a
   corrupt file mixed in; assert the run completes and the bad file surfaces as an exception.
6. README with honest scope: uncompressed Bayer DNG (+ mono) only; compressed
   DNG/lossless-JPEG (Pentax, etc.) returns `ValueError`; demosaic output is uncorrected.

## Rules
- No unsafe of our own beyond what PyO3/numpy require; no panics (`unwrap`/indexing) in glue.
- Don't claim bit-identical across arch until verified on a second arch (CI: linux x86_64 + macOS arm64).
- Don't publish to PyPI; the crate dep must move from the path to a crates.io version first
  (needs an `irradiance` release containing `demosaic`).
- Corpus: `~/PSeven/experiments/crustimg_redo_plus/images` (Bayer uncompressed: Canon 5D3 DNG; Nikon D300S DNG is a different layout).
