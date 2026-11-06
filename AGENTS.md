# pyirradiance — agent brief

Python/NumPy bindings (PyO3 + maturin) over the pure-Rust RAW crate `irradiance`
(`../../crustimg_redo_plus/irradiance`, path dep; READ-ONLY — never edit it from here).

## Goal
A crash-proof, deterministic, GIL-free RAW dataloader primitive for PyTorch/ML:
a truncated or hostile file raises `ValueError`, never kills a worker; output is
bit-identical across OS/arch; wheel is small, no C++ toolchain.

## State (prototype, gaps 1-5 done; macOS arm64 only, no CI yet)
- `src/lib.rs`: `unpack_bayer(bytes, out=None) -> (uint16[H,W], SensorMeta)`,
  `pack_bayer_4ch(plane, meta, out=None, normalize=False) -> [4,H/2,W/2]`,
  `demosaic(bytes) -> (float32[H,W,3], DEVELOP_VERSION)` (uncorrected camera-space
  RGB — colour correction is upstream STAGE-008, not done), GIL released.
- `tests/smoke.py <dng>`: real file, 83 hostile inputs, determinism, 4-process agreement.
- `tests/hostile_dims.py <dng>`: forged width/height (6 cases, each in a subprocess) must
  raise `ValueError`, not panic/abort.
- `tests/api.py <dng>`: `out=`, `pack_bayer_4ch` (4 phases, normalise, rejections), meta types.
- `tests/test_dataloader.py <corpus_dir>`: torch `DataLoader(num_workers=4)` over the corpus
  + 6 corrupt files; epoch completes, worker results match main process, uncaught
  `ValueError` re-raises in the main process (no dead workers).
- The `<dng>` scripts pass on `~/PSeven/experiments/crustimg_redo_plus/images/CANON-EOS-5D-MARK-III/5G4A9394.DNG`.
- Build: `uv venv .venv && uv pip install maturin numpy torch && .venv/bin/maturin develop --release`
  (torch is only for `tests/test_dataloader.py`).

## Known gaps (do in this order)
1. ~~**Allocation guard.**~~ Done (interim): `plane_len()` in `src/lib.rs` bounds the
   plane by the strips, and the strips by `data.len()`, before allocating. Swap it for
   upstream `plane::validate` once irradiance 0.2.0 ships it (SPEC-039, STAGE-010).
2. ~~`unpack_bayer(data, out=None)`~~ Done: C-contiguous writable `uint16[H,W]` (use
   `tensor.numpy()` for torch); bad shape/order/read-only -> `ValueError`, bad dtype -> `TypeError`.
3. ~~`pack_bayer_4ch(plane, meta, out=None, normalize=False)`~~ Done: `[R, Gr, Gb, B]` for all
   4 phases (`src/pack.rs`); normalise = per-site BlackLevel (origin = ActiveArea) to
   WhiteLevel, float32, unclipped. Meta gained `cfa_repeat_dim`, `cfa_plane_color`,
   `black_levels`, `black_level_repeat_dim`, `active_area`.
4. ~~`meta.cfa_pattern` should be a `list[int]`, not `bytes`.~~ Done (stored as `u16`).
5. ~~`tests/test_dataloader.py`~~ Done (see State).
6. README with honest scope: uncompressed Bayer DNG (+ mono) only; compressed
   DNG/lossless-JPEG (Pentax, etc.) returns `ValueError`; demosaic output is uncorrected.

## Rules
- No unsafe of our own beyond what PyO3/numpy require; no panics (`unwrap`/indexing) in glue.
- Don't claim bit-identical across arch until verified on a second arch (CI: linux x86_64 + macOS arm64).
- Don't publish to PyPI; the crate dep must move from the path to a crates.io version first
  (needs an `irradiance` release containing `demosaic`).
- Corpus: `~/PSeven/experiments/crustimg_redo_plus/images`. Decodes today: Canon 5D3 DNG and
  Nikon D300S DNG (Bayer), Leica M Monochrom + Q2 Mono DNGs (mono). `ValueError` (compression 7 /
  65535): Leica M Typ 246 DNG, Pentax K-3 III DNG + PEF — upstream PROJ-003 (lossless JPEG) may change this.
