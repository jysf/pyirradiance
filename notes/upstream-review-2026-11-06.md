# irradiance: findings from the Python bindings' rawpy benchmark

**Status 2026-11-07:** P0 shipped (SPEC-047, DEVELOP_VERSION 3) and P1 shipped
(SPEC-048, bit-identical). Both are pinned in pyirradiance (`03fb097`). Develop on
the Canon went from 587 to 392 ms (LibRaw: 463 ms); odd ActiveArea origins now
develop grey to grey. P2 shipped (SPEC-049, bit-identical, `16d10ce`): the
bindings use `develop_color_f32_into`, and peak memory for decode + develop
to float32 on the Canon is 705 MB vs LibRaw's 990 MB (whole process; the
earlier 993-vs-488 compared our float32 with rawpy's uint16). P3 is in progress.

From: pyirradiance (Python bindings, `jysf/pyirradiance`), 2026-11-06.
Code read at irradiance `0b5083f` (main). The Python numbers were measured on
the pinned build (`6d0d46a`), which has the same `color.rs` / `demosaic.rs` code
paths. Machine: Apple M4 Pro, single-threaded per call.
Benchmark method and full results: `bench/README.md` in pyirradiance.

Ordered by priority: correctness first, then the biggest win that keeps the
output unchanged, then memory, then optional extras.

## P0: the colour path ignores the ActiveArea origin (wrong colours)

**What.** `color::develop_color_into` and `demosaic::demosaic_into` pick each
pixel's CFA colour (`BayerPhase::colour(row_odd, col_odd)`) and its BlackLevel
pattern slot (`normalize_mosaic`, `src/color.rs:723`) from the pixel's position
counted from the **plane's** top-left corner. In DNG, both patterns start at the
**ActiveArea's** top-left corner. When `ActiveArea.top` or `.left` is odd, every
pixel is assigned the wrong colour.

**Repro** (synthetic 256x192 RGGB DNG, flat grey subject, Canon 5D3 colour tags;
pyirradiance `notes/active_area_origin_repro.py`, LibRaw as the reference):

| ActiveArea origin (top, left) | irradiance develop mean RGB | LibRaw mean RGB |
|---|---|---|
| (2, 4) even | 0.200 0.200 0.200 | 0.200 0.200 0.200 |
| (2, 5) odd left | **0.589 0.000 0.669** (magenta) | 0.200 0.199 0.202 |
| (3, 4) odd top | **0.589 0.000 0.669** (magenta) | 0.200 0.199 0.202 |

The real files in the corpus have even origins (Canon 5D3 `(80, 122)`, Nikon
D300S `(0, 0)`), so the goldens there are unaffected. We don't know how many
cameras have odd origins, but LibRaw handles them and a wrong-colour image
raises no error, so it would go unnoticed.

**Also in the same path:** `develop_color_into` demosaics and outputs the whole
plane, including the masked border outside ActiveArea (the Canon's 80 rows and
122 columns, about 4% of its pixels). The older `develop.rs` already resolves
ActiveArea and DefaultCrop (`resolve_geometry`, `src/develop.rs:135`); the colour
path could reuse that.

**Suggested fix.** Take the phase and black-level slot relative to the
ActiveArea origin, i.e. `((y - top) & 1, (x - left) & 1)`, or crop to ActiveArea
before demosaic so the origin is (0, 0). Cropping also removes the border work.
Either way this changes output for odd-origin files and for any file with a
masked border, so it needs a DEVELOP_VERSION bump.

**Acceptance.** Grey stays grey (channel spread < 1e-3) for all four origin
parities, on all four Bayer phases.

## P1: demosaic is ~90% of develop time; add an interior fast path

On the Canon (23.4 MP): `develop` takes 587 ms, of which unpack is 1 ms and
`demosaic_into` about 520 ms. LibRaw's full develop takes 477 ms on the same file.

Per pixel, `demosaic_into` does a `checked_div` and `checked_rem` to find
(x, y). Each of the ~9 neighbour reads then goes through `sample()`, i.e.
`reflect()` (i64 saturating maths), `plane_index()` and a bounds-checked `get`.
Edge reflection only matters for the 2-pixel border.

**Suggested fix.** Iterate row by row. Handle the 2-pixel border with the
current code, and the interior with direct row offsets and no reflection.
Slices taken per row (`src[row-2..=row+2]`) keep the code panic-free without
per-read checks. The arithmetic doesn't change, so the output is bit-identical
and the existing golden hashes are the acceptance test. My estimate (not
measured) is a severalfold speedup on the loop.

## P2: memory, two buffers that don't need to exist

For one 23.4 MP develop, these are live together:

| Buffer | Size | Owner |
|---|---|---|
| unpacked plane (u16) | 47 MB | caller |
| `normalized` scratch (u16) | 47 MB | `develop_color_into` |
| `camera_rgb` scratch (u16 x3) | 140 MB | `develop_color_into` |
| `dst` linear sRGB (u16 x3) | 140 MB | caller |
| f32 output (x3) | 280 MB | caller, via `to_linear_f32_into` |

Peak on our corrupt-file benchmark: 993 MB, against LibRaw's 488 MB.

1. **Fuse the scratch passes.** Apply level-normalisation while demosaic reads
   samples, which removes `normalized`. Have pass 2 write final linear sRGB
   (3x3 matrix applied per pixel) directly into `dst`, keeping green in a w*h
   plane instead of in `dst`, which removes `camera_rgb`. Net saving about
   140 MB per 23 MP. Output must stay bit-identical: same integer ops in the
   same order.
2. **Add an f32 entry point** (`develop_color_f32_into(sensor, src, dst: &mut
   [f32])`) that writes `/ 65535.0` directly. The Python bindings (and any ML
   caller) want f32, and today must hold the u16 result and its f32 copy at
   once. That saves 140 MB and one full pass. The values must equal
   `to_linear_f32_into(develop_color_into(..))` exactly.

## P3: optional threading (opt-in)

Every demosaic output row depends only on the input, so splitting rows into
bands across `std::thread::scope` keeps the output bit-identical and needs no
dependency. It must be opt-in, e.g. a `threads` argument with default 1: ML
dataloaders already run one decode per worker process, and nested threads
would oversubscribe the CPU.

## Not asks: what the benchmark showed is already good

- Unpack plus Bayer packing: 11 ms vs LibRaw's 26 ms, and 2x LibRaw's
  throughput under threads and torch DataLoader workers (GIL released).
- 212 corrupt files: no crashes or hangs. All 40 files with plausible but wrong
  IFD values were rejected, where LibRaw returned 3 silently different images.
- Deterministic across repeats and processes; the synthetic golden hashes match
  on macOS arm64, Linux x86_64 and Linux arm64.
- Develop mean colour agrees with LibRaw within 0.01 on the Canon and Nikon
  (after DEC-038).

## For pyirradiance itself (not upstream)

- Fixed: `pack_bayer_4ch` had the CFA half of the P0 bug (black levels already
  used the ActiveArea origin, the channel order didn't). With an odd origin
  the normalised means came out `[0.2, 0.121, 0.092, 0.2]` instead of
  `[0.121, 0.2, 0.2, 0.092]` (R, Gr, Gb, B). Now covered by
  `tests/api.py::test_pack_active_area_origin` and `RGGB_active_odd/*` goldens.
- After P0 lands upstream: move `.github/irradiance-pin`, regenerate goldens
  with the reason in the commit, and release.
- Upstream PROJ-003 (lossless JPEG, compression 7) has closed. Re-test the
  Leica M Typ 246 and Pentax K-3 III files when the pin moves.
