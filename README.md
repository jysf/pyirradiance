# pyirradiance

Python/NumPy bindings for [`irradiance`](https://github.com/jysf/irradiance), a
pure-Rust camera RAW reader. Built as a dataloader primitive for PyTorch/ML:

- **Hostile files raise `ValueError`.** Truncated, corrupted or forged files
  never panic, abort or kill a `DataLoader` worker.
- **The GIL is released** while decoding, so threads scale.
- **Deterministic output**, with no C/C++ toolchain or system libraries.

> **Status: prototype.** Prebuilt wheels are on the
> [releases page](https://github.com/jysf/pyirradiance/releases); not on PyPI
> until an `irradiance` crates.io release ships `demosaic` (planned as 0.2.0).

## Scope

| Input | Result |
|---|---|
| Uncompressed Bayer DNG (any of RGGB, GRBG, GBRG, BGGR) | decodes |
| Uncompressed monochrome (`LinearRaw`) DNG | `unpack_bayer` decodes; `pack_bayer_4ch` raises `ValueError` |
| Compressed DNG (lossless JPEG, compression 7), vendor RAW (PEF, CR2, NEF, ...) | `ValueError` ("compression N is not supported") |
| Tiled DNG, non-Bayer CFA (X-Trans, ...) | `ValueError` |

Upstream unpacks 8, 12, 14 and 16-bit samples. Tested here on real files:
Canon EOS 5D Mark III and Nikon D300S DNGs (Bayer, 16-bit), Leica M Monochrom
(mono, 16-bit) and Leica Q2 Monochrom (mono, 14-bit). Bayer files below
16-bit are untested here. The Leica M Monochrom Typ 246
and Pentax K-3 III DNGs are lossless-JPEG compressed and are refused for now.

`develop` output is **linear sRGB** (white balance from `AsShotNeutral`, the
DNG colour matrices, no tone curve or gamma), clipped to `[0, 1]`. Files with a
`ForwardMatrix` (Canon 5D3) and with `ColorMatrix` only (Nikon D300S) both work.
`demosaic` output is **camera-space linear RGB**: raw values, no black/white
levels, white balance or colour matrix.

## Install

Prebuilt wheels (Python 3.9+, no Rust needed) for Linux x86_64, Linux arm64
and macOS (Intel and Apple Silicon) are attached to each
[release](https://github.com/jysf/pyirradiance/releases). Install the one for
your platform by URL:

```sh
R=https://github.com/jysf/pyirradiance/releases/download/v0.0.1
# Linux x86_64
pip install $R/irradiance-0.0.1-cp39-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# Linux arm64
pip install $R/irradiance-0.0.1-cp39-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
# macOS
pip install $R/irradiance-0.0.1-cp39-abi3-macosx_10_12_universal2.macosx_10_12_x86_64.macosx_11_0_arm64.whl
```

Install by exact URL rather than `pip install irradiance`: the name isn't
registered on PyPI yet, so a bare name could resolve to someone else's package.
Each wheel passed the full test suite in CI before release. Windows isn't built
yet.

### From source

Needs Rust 1.90+ and the `irradiance` repo checked out at
`../../crustimg_redo_plus/irradiance` relative to this one (Cargo path dep).

```sh
uv venv .venv && uv pip install maturin numpy
.venv/bin/maturin develop --release
```

The package is `irradiance` both to install and to import (this repo is
`pyirradiance`; not on PyPI yet).

## API

```python
import irradiance

data = open("photo.dng", "rb").read()

# Raw sensor plane, unmodified (masked border included).
plane, meta = irradiance.unpack_bayer(data)          # uint16[H, W], SensorMeta

# ML input: 2x2 tiles -> 4 half-resolution channels, always [R, Gr, Gb, B]
# (Gr = the green on red's row), whatever the camera's CFA phase.
x = irradiance.pack_bayer_4ch(plane, meta)                  # uint16[4, H/2, W/2]
x = irradiance.pack_bayer_4ch(plane, meta, normalize=True)  # float32, see below

# Full develop: levels, Hamilton-Adams demosaic, white balance, colour matrix.
rgb, version = irradiance.develop(data)               # float32[H, W, 3] linear sRGB, [0, 1]

# Demosaic only: camera-space RGB, raw values / 65535.
rgb, version = irradiance.demosaic(data)              # float32[H, W, 3]
```

**`normalize=True`** returns `(v - black) / (white - black)` per channel in
float32, **without clipping**: noise goes slightly below 0 and highlights can
exceed 1. `black` is the per-site `BlackLevel` pattern (origin at the
`ActiveArea` top-left, per the DNG spec), `white` is `WhiteLevel` (or
`2**bits - 1`). `BlackLevelDeltaH/V` and a `BlackLevelRepeatDim` that doesn't
fit the 2x2 tile are not supported (the latter raises `ValueError`).

**Cropping:** outputs cover the full plane, including the masked border. To
keep only active pixels, use `meta.active_area` = `(top, left, bottom, right)`
(halve it for `pack_bayer_4ch` output when top/left are even).

**`out=`**: `unpack_bayer(data, out=arr)` and `pack_bayer_4ch(..., out=arr)`
write into a preallocated, C-contiguous, writable array of the right dtype and
shape, and return it. For torch, pass `tensor.numpy()` (works for pinned
tensors). Wrong shape/order/read-only raises `ValueError`; wrong dtype raises
`TypeError`. On a decode error `out` may be partly written.

**`SensorMeta`** fields: `width`, `height`, `bits_per_sample`, `black_level`,
`black_levels`, `black_level_repeat_dim`, `white_level`, `cfa_pattern`
(`list[int]`, 0=R 1=G 2=B), `cfa_repeat_dim`, `cfa_plane_color`, `active_area`.

### DataLoader

```python
class RawDataset(torch.utils.data.Dataset):
    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            plane, meta = irradiance.unpack_bayer(f.read())
        return torch.from_numpy(irradiance.pack_bayer_4ch(plane, meta, normalize=True))
```

A corrupt file raises `ValueError` in the worker, which torch re-raises in the
main process. Catch it in `__getitem__` to skip bad files instead.

## Guarantees and how they're checked

| Claim | Evidence |
|---|---|
| Hostile input -> `ValueError`, never a crash | `tests/smoke.py` (83 truncated/corrupted files), `tests/hostile_dims.py` (forged sizes up to 2^32 x 2^32, each in a subprocess), `tests/test_dataloader.py` (corrupt files mixed into a 4-worker `DataLoader`) |
| Allocation bounded by input size | Before allocating, the decoder requires the pixel data to fit in the file; a forged header can't request more than ~8 samples per input byte |
| Deterministic | Repeat decodes and 4 processes produce identical hashes |
| Same bits on every platform | `tests/golden.py`: 29 output hashes on synthetic DNGs (all 4 CFA phases; unpack, pack, normalise, demosaic, develop on both colour paths). Match verified on macOS arm64 (also on the real corpus), Linux x86_64 and Linux arm64 (synthetic files only). CI runs this on every push, and `wheels.yml` runs it against each built wheel, including under Python 3.9. |

## Development

```sh
tests/ci.sh                    # clippy, cargo test, build, all tests on synthetic DNGs
tests/linux_docker.sh linux/amd64   # the same inside a Linux container
```

`tests/ci.sh` needs `torch` in the venv (`uv pip install torch`). The scripts
in `tests/` also run on real files; see `CLAUDE.md`.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE)
at your option, the same as upstream `irradiance`.
