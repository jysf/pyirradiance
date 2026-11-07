# irradiance vs rawpy

`rawpy_compare.py` compares irradiance with [rawpy](https://github.com/letmaik/rawpy)
(Python bindings for LibRaw, the usual way to read RAW files in Python) as a
data-loading primitive for PyTorch.

```sh
just bench                  # all uncompressed Bayer DNGs in the corpus
just bench --quick          # fewer repeats
just bench a.dng b.dng      # your own files
```

Needs `rawpy` and `torch` in `.venv` (`just setup`). A full run takes about 10 minutes.

## Method

Both libraries decode the same files from in-memory bytes. Two pipelines,
each written the way a user of that library would write it:

- **pack**: raw Bayer to the usual ML input, float32 `[4, H/2, W/2]` =
  `[R, Gr, Gb, B]`, black/white normalised. irradiance: `unpack_bayer` +
  `pack_bayer_4ch(normalize=True)`. rawpy: `raw_image_visible`, packed in numpy.
  On the pixels both cover the outputs are **bit-identical** (checked on both files).
- **develop**: full develop to linear sRGB float32. irradiance: `develop()`
  (Hamilton-Adams demosaic). rawpy: `postprocess(gamma=(1, 1), camera WB,
  sRGB, 16-bit, no auto-bright)` (AHD demosaic). Different algorithms, so the
  timings compare two reasonable defaults, not identical work. Mean colours
  agree to within 0.01 on both files.

rawpy crops to the visible area; irradiance keeps the full sensor plane, so it
processes up to 4% more pixels.

The corrupt files are built from the two real DNGs: truncations, random type /
count / value fields in 1-3 TIFF directory entries, small offsets added to many
entries at once, and forged width/height (up to 2^32 x 2^32). Each is decoded
(raw + develop) in its own process, so a crash or hang is recorded rather than
ending the run. A decoded image is compared with the clean file's output.

## Results

Run on 2026-11-07 against upstream `03fb097` (DEVELOP_VERSION 3, with the
demosaic interior fast path). The previous run (2026-11-06, upstream
`6d0d46a`) had develop at 602 ms (Canon) and 328 ms (Nikon); everything else
was within noise.

Machine: Apple M4 Pro, 14 cores, Darwin 25.6.0, Python 3.14.4, irradiance 0.0.2 (DEVELOP_VERSION 3), rawpy 0.28.0 (LibRaw 0.22.2), torch 2.15.0, numpy 2.5.4

Files: CANON-EOS-5D-MARK-III (23.4 MP), NIKON-D300S (12.5 MP)

### 1. Single decode (median)

| file | pipeline | irradiance | rawpy |
|---|---|---|---|
| CANON-EOS-5D-MARK-III | pack | 11 ms | 25 ms |
| CANON-EOS-5D-MARK-III | develop | 394 ms | 463 ms |
| NIKON-D300S | pack | 6 ms | 14 ms |
| NIKON-D300S | develop | 211 ms | 254 ms |

### 2. Threads in one process (files/s)

| pipeline | threads | irradiance | rawpy |
|---|---|---|---|
| pack | 1 | 90.1 | 48.1 |
| pack | 2 | 220.4 | 96.1 |
| pack | 4 | 383.0 | 165.8 |
| pack | 8 | 470.7 | 233.8 |
| develop | 1 | 3.2 | 2.7 |
| develop | 2 | 6.2 | 5.3 |
| develop | 4 | 9.3 | 9.5 |
| develop | 8 | 19.7 | 15.7 |

### 3. torch DataLoader, pack pipeline (files/s, second epoch)

| workers | irradiance | rawpy |
|---|---|---|
| 0 | 85.4 | 41.0 |
| 2 | 125.5 | 65.0 |
| 4 | 221.8 | 117.0 |
| 8 | 264.0 | 173.7 |

### 4. Corrupt files, each in its own process (timeout 60 s)

error = raised a Python exception; same image = decoded to exactly the clean file's output (the corruption didn't matter); DIFFERENT image = returned an image without complaint that differs from the clean file's (silently wrong); crash = the process died; hang = timed out.

| corruption | cases | irradiance | rawpy |
|---|---|---|---|
| truncated | 40 | 40 error | 40 error |
| IFD entry field randomised | 120 | 13 DIFFERENT image, 37 error, 70 same image | 18 DIFFERENT image, 15 error, 87 same image |
| many IFD values off by a little | 40 | 40 error | 3 DIFFERENT image, 37 error |
| forged width/height | 12 | 12 error | 12 error |
| peak memory of any case |  | 972 MB | 485 MB |

### 5. Determinism (CANON-EOS-5D-MARK-III: 3 repeat decodes + 4 processes)

| pipeline | irradiance | rawpy |
|---|---|---|
| pack | identical | identical |
| develop | identical | identical |


## What this says

- **ML input path (pack): irradiance is about 2x faster**: 11 ms vs 25 ms per
  23 MP file, 471 vs 234 files/s on 8 threads, 1.5-2x through a DataLoader.
- **Full develop: irradiance is about 15% faster** (394 vs 463 ms on 23 MP,
  211 vs 254 ms on 12.5 MP). Until upstream's demosaic fast path (SPEC-048)
  rawpy was 25% faster. The demosaic algorithms differ (Hamilton-Adams vs AHD),
  so this compares two reasonable defaults, not identical work.
- **Both release the GIL**: develop scales about 6x on 8 threads for each.
- **Neither crashed or hung** on 212 corrupt files: truncated and forged-size
  files are clean errors in both. On "plausible but wrong" directories
  irradiance refused all 40 while rawpy returned 3 silently different images.
  When a random directory field changes a value neither can check (a black
  level, a colour matrix), both return a different image without complaint
  (13 irradiance, 18 rawpy of 120).
- **Peak memory is higher for irradiance** (972 vs 485 MB in the worst corrupt
  case). Its develop holds several full-size buffers, and the allocation guard
  bounds the plane by the file size rather than by what's sensible.
- **Both are deterministic** on one machine. irradiance is also checked
  bit-identical across macOS/Linux and x86_64/arm64 (`tests/golden.py` in CI);
  rawpy wasn't tested across platforms here.
- **Not measured but matters**: rawpy reads nearly every RAW format; irradiance
  reads only uncompressed DNG today. irradiance's wheel is 0.3 MB (rawpy: 2-3
  MB), one wheel per platform covers Python 3.9+, and it has no C/C++ code.
