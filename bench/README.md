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

For develop, both crop to the visible area (irradiance since DEVELOP_VERSION 4,
release 0.0.4). For pack, irradiance packs the full sensor plane and rawpy the
visible area, so irradiance packs up to 4% more pixels.

The corrupt files are built from the two real DNGs: truncations, random type /
count / value fields in 1-3 TIFF directory entries, small offsets added to many
entries at once, and forged width/height (up to 2^32 x 2^32). Each is decoded
(raw + develop) in its own process, so a crash or hang is recorded rather than
ending the run. A decoded image is compared with the clean file's output.

## Results

Run on 2026-11-10 with the 0.0.4 code (upstream `75f681a`, DEVELOP_VERSION 4:
develop crops to ActiveArea/DefaultCrop and applies Orientation). All five
sections come from one run, single-threaded per call (`threads=1`, the default).
The machine was not fully idle (load average 4.6 at the start, 8 at the end),
so treat differences under about 5% as noise. Earlier results: 2026-11-07
(`16d10ce`, before the crop) gave develop 398 ms (Canon) and 216 ms (Nikon);
2026-11-06 (`6d0d46a`, before upstream's fast path) gave 602 ms and 328 ms.

Machine: Apple M4 Pro, 14 cores, Darwin 25.6.0, Python 3.14.4, irradiance 0.0.4 (DEVELOP_VERSION 4), rawpy 0.28.0 (LibRaw 0.22.2), torch 2.15.0, numpy 2.5.4

Files: CANON-EOS-5D-MARK-III (23.4 MP), NIKON-D300S (12.5 MP)

### 1. Single decode (median)

| file | pipeline | irradiance | rawpy |
|---|---|---|---|
| CANON-EOS-5D-MARK-III | pack | 11 ms | 26 ms |
| CANON-EOS-5D-MARK-III | develop | 410 ms | 472 ms |
| NIKON-D300S | pack | 6 ms | 14 ms |
| NIKON-D300S | develop | 220 ms | 258 ms |

### 2. Threads in one process (files/s)

| pipeline | threads | irradiance | rawpy |
|---|---|---|---|
| pack | 1 | 86.0 | 49.5 |
| pack | 2 | 223.8 | 96.1 |
| pack | 4 | 387.9 | 163.3 |
| pack | 8 | 481.9 | 240.3 |
| develop | 1 | 3.0 | 2.6 |
| develop | 2 | 6.1 | 5.3 |
| develop | 4 | 10.1 | 9.4 |
| develop | 8 | 20.0 | 16.3 |

### 3. torch DataLoader, pack pipeline (files/s, second epoch)

| workers | irradiance | rawpy |
|---|---|---|
| 0 | 84.7 | 42.5 |
| 2 | 130.1 | 64.8 |
| 4 | 223.2 | 111.7 |
| 8 | 265.9 | 165.9 |

### 4. Corrupt files, each in its own process (timeout 60 s)

error = raised a Python exception; same image = decoded to exactly the clean file's output (the corruption didn't matter); DIFFERENT image = returned an image without complaint that differs from the clean file's (silently wrong); crash = the process died; hang = timed out. Memory is the child process's peak RSS for decode + develop to float32, interpreter included.

| corruption | cases | irradiance | rawpy |
|---|---|---|---|
| truncated | 40 | 40 error | 40 error |
| IFD entry field randomised | 120 | 15 DIFFERENT image, 41 error, 64 same image | 16 DIFFERENT image, 15 error, 89 same image |
| many IFD values off by a little | 40 | 40 error | 3 DIFFERENT image, 37 error |
| forged width/height | 12 | 12 error | 12 error |
| peak memory, clean CANON-EOS-5D-MARK-III |  | 672 MB | 986 MB |
| peak memory, clean NIKON-D300S |  | 394 MB | 572 MB |
| peak memory, worst corrupt case |  | 675 MB | 1025 MB |

### 5. Determinism (CANON-EOS-5D-MARK-III: 3 repeat decodes + 4 processes)

| pipeline | irradiance | rawpy |
|---|---|---|
| pack | identical | identical |
| develop | identical | identical |


## What this says

- **ML input path (pack): irradiance is about 2x faster**: 11 ms vs 26 ms per
  23 MP file, about 2x on 8 threads, 1.6-2x through a DataLoader.
- **Full develop: irradiance is about 15% faster** (410 vs 472 ms on 23 MP,
  220 vs 258 ms on 12.5 MP), single-threaded. With `develop(threads=8)` one
  23 MP file takes about 61 ms (not part of this benchmark, which compares
  one decode per thread). Until upstream's demosaic fast path (SPEC-048)
  rawpy was 25% faster. The demosaic algorithms differ (Hamilton-Adams vs AHD),
  so this compares two reasonable defaults, not identical work.
- **Both release the GIL**: develop scales about 6-7x on 8 threads for each.
- **Neither crashed or hung** on 212 corrupt files: truncated and forged-size
  files are clean errors in both. On "plausible but wrong" directories
  irradiance refused all 40 while rawpy returned 3 silently different images.
  When a random directory field changes a value neither can check (a black
  level, a colour matrix), both return a different image without complaint
  (15 irradiance, 16 rawpy of 120).
- **Peak memory is lower for irradiance** for decode + develop to float32:
  672 vs 986 MB on the 23 MP file, 394 vs 572 MB on 12.5 MP (whole process,
  interpreter included). Corrupt files don't push it higher: the allocation
  guard bounds every buffer by the input size.
- **Both are deterministic** on one machine. irradiance is also checked
  bit-identical across macOS/Linux and x86_64/arm64 (`tests/golden.py` in CI);
  rawpy wasn't tested across platforms here.
- **Not measured but matters**: rawpy reads nearly every RAW format; irradiance
  reads only uncompressed DNG today. irradiance's wheel is 0.3 MB (rawpy: 2-3
  MB), one wheel per platform covers Python 3.9+, and it has no C/C++ code.
