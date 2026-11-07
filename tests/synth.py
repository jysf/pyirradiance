"""Synthetic uncompressed DNGs for CI, where the real-camera corpus can't go.

Usage: synth.py <out_dir>    writes bayer.dng (RGGB) and mono.dng

Pure-integer generation (no RNG, no floats), so the bytes are identical on
every platform; tests/golden.py pins their sha256 to prove it. The Bayer file
mirrors the Canon 5D3 DNG's shape of metadata: 16-bit samples, 2x2 per-site
BlackLevel, WhiteLevel, an ActiveArea with an even origin, and several strips
(so the multi-strip path and the allocation guard's strip sum are exercised),
plus the Canon's colour calibration so develop() has real matrices to use.
"""
import os
import struct
import sys

BYTE, SHORT, LONG, RATIONAL, SRATIONAL = 1, 3, 4, 5, 10
SIZE = {BYTE: 1, SHORT: 2, LONG: 4, RATIONAL: 8, SRATIONAL: 8}
FMT = {BYTE: "B", SHORT: "H", LONG: "I", RATIONAL: "II", SRATIONAL: "ii"}

PHOTOMETRIC_CFA, PHOTOMETRIC_LINEAR_RAW = 32803, 34892


def tiff(entries, pixel_data, strip_rows, row_bytes):
    """Little-endian single-IFD TIFF. `entries` is [(tag, type, [values])] without
    the strip tags, which are filled in here to point at `pixel_data`."""
    n_rows = len(pixel_data) // row_bytes
    strips = [(r, min(strip_rows, n_rows - r)) for r in range(0, n_rows, strip_rows)]
    entries = entries + [
        (273, LONG, [0] * len(strips)),        # StripOffsets, patched below
        (278, LONG, [strip_rows]),             # RowsPerStrip
        (279, LONG, [rows * row_bytes for _, rows in strips]),  # StripByteCounts
    ]
    entries.sort(key=lambda e: e[0])
    ifd_size = 2 + 12 * len(entries) + 4
    extra_at = 8 + ifd_size
    extra = bytearray()
    # Lay out out-of-line values first so the pixel offset is known.
    layout = []
    for tag, typ, vals in entries:
        nbytes = SIZE[typ] * len(vals)
        if nbytes > 4:
            layout.append(extra_at + len(extra))
            extra += b"\0" * (nbytes + nbytes % 2)  # word-align
        else:
            layout.append(None)
    pixels_at = extra_at + len(extra)
    offsets = [pixels_at + r * row_bytes for r, _ in strips]
    entries = [(t, ty, offsets if t == 273 else v) for t, ty, v in entries]

    out = bytearray(b"II*\0" + struct.pack("<I", 8))
    out += struct.pack("<H", len(entries))
    for (tag, typ, vals), at in zip(entries, layout):
        flat = [x for v in vals for x in (v if isinstance(v, tuple) else (v,))]
        packed = struct.pack("<" + FMT[typ] * len(vals), *flat)
        if at is None:
            out += struct.pack("<HHI", tag, typ, len(vals)) + packed.ljust(4, b"\0")
        else:
            out += struct.pack("<HHII", tag, typ, len(vals), at)
            rel = at - extra_at
            extra[rel:rel + len(packed)] = packed
    out += struct.pack("<I", 0)  # no next IFD
    out += extra
    assert len(out) == pixels_at
    return bytes(out + pixel_data)


def sample(x, y, lo, hi):
    """Deterministic texture: gradient + interference, within [lo, hi]."""
    v = (x * 37 + y * 101 + (x * y) % 97 * 13 + ((x ^ y) & 31) * 41) % (hi - lo + 1)
    return lo + v


def ratios(values, den):
    """Exact (numerator, denominator) pairs: values are integers over `den`."""
    return [(v, den) for v in values]


# Colour calibration copied from the Canon EOS 5D Mark III DNG in the corpus
# (matrices x10000, AsShotNeutral x1000000), so develop() takes the same
# dual-illuminant ForwardMatrix path as that camera.
CANON_COLOR = [
    (50721, SRATIONAL, ratios([7234, -1413, -600, -3631, 11150, 2850, -382, 1335, 6437], 10000)),
    (50722, SRATIONAL, ratios([6722, -635, -963, -4287, 12460, 2028, -908, 2162, 5668], 10000)),
    (50728, RATIONAL, ratios([606276, 1000000, 461885], 1000000)),  # AsShotNeutral
    (50778, SHORT, [17]),  # CalibrationIlluminant1: standard light A
    (50779, SHORT, [21]),  # CalibrationIlluminant2: D65
    (50964, SRATIONAL, ratios([7868, 92, 1683, 2291, 8615, -906, 27, -4752, 12976], 10000)),
    (50965, SRATIONAL, ratios([7637, 805, 1201, 2649, 9179, -1828, 137, -2456, 10570], 10000)),
]
FORWARD_MATRIX_TAGS = (50964, 50965)


def bayer_dng(width=256, height=192, pattern=(0, 1, 1, 2), forward_matrix=True, active=(2, 4)):
    """forward_matrix=False drops ForwardMatrix1/2 (ColorMatrix-only colour).
    `active` is the ActiveArea (top, left); the CFA and BlackLevel patterns
    start there, so an odd value shifts every pixel's colour."""
    black, white = [512, 512, 512, 520], 16000
    # Slightly below black and above white, as real sensors read.
    px = b"".join(struct.pack("<H", sample(x, y, 480, 16200)) for y in range(height) for x in range(width))
    entries = [
        (254, LONG, [0]),                    # NewSubfileType: full-res raw
        (256, LONG, [width]),
        (257, LONG, [height]),
        (258, SHORT, [16]),                  # BitsPerSample
        (259, SHORT, [1]),                   # Compression: none
        (262, SHORT, [PHOTOMETRIC_CFA]),
        (277, SHORT, [1]),                   # SamplesPerPixel
        (33421, SHORT, [2, 2]),              # CFARepeatPatternDim
        (33422, BYTE, list(pattern)),        # CFAPattern
        (50706, BYTE, [1, 4, 0, 0]),         # DNGVersion
        (50713, SHORT, [2, 2]),              # BlackLevelRepeatDim
        (50714, LONG, black),                # BlackLevel
        (50717, LONG, [white]),              # WhiteLevel
        (50829, LONG, [active[0], active[1], height - 2, width - 4]),  # ActiveArea (top, left, bottom, right)
    ]
    entries += [e for e in CANON_COLOR if forward_matrix or e[0] not in FORWARD_MATRIX_TAGS]
    return tiff(entries, px, strip_rows=64, row_bytes=2 * width)


def mono_dng(width=128, height=96):
    px = b"".join(struct.pack("<H", sample(x, y, 0, 4095)) for y in range(height) for x in range(width))
    entries = [
        (254, LONG, [0]),
        (256, LONG, [width]),
        (257, LONG, [height]),
        (258, SHORT, [16]),
        (259, SHORT, [1]),
        (262, SHORT, [PHOTOMETRIC_LINEAR_RAW]),
        (277, SHORT, [1]),
        (50706, BYTE, [1, 4, 0, 0]),
        (50714, LONG, [64]),
        (50717, LONG, [4095]),
    ]
    return tiff(entries, px, strip_rows=height, row_bytes=2 * width)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in (("bayer.dng", bayer_dng()), ("mono.dng", mono_dng())):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(payload)
        print(os.path.join(out_dir, name), len(payload), "bytes")


if __name__ == "__main__":
    main()
