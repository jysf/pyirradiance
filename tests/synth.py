"""Synthetic uncompressed DNGs for CI, where the real-camera corpus can't go.

Usage: synth.py <out_dir>    writes bayer.dng (RGGB) and mono.dng

Pure-integer generation (no RNG, no floats), so the bytes are identical on
every platform; tests/golden.py pins their sha256 to prove it. The Bayer file
mirrors the Canon 5D3 DNG's shape of metadata: 16-bit samples, 2x2 per-site
BlackLevel, WhiteLevel, an ActiveArea with an even origin, and several strips
(so the multi-strip path and the allocation guard's strip sum are exercised).
"""
import os
import struct
import sys

BYTE, SHORT, LONG = 1, 3, 4
SIZE = {BYTE: 1, SHORT: 2, LONG: 4}
FMT = {BYTE: "B", SHORT: "H", LONG: "I"}

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
        packed = struct.pack("<" + FMT[typ] * len(vals), *vals)
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


def bayer_dng(width=256, height=192, pattern=(0, 1, 1, 2)):
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
        (50829, LONG, [2, 4, height - 2, width - 4]),  # ActiveArea (top, left, bottom, right)
    ]
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
