"""Gap 1: forged width/height must raise ValueError, never allocate-and-abort.

Usage: hostile_dims.py <bayer.dng>

Patches ImageWidth/ImageLength in every IFD of a real DNG (so the sensor IFD is
hit whichever one irradiance selects), then decodes each forgery in a fresh
subprocess: an OOM abort kills the child, not this script, and shows up as a
non-zero exit code.
"""
import os
import struct
import subprocess
import sys

TAG_WIDTH, TAG_HEIGHT, TAG_SUBIFDS = 256, 257, 330
TYPE_SHORT, TYPE_LONG = 3, 4


def ifd_entries(data):
    """Yield (entry_offset, tag, type, count) for every IFD reachable from IFD0."""
    bo = "<" if data[:2] == b"II" else ">"
    todo, seen = [struct.unpack_from(bo + "I", data, 4)[0]], set()
    while todo:
        off = todo.pop()
        if off in seen or off + 2 > len(data):
            continue
        seen.add(off)
        (n,) = struct.unpack_from(bo + "H", data, off)
        for i in range(n):
            e = off + 2 + 12 * i
            tag, typ, count = struct.unpack_from(bo + "HHI", data, e)
            yield e, tag, typ, count
            if tag == TAG_SUBIFDS and typ == TYPE_LONG:
                if count == 1:
                    todo.append(struct.unpack_from(bo + "I", data, e + 8)[0])
                else:
                    p = struct.unpack_from(bo + "I", data, e + 8)[0]
                    todo.extend(struct.unpack_from(bo + f"{count}I", data, p))
        todo.append(struct.unpack_from(bo + "I", data, off + 2 + 12 * n)[0])


def forge(data, width, height):
    bo = "<" if data[:2] == b"II" else ">"
    b = bytearray(data)
    for e, tag, typ, count in ifd_entries(data):
        if tag in (TAG_WIDTH, TAG_HEIGHT) and count == 1:
            v = width if tag == TAG_WIDTH else height
            if typ == TYPE_SHORT:
                struct.pack_into(bo + "H", b, e + 8, min(v, 0xFFFF))
            elif typ == TYPE_LONG:
                struct.pack_into(bo + "I", b, e + 8, v)
    return bytes(b)


CHILD = """
import sys, irradiance
data = sys.stdin.buffer.read()
for fn in (irradiance.unpack_bayer, irradiance.demosaic):
    try:
        fn(data)
    except ValueError as e:
        print(f"{fn.__name__}: ValueError: {e}")
    else:
        print(f"{fn.__name__}: DECODED (unexpected)")
        sys.exit(2)
"""

CASES = [
    (0xFFFFFFFF, 0xFFFFFFFF),  # ~3.7e19 bytes: guaranteed allocation failure
    (0x7FFFFFFF, 0x7FFFFFFF),
    (65535, 65535),            # ~8.6 GB u16 plane
    (1_000_000, 1_000_000),
    (0xFFFFFFFF, 1),           # one axis only
    (1, 0xFFFFFFFF),
]


def main():
    data = open(os.path.expanduser(sys.argv[1]), "rb").read()
    failures = 0
    for w, h in CASES:
        r = subprocess.run(
            [sys.executable, "-c", CHILD],
            input=forge(data, w, h),
            capture_output=True,
            timeout=120,
        )
        status = "ok" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
        failures += r.returncode != 0
        print(f"{w}x{h}: {status}")
        for line in (r.stdout + r.stderr).decode(errors="replace").strip().splitlines()[-3:]:
            print("   ", line)
    print(f"hostile dims: {len(CASES) - failures}/{len(CASES)} raised ValueError cleanly")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
