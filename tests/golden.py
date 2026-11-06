"""Cross-platform determinism: output hashes on the synthetic DNGs must match
tests/golden.json, which was generated on macOS arm64. CI runs this on
linux x86_64 too; a match there is the evidence for "bit-identical across arch".

Usage: golden.py [--update]
"""
import hashlib
import json
import os
import sys

import irradiance
import synth

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")
PHASES = {"RGGB": (0, 1, 1, 2), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1), "BGGR": (2, 1, 1, 0)}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def compute():
    got = {"develop_version": irradiance.DEVELOP_VERSION}
    mono = synth.mono_dng()
    got["mono.dng"] = sha(mono)
    got["mono/unpack"] = sha(irradiance.unpack_bayer(mono)[0].tobytes())
    for name, pattern in PHASES.items():
        data = synth.bayer_dng(pattern=pattern)
        plane, meta = irradiance.unpack_bayer(data)
        got[f"{name}.dng"] = sha(data)
        got[f"{name}/unpack"] = sha(plane.tobytes())
        got[f"{name}/pack"] = sha(irradiance.pack_bayer_4ch(plane, meta).tobytes())
        got[f"{name}/pack_normalized"] = sha(irradiance.pack_bayer_4ch(plane, meta, normalize=True).tobytes())
        got[f"{name}/demosaic"] = sha(irradiance.demosaic(data)[0].tobytes())
    return got


def main():
    got = compute()
    if "--update" in sys.argv:
        with open(GOLDEN, "w") as f:
            json.dump(got, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {len(got)} hashes to {GOLDEN}")
        return
    with open(GOLDEN) as f:
        want = json.load(f)
    diff = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
    for k in diff:
        print(f"  MISMATCH {k}: golden {want.get(k, '-')[:16]} got {got.get(k, '-')[:16]}")
    if diff:
        # Input-file mismatches mean synth.py itself is non-portable; output
        # mismatches mean decode/demosaic/normalise differ across platforms.
        sys.exit(f"golden: {len(diff)}/{len(want)} hashes differ")
    print(f"golden: {len(want)}/{len(want)} hashes match")


if __name__ == "__main__":
    main()
