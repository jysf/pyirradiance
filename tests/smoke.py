"""Smoke test: real Bayer DNG, hostile inputs, determinism, multiprocess."""
import hashlib, os, random, sys, time
from multiprocessing import Pool
import numpy as np
import irradiance

PATH = os.path.expanduser(sys.argv[1])
data = open(PATH, "rb").read()

def worker(i):
    return hashlib.sha256(irradiance.demosaic(data)[0].tobytes()).hexdigest()[:16]

def main():

    t = time.perf_counter()
    plane, meta = irradiance.unpack_bayer(data)
    print(f"unpack  {plane.shape} {plane.dtype} cfa={meta.cfa_pattern} "
          f"black={meta.black_level} white={meta.white_level} {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    rgb, ver = irradiance.demosaic(data)
    print(f"demosaic {rgb.shape} {rgb.dtype} v{ver} range=[{rgb.min():.3f},{rgb.max():.3f}] {time.perf_counter()-t:.2f}s")
    d1 = hashlib.sha256(rgb.tobytes()).hexdigest()
    d2 = hashlib.sha256(irradiance.demosaic(data)[0].tobytes()).hexdigest()
    assert d1 == d2, "non-deterministic"
    print("sha256", d1[:16], "(repeat identical)")

    # Hostile inputs: truncations + random byte corruption must raise ValueError, never crash.
    rng = random.Random(0)
    ok = err = 0
    cases = [data[: rng.randrange(len(data))] for _ in range(40)] + [b"", b"II*\x00", b"\x00" * 64]
    for _ in range(40):
        b = bytearray(data[:200_000] + data[200_000:])  # copy
        for _ in range(50):
            b[rng.randrange(min(len(b), 4096))] = rng.randrange(256)  # hit the IFD region
        cases.append(bytes(b))
    for c in cases:
        try:
            irradiance.unpack_bayer(c); ok += 1
        except ValueError:
            err += 1
    print(f"hostile: {len(cases)} cases -> {ok} decoded, {err} ValueError, 0 crashes")

    with Pool(4) as p:
        hs = set(p.map(worker, range(4)))
    assert len(hs) == 1
    print("4 processes agree:", hs)

if __name__ == "__main__":
    main()
