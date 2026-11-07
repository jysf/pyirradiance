"""irradiance vs rawpy (LibRaw) as a PyTorch data-loading primitive.

Usage: rawpy_compare.py [--quick] [--corpus DIR] [file.dng ...]

Measures, on the same uncompressed Bayer DNGs, with both libraries reading
from in-memory bytes:

  1. latency      one decode, median of several, per file
  2. threads      files/s with 1..8 threads (does the library release the GIL?)
  3. dataloader   files/s through torch DataLoader with 0..8 workers
  4. corrupt      truncated / header-corrupted / forged-size files, each decoded
                  in its own subprocess: clean exception, decoded anyway, crash,
                  or hang (timeout)
  5. determinism  output hash across repeat decodes and across 4 processes

Two pipelines are compared, written the way a user of each library would:

  pack     raw Bayer -> float32 [4, H/2, W/2] = [R, Gr, Gb, B], black/white
           normalised (the usual ML input). irradiance: unpack_bayer +
           pack_bayer_4ch(normalize=True). rawpy: raw_image_visible + the
           same packing in numpy.
  develop  full develop to linear sRGB float32 [H, W, 3]. irradiance:
           develop(). rawpy: postprocess(gamma=(1, 1), camera WB, sRGB,
           16-bit, no auto-bright) / 65535.

The two develops are not the same algorithm (irradiance: Hamilton-Adams;
LibRaw: AHD by default) and rawpy crops to the visible area while irradiance
keeps the full sensor plane, so irradiance processes slightly more pixels.
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
from hostile_dims import forge, ifd_entries  # noqa: E402

import irradiance  # noqa: E402
import rawpy  # noqa: E402
import torch  # noqa: E402

DEFAULT_CORPUS = os.path.expanduser("~/PSeven/experiments/crustimg_redo_plus/images")
LIBS = ("irradiance", "rawpy")


# --- the two pipelines, per library -----------------------------------------

def irr_pack(data):
    plane, meta = irradiance.unpack_bayer(data)
    return irradiance.pack_bayer_4ch(plane, meta, normalize=True)


def irr_develop(data):
    return irradiance.develop(data)[0]


def rawpy_pack(data):
    with rawpy.imread(io.BytesIO(data)) as r:
        raw = r.raw_image_visible
        # raw_pattern is for the full raw image; shift it to the visible origin.
        pat = np.roll(r.raw_pattern, (-(r.sizes.top_margin % 2), -(r.sizes.left_margin % 2)), (0, 1))
        blacks, white = r.black_level_per_channel, r.white_level
        slots = {int(pat[y, x]): (y, x) for y in range(2) for x in range(2)}
        r_slot, b_slot = slots[0], slots[2]
        greens = [(y, x) for y in range(2) for x in range(2) if pat[y, x] in (1, 3)]
        gr = next(s for s in greens if s[0] == r_slot[0])
        gb = next(s for s in greens if s[0] == b_slot[0])
        h2, w2 = raw.shape[0] // 2, raw.shape[1] // 2
        out = np.empty((4, h2, w2), np.float32)
        for k, (y, x) in enumerate((r_slot, gr, gb, b_slot)):
            black = np.float32(blacks[int(pat[y, x])])
            ch = raw[y:2 * h2:2, x:2 * w2:2].astype(np.float32)
            out[k] = (ch - black) / (np.float32(white) - black)
        return out


def rawpy_develop(data):
    with rawpy.imread(io.BytesIO(data)) as r:
        rgb = r.postprocess(
            gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB, user_flip=0,
        )
    return rgb.astype(np.float32) / np.float32(65535)


PIPELINES = {
    ("irradiance", "pack"): irr_pack,
    ("irradiance", "develop"): irr_develop,
    ("rawpy", "pack"): rawpy_pack,
    ("rawpy", "develop"): rawpy_develop,
}


# --- DataLoader --------------------------------------------------------------

class Files(torch.utils.data.Dataset):
    """Reads a file per item and runs one pipeline on it, like a real dataset."""

    def __init__(self, paths, lib, n):
        self.paths, self.lib, self.n = paths, lib, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        with open(self.paths[i % len(self.paths)], "rb") as f:
            return torch.from_numpy(PIPELINES[(self.lib, "pack")](f.read()))


def first(batch):
    return batch


# --- corrupt-file child ------------------------------------------------------

CHILD = r"""
import hashlib, io, resource, sys
lib = sys.argv[1]
data = sys.stdin.buffer.read()
try:
    if lib == "irradiance":
        import irradiance
        irradiance.unpack_bayer(data)
        img = irradiance.develop(data)[0]
    else:
        import rawpy
        with rawpy.imread(io.BytesIO(data)) as r:
            r.raw_image.copy()
            img = r.postprocess(use_camera_wb=True, output_bps=16)
    outcome = "decoded:" + hashlib.sha256(img.tobytes()).hexdigest()[:16]
except Exception as e:
    outcome = "error:" + type(e).__name__
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10
print(outcome, round(rss_mb))
"""


def corruptions(data, rng, n_trunc, n_field, n_value):
    """(kind, bytes) cases built from one valid file. The IFD cases hit the TIFF
    directory entries themselves (where parser bugs live), not pixel data."""
    cases = []
    for _ in range(n_trunc):
        cases.append(("truncated", data[: rng.randrange(1024, len(data))]))
    entries = [e for e, *_ in ifd_entries(data)]
    for _ in range(n_field):
        # 1-3 entries get a random type, count or value/offset field.
        b = bytearray(data)
        for e in rng.sample(entries, rng.randint(1, 3)):
            at, size = rng.choice([(e + 2, 2), (e + 4, 4), (e + 8, 4)])
            b[at:at + size] = rng.randbytes(size)
        cases.append(("IFD entry field randomised", bytes(b)))
    for _ in range(n_value):
        # Every entry's value/offset nudged by a small amount: plausible but wrong.
        b = bytearray(data)
        for e in entries:
            if rng.random() < 0.3:
                b[e + 8] = (b[e + 8] + rng.choice([-1, 1, 2, 255])) % 256
        cases.append(("many IFD values off by a little", bytes(b)))
    for w, h in [(0xFFFFFFFF, 0xFFFFFFFF), (0x7FFFFFFF, 0x7FFFFFFF), (65535, 65535),
                 (1_000_000, 1_000_000), (0xFFFFFFFF, 1), (1, 0xFFFFFFFF)]:
        cases.append(("forged width/height", forge(data, w, h)))
    return cases


def run_child(lib, payload, timeout):
    try:
        r = subprocess.run([sys.executable, "-c", CHILD, lib], input=payload,
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "hang", None
    if r.returncode != 0:
        return "crash", None  # killed by a signal / abort / OOM
    outcome, rss = r.stdout.decode().split()
    return outcome, int(rss)


# --- helpers -----------------------------------------------------------------

def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def det_worker(args):
    lib, mode, path = args
    with open(path, "rb") as f:
        return sha(PIPELINES[(lib, mode)](f.read()))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def machine():
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    return (f"{cpu}, {os.cpu_count()} cores, {platform.system()} {platform.release()}, "
            f"Python {platform.python_version()}, irradiance {_version('irradiance')} "
            f"(DEVELOP_VERSION {irradiance.DEVELOP_VERSION}), rawpy {rawpy.__version__} "
            f"(LibRaw {'.'.join(map(str, rawpy.libraw_version))}), torch {torch.__version__}, "
            f"numpy {np.__version__}")


def _version(dist):
    from importlib.metadata import version
    return version(dist)


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="uncompressed Bayer DNGs (default: from --corpus)")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--quick", action="store_true", help="fewer repeats, for a smoke run")
    ap.add_argument("--json", help="also write raw results here")
    args = ap.parse_args()

    paths = args.files or [p for p in sorted(glob.glob(os.path.join(args.corpus, "*", "*")))
                           if p.lower().endswith(".dng") and _is_bayer(p)]
    if not paths:
        sys.exit("no uncompressed Bayer DNGs found; pass files or --corpus")
    datas = [open(p, "rb").read() for p in paths]
    names = [os.path.basename(os.path.dirname(p)) for p in paths]
    q = args.quick
    results = {"machine": machine(), "files": names}
    md = [f"Machine: {results['machine']}", "",
          "Files: " + ", ".join(f"{n} ({irradiance.unpack_bayer(d)[0].size / 1e6:.1f} MP)"
                                for n, d in zip(names, datas)), ""]
    print("\n".join(md), flush=True)

    # 1. latency
    reps = 3 if q else 7
    rows = []
    for name, data in zip(names, datas):
        for mode in ("pack", "develop"):
            row = [name, mode]
            for lib in LIBS:
                fn = PIPELINES[(lib, mode)]
                fn(data)  # warm up
                ts = []
                for _ in range(reps):
                    t = time.perf_counter()
                    fn(data)
                    ts.append(time.perf_counter() - t)
                row.append(f"{statistics.median(ts) * 1000:.0f} ms")
                results.setdefault("latency", []).append(
                    {"file": name, "mode": mode, "lib": lib, "median_s": statistics.median(ts)})
            rows.append(row)
    md += ["### 1. Single decode (median)", "", table(["file", "pipeline", *LIBS], rows), ""]
    print("\n".join(md[-4:]), flush=True)

    # 2. thread scaling
    n_tasks = 8 if q else 24
    rows = []
    for mode in ("pack", "develop"):
        for threads in (1, 2, 4, 8):
            row = [mode, threads]
            for lib in LIBS:
                fn = PIPELINES[(lib, mode)]
                jobs = [datas[i % len(datas)] for i in range(n_tasks)]
                t = time.perf_counter()
                with cf.ThreadPoolExecutor(threads) as ex:
                    list(ex.map(fn, jobs))
                fps = n_tasks / (time.perf_counter() - t)
                row.append(f"{fps:.1f}")
                results.setdefault("threads", []).append(
                    {"mode": mode, "threads": threads, "lib": lib, "files_per_s": fps})
            rows.append(row)
    md += ["### 2. Threads in one process (files/s)", "",
           table(["pipeline", "threads", *LIBS], rows), ""]
    print("\n".join(md[-4:]), flush=True)

    # 3. DataLoader
    n_items = 16 if q else 48
    rows = []
    for workers in (0, 2, 4, 8):
        row = [workers]
        for lib in LIBS:
            ds = Files(paths, lib, n_items)
            dl = torch.utils.data.DataLoader(
                ds, batch_size=None, num_workers=workers, collate_fn=first,
                persistent_workers=workers > 0)
            for _ in dl:  # warm-up epoch: worker startup and imports
                pass
            t = time.perf_counter()
            for _ in dl:
                pass
            fps = n_items / (time.perf_counter() - t)
            row.append(f"{fps:.1f}")
            results.setdefault("dataloader", []).append(
                {"workers": workers, "lib": lib, "files_per_s": fps})
            del dl
        rows.append(row)
    md += ["### 3. torch DataLoader, pack pipeline (files/s, second epoch)", "",
           table(["workers", *LIBS], rows), ""]
    print("\n".join(md[-4:]), flush=True)

    # 4. corrupt files
    rng = random.Random(0)
    cases, source = [], []
    for j, data in enumerate(datas):
        new = corruptions(data, rng, *((4, 10, 4) if q else (20, 60, 20)))
        cases += new
        source += [j] * len(new)
    timeout = 60
    with cf.ThreadPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as ex:
        clean = {(lib, j): ex.submit(run_child, lib, data, timeout)
                 for j, data in enumerate(datas) for lib in LIBS}
        futs = {(lib, i): ex.submit(run_child, lib, payload, timeout)
                for i, (_, payload) in enumerate(cases) for lib in LIBS}
        clean = {k: f.result()[0] for k, f in clean.items()}
        raw_outcomes = {k: f.result() for k, f in futs.items()}

    def classify(lib, i, outcome):
        if not outcome.startswith("decoded:"):
            return outcome.split(":")[0]  # error / crash / hang
        same = outcome == clean[(lib, source[i])]
        return "same image" if same else "DIFFERENT image"

    outcomes = {(lib, i): (classify(lib, i, o), r) for (lib, i), (o, r) in raw_outcomes.items()}
    kinds = list(dict.fromkeys(k for k, _ in cases))
    rows = []
    for kind in kinds:
        idx = [i for i, (k, _) in enumerate(cases) if k == kind]
        row = [kind, len(idx)]
        for lib in LIBS:
            counts = {}
            for i in idx:
                counts[outcomes[(lib, i)][0]] = counts.get(outcomes[(lib, i)][0], 0) + 1
            row.append(", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        rows.append(row)
    peak = {lib: max((r for (l, _), (_, r) in outcomes.items() if l == lib and r), default=0)
            for lib in LIBS}
    rows.append(["peak memory of any case", "", *(f"{peak[lib]} MB" for lib in LIBS)])
    results["corrupt"] = [{"kind": cases[i][0], "lib": lib, "outcome": o, "rss_mb": r}
                          for (lib, i), (o, r) in outcomes.items()]
    md += [f"### 4. Corrupt files, each in its own process (timeout {timeout} s)", "",
           "error = raised a Python exception; same image = decoded to exactly the clean "
           "file's output (the corruption didn't matter); DIFFERENT image = returned an "
           "image without complaint that differs from the clean file's (silently wrong); "
           "crash = the process died; hang = timed out.", "",
           table(["corruption", "cases", *LIBS], rows), ""]
    print("\n".join(md[-6:]), flush=True)

    # 5. determinism
    import multiprocessing as mp
    rows = []
    for mode in ("pack", "develop"):
        row = [mode]
        for lib in LIBS:
            fn = PIPELINES[(lib, mode)]
            local = {sha(fn(datas[0])) for _ in range(3)}
            with mp.get_context("spawn").Pool(4) as pool:
                procs = set(pool.map(det_worker, [(lib, mode, paths[0])] * 4))
            same = len(local | procs) == 1
            row.append("identical" if same else f"{len(local | procs)} different outputs")
            results.setdefault("determinism", []).append(
                {"mode": mode, "lib": lib, "distinct_hashes": len(local | procs)})
        rows.append(row)
    md += [f"### 5. Determinism ({names[0]}: 3 repeat decodes + 4 processes)", "",
           table(["pipeline", *LIBS], rows), ""]
    print("\n".join(md[-4:]), flush=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")


def _is_bayer(path):
    try:
        _, meta = irradiance.unpack_bayer(open(path, "rb").read())
        return meta.cfa_pattern is not None
    except ValueError:
        return False


if __name__ == "__main__":
    main()
