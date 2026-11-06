"""Gap 5: torch DataLoader(num_workers=4) over the corpus with corrupt files mixed in.

Usage: test_dataloader.py <bayer.dng> [file_or_dir ...]
  <bayer.dng> must decode (it is also the base for the corrupt files); extra
  files, or directories searched one level deep for *.DNG/*.PEF, join the epoch.
  Local: test_dataloader.py <canon.dng> ~/PSeven/experiments/crustimg_redo_plus/images
  CI:    test_dataloader.py <synth>/bayer.dng <synth>/mono.dng

Proves the "never kills a worker" claim end to end:
1. A dataset that catches ValueError finishes the whole epoch; every corrupt
   file comes back as an error record and every good file matches a
   main-process decode bit for bit.
2. A dataset that does not catch it makes the DataLoader re-raise ValueError
   in the main process. A worker killed by an abort or segfault would surface
   as RuntimeError("DataLoader worker ... exited unexpectedly") instead.
"""
import glob
import hashlib
import os
import struct
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import irradiance
from hostile_dims import forge

def digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


class RawDataset(Dataset):
    """Decode each file to a uint16 plane; packs Bayer planes to [4, H/2, W/2]."""

    def __init__(self, paths, catch):
        self.paths, self.catch = paths, catch

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            with open(path, "rb") as f:
                plane, meta = irradiance.unpack_bayer(f.read())
            x = irradiance.pack_bayer_4ch(plane, meta, normalize=True) if meta.cfa_pattern else plane
            return {"path": path, "digest": digest(x), "shape": tuple(x.shape), "error": None}
        except ValueError as e:
            if not self.catch:
                raise
            return {"path": path, "digest": None, "shape": None, "error": str(e)}


def first(batch):
    """collate_fn for batch_size=1; module-level so spawned workers can pickle it."""
    return batch[0]


def make_corrupt(tmp, canon_bytes):
    """Files that each must raise ValueError; the forged-size ones used to abort."""
    cases = {
        "truncated.dng": canon_bytes[: len(canon_bytes) // 3],
        "forged_dims_abort.dng": forge(canon_bytes, 0x7FFFFFFF, 0x7FFFFFFF),
        "forged_dims_panic.dng": forge(canon_bytes, 0xFFFFFFFF, 0xFFFFFFFF),
        "garbage.dng": bytes(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8)),
        "empty.dng": b"",
        "tiff_header_only.dng": b"II*\x00" + struct.pack("<I", 8),
    }
    paths = []
    for name, payload in cases.items():
        p = os.path.join(tmp, name)
        with open(p, "wb") as f:
            f.write(payload)
        paths.append(p)
    return paths


def main():
    args = [os.path.abspath(os.path.expanduser(a)) for a in sys.argv[1:]]
    canon = args[0]  # the known-good Bayer file
    real = {canon}
    for a in args[1:]:
        if os.path.isdir(a):
            real.update(glob.glob(os.path.join(a, "*", "*.DNG")) + glob.glob(os.path.join(a, "*", "*.PEF")))
        else:
            real.add(a)
    real = sorted(real)
    canon_bytes = open(canon, "rb").read()
    plane, meta = irradiance.unpack_bayer(canon_bytes)
    assert meta.cfa_pattern, f"{canon} is not a Bayer file"
    canon_shape = (4, plane.shape[0] // 2, plane.shape[1] // 2)

    with tempfile.TemporaryDirectory() as tmp:
        corrupt = make_corrupt(tmp, canon_bytes)
        # Interleave so corrupt files land in different workers mid-epoch.
        paths = [p for pair in zip(real, corrupt) for p in pair] + real[len(corrupt):] + corrupt[len(real):]

        # Main-process reference for every file.
        expected = {r["path"]: r for r in map(RawDataset(paths, catch=True).__getitem__, range(len(paths)))}

        # 1. Catching dataset: the epoch completes, results match the main process.
        loader = DataLoader(RawDataset(paths, catch=True), batch_size=1, num_workers=4,
                            collate_fn=first)
        seen = {r["path"]: r for r in loader}
        assert seen.keys() == set(paths), "epoch did not cover every file"
        for p in paths:
            assert seen[p] == expected[p], f"worker result differs from main process for {p}"
        for p in corrupt:
            assert seen[p]["error"], f"{os.path.basename(p)} decoded but should have raised"
        assert seen[canon]["error"] is None and seen[canon]["shape"] == canon_shape

        print(f"epoch: {len(paths)} files, 4 workers, completed")
        for p in paths:
            r = seen[p]
            parent = os.path.basename(os.path.dirname(p))
            label = ("corrupt" if p in corrupt else parent) + "/" + os.path.basename(p)
            print(f"  {label:45s} " + (f"ok {r['shape']} {r['digest']}" if r["error"] is None
                                         else f"ValueError: {r['error'][:70]}"))

        # 2. Non-catching dataset: each corrupt file alone must surface as ValueError
        #    in the main process, never as a dead worker.
        for p in corrupt:
            loader = DataLoader(RawDataset([canon, p, canon], catch=False), batch_size=1,
                                num_workers=2, collate_fn=first)
            try:
                list(loader)
            except ValueError:
                pass
            except RuntimeError as e:
                raise AssertionError(f"worker died on {os.path.basename(p)}: {e}") from None
            else:
                raise AssertionError(f"{os.path.basename(p)} did not raise")
        print(f"propagation: {len(corrupt)}/{len(corrupt)} corrupt files re-raised ValueError, 0 dead workers")
    print("dataloader: all passed")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)  # the macOS default; explicit for Linux CI
    main()
