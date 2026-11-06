"""API tests: unpack_bayer(out=), pack_bayer_4ch, SensorMeta field types.

Usage: api.py <bayer.dng>
"""
import os
import sys

import numpy as np
import irradiance

from hostile_dims import ifd_entries  # same tests/ dir

TAG_CFA_PATTERN = 33422
MONO = os.path.expanduser("~/PSeven/experiments/crustimg_redo_plus/images/LEICA-M-MONOCHROM/L1000622.DNG")

data = open(os.path.expanduser(sys.argv[1]), "rb").read()
ref, meta = irradiance.unpack_bayer(data)
H, W = ref.shape


def raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def test_out():
    out = np.full((H, W), 0xFFFF, dtype=np.uint16)
    got, m = irradiance.unpack_bayer(data, out=out)
    assert got is out, "out= must return the same array"
    assert np.array_equal(out, ref)
    assert (m.width, m.height) == (meta.width, meta.height)

    # Rejections: each must be a clean exception, never a crash or a silent misfill.
    print("  wrong shape:", raises(ValueError, irradiance.unpack_bayer, data, out=np.zeros((H, W + 1), np.uint16)))
    print("  transposed shape:", raises(ValueError, irradiance.unpack_bayer, data, out=np.zeros((W, H), np.uint16)))
    print("  Fortran order:", raises(ValueError, irradiance.unpack_bayer, data, out=np.zeros((H, W), np.uint16, order="F")))
    strided = np.zeros((H, 2 * W), np.uint16)[:, ::2]
    print("  strided view:", raises(ValueError, irradiance.unpack_bayer, data, out=strided))
    ro = np.zeros((H, W), np.uint16)
    ro.flags.writeable = False
    print("  read-only:", raises(ValueError, irradiance.unpack_bayer, data, out=ro))
    print("  wrong dtype:", raises(TypeError, irradiance.unpack_bayer, data, out=np.zeros((H, W), np.int32)))
    print("  1-D:", raises(TypeError, irradiance.unpack_bayer, data, out=np.zeros(H * W, np.uint16)))
    # A hostile file with out= still raises ValueError.
    raises(ValueError, irradiance.unpack_bayer, data[:1000], out=np.zeros((H, W), np.uint16))

    try:
        import torch
    except ImportError:
        print("  torch not installed: skipped torch.from_numpy check")
    else:
        t = torch.empty((H, W), dtype=torch.uint16) if hasattr(torch, "uint16") else None
        if t is not None:
            irradiance.unpack_bayer(data, out=t.numpy())
            assert np.array_equal(t.numpy(), ref)
            print("  torch tensor .numpy() view: ok")


def test_meta_types():
    assert meta.cfa_pattern == [0, 1, 1, 2] and type(meta.cfa_pattern) is list
    assert all(type(v) is int for v in meta.cfa_pattern)
    assert type(meta.cfa_plane_color) is list and type(meta.black_levels) is list
    print(f"  cfa_pattern={meta.cfa_pattern} plane_color={meta.cfa_plane_color} "
          f"black_levels={meta.black_levels} active_area={meta.active_area}")


def forge_cfa(pattern):
    """Rewrite the 4 inline CFAPattern bytes (sensor IFD) to another phase."""
    b = bytearray(data)
    hits = 0
    for e, tag, typ, count in ifd_entries(data):
        if tag == TAG_CFA_PATTERN and count == 4:
            b[e + 8:e + 12] = bytes(pattern)
            hits += 1
    assert hits, "no CFAPattern entry found"
    return bytes(b)


def reference_4ch(plane, pattern):
    """Independent numpy reference: [R, Gr, Gb, B], Gr on red's row."""
    pos = [(i // 2, i % 2) for i in range(4)]
    r = pos[pattern.index(0)]
    b = pos[pattern.index(2)]
    greens = [p for p, c in zip(pos, pattern) if c == 1]
    gr = next(g for g in greens if g[0] == r[0])
    gb = next(g for g in greens if g[0] == b[0])
    h2, w2 = plane.shape[0] // 2, plane.shape[1] // 2
    return np.stack([plane[y::2, x::2][:h2, :w2] for y, x in (r, gr, gb, b)])


def test_pack_phases():
    for name, pattern in [("RGGB", [0, 1, 1, 2]), ("GRBG", [1, 0, 2, 1]),
                          ("GBRG", [1, 2, 0, 1]), ("BGGR", [2, 1, 1, 0])]:
        plane, m = irradiance.unpack_bayer(forge_cfa(pattern))
        got = irradiance.pack_bayer_4ch(plane, m)
        assert got.dtype == np.uint16 and got.shape == (4, H // 2, W // 2), got.shape
        assert np.array_equal(got, reference_4ch(plane, pattern)), name
        print(f"  {name}: matches numpy reference")
    for bad in ([0, 0, 1, 2], [0, 1, 2, 1], [1, 1, 0, 2], [0, 1, 1, 3]):
        plane, m = irradiance.unpack_bayer(forge_cfa(bad))
        print(f"  {bad}:", raises(ValueError, irradiance.pack_bayer_4ch, plane, m))


def test_pack_normalize():
    assert (meta.cfa_repeat_dim, meta.black_level_repeat_dim) == ((2, 2), (2, 2))
    got = irradiance.pack_bayer_4ch(ref, meta, normalize=True)
    assert got.dtype == np.float32
    # Black pattern origin is the ActiveArea top-left (even here, so slot == black index).
    top, left = meta.active_area[0], meta.active_area[1]
    lv = np.array(meta.black_levels, dtype=np.float32).reshape(2, 2)
    blacks = [lv[(y - top) % 2, (x - left) % 2] for y, x in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    white = np.float32(meta.white_level)
    raw = reference_4ch(ref, [0, 1, 1, 2]).astype(np.float32)
    want = np.stack([(raw[k] - blacks[k]) / (white - blacks[k]) for k in range(4)])
    assert np.array_equal(got, want), "normalize must match the float32 formula bit-for-bit"
    print(f"  blacks per channel {blacks}, range [{got.min():.4f}, {got.max():.4f}] (unclipped)")


def test_pack_out():
    out = np.zeros((4, H // 2, W // 2), np.uint16)
    assert irradiance.pack_bayer_4ch(ref, meta, out=out) is out
    assert np.array_equal(out, irradiance.pack_bayer_4ch(ref, meta))
    outf = np.zeros((4, H // 2, W // 2), np.float32)
    assert irradiance.pack_bayer_4ch(ref, meta, out=outf, normalize=True) is outf
    assert np.array_equal(outf, irradiance.pack_bayer_4ch(ref, meta, normalize=True))

    print("  u16 out with normalize:", raises(TypeError, irradiance.pack_bayer_4ch, ref, meta, out=out, normalize=True))
    print("  wrong out shape:", raises(ValueError, irradiance.pack_bayer_4ch, ref, meta, out=np.zeros((4, H // 2, W // 2 + 1), np.uint16)))
    print("  out aliases plane:", raises(ValueError, irradiance.pack_bayer_4ch, ref, meta, out=ref.reshape(4, H // 2, W // 2)))
    print("  cropped plane:", raises(ValueError, irradiance.pack_bayer_4ch, ref[1:, 1:].copy(), meta))
    print("  Fortran plane:", raises(ValueError, irradiance.pack_bayer_4ch, np.asfortranarray(ref), meta))
    print("  float plane:", raises(TypeError, irradiance.pack_bayer_4ch, ref.astype(np.float32), meta))
    if os.path.exists(MONO):
        mplane, mmeta = irradiance.unpack_bayer(open(MONO, "rb").read())
        print("  monochrome:", raises(ValueError, irradiance.pack_bayer_4ch, mplane, mmeta))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("api: all passed")
