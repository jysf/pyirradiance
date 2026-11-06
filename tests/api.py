"""API tests: unpack_bayer(out=).

Usage: api.py <bayer.dng>
"""
import os
import sys

import numpy as np
import irradiance

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("api: all passed")
