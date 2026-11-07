# Does develop honour the ActiveArea origin for CFA phase and BlackLevel pattern?
# Grey subject, mosaic laid out from the ActiveArea top-left (DNG convention);
# rawpy (LibRaw) is the reference. Run from the repo root (needs rawpy):
#   .venv/bin/python notes/active_area_origin_repro.py
import sys, struct, io, numpy as np
sys.path.insert(0, "tests")
import synth, irradiance, rawpy
W, H = 256, 192
black, white = [512, 512, 512, 520], 16000
asn = [0.606276, 1.0, 0.461885]
colour = [0, 1, 1, 2]
def make(top, left):
    px = bytearray()
    for y in range(H):
        for x in range(W):
            i = ((y - top) % 2) * 2 + (x - left) % 2
            v = black[i] + round(0.2 * (white - black[i]) * asn[colour[i]])
            px += struct.pack("<H", v)
    entries = [
        (254, synth.LONG, [0]), (256, synth.LONG, [W]), (257, synth.LONG, [H]),
        (258, synth.SHORT, [16]), (259, synth.SHORT, [1]), (262, synth.SHORT, [synth.PHOTOMETRIC_CFA]),
        (277, synth.SHORT, [1]), (33421, synth.SHORT, [2, 2]), (33422, synth.BYTE, [0, 1, 1, 2]),
        (50706, synth.BYTE, [1, 4, 0, 0]), (50713, synth.SHORT, [2, 2]), (50714, synth.LONG, black),
        (50717, synth.LONG, [white]), (50829, synth.LONG, [top, left, H - 2, W - 4]),
    ] + list(synth.CANON_COLOR)
    return synth.tiff(entries, bytes(px), strip_rows=64, row_bytes=2 * W)
for top, left in [(2, 4), (2, 5), (3, 4)]:
    d = make(top, left)
    g = irradiance.develop(d)[0][top + 8:H - 10, left + 8:W - 12].reshape(-1, 3)
    with rawpy.imread(io.BytesIO(d)) as r:
        rr = r.postprocess(gamma=(1, 1), use_camera_wb=True, output_bps=16, no_auto_bright=True, user_flip=0) / 65535
        rr = rr[8:-8, 8:-8].reshape(-1, 3)
        rshape = r.raw_image_visible.shape
    print(f"ActiveArea origin ({top},{left}): irradiance mean {g.mean(0).round(3)} spread {(g.max(1)-g.min(1)).max():.3f} | "
          f"rawpy mean {rr.mean(0).round(3)} spread {(rr.max(1)-rr.min(1)).max():.3f} visible {rshape}")
print("pack_bayer_4ch, normalised channel means [R,Gr,Gb,B], expect", np.round([0.2*asn[0], .2, .2, 0.2*asn[2]], 3))
for top, left in [(2, 4), (2, 5), (3, 4)]:
    plane, meta = irradiance.unpack_bayer(make(top, left))
    p = irradiance.pack_bayer_4ch(plane, meta, normalize=True)
    print(f"  ({top},{left}) {[round(float(c[4:-4, 4:-4].mean()), 3) for c in p]}")
