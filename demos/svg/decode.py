#!/usr/bin/env python3
"""Decode a rendered live picture (pixels only) and compare it with the integer reference.
  python3 demos/svg/decode.py "<text>"  -> renders gen.make(text) in headless Chrome (DPR 1, via <img>), prints JSON with the network's predicted
  glyph of every window (matched against the glyph atlas), the reference, and exactness counts for every pixel block of the picture."""
import os, sys, json, tempfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build, view, gen, harness, train
GRAY = np.load(os.path.join(HERE, "model_atlas.npz"))["atlas"]
ATL = {t: view.color_atlas(GRAY, t) for t in view.INK}

def glyph_at(a, y, x, tag):
    """index of the atlas glyph equal to the 16x24 block at (y,x), or -1 (pixel-exact match required)"""
    blk = a[y:y + 24, x:x + 16]
    if not (blk[:, :, 3] == 255).all(): return -1
    m = [i for i in range(32) if (blk[:, :, :3] == ATL[tag][:, 16 * i:16 * i + 16]).all()]
    return m[0] if len(m) == 1 else -1

def decode(a, info, M):
    """a: RGBA screenshot of the picture (viewport = svg size).  Returns dict of counts + decoded predicted string."""
    ctxids = info["ctxids"]; h, q, pred = None, None, None
    _, _, h, q = train.int_forward(M, np.array(ctxids)); pred = q.argmax(1)
    hdr, wy0, N, T, CTX, H = view.HDR, info["wy0"], info["N"], info["T"], info["CTX"], info["H"]; bw = info["band"]
    tr, tg, tb = view.heat_tables(); res = dict(heat=[0, 0], pred=[0, 0], ctx=[0, 0], actual=[0, 0], text=[0, 0]); got_pred = ""
    text_ids = [info["idx"][ch] for ch in info["text"]]
    for m, i in enumerate(text_ids):
        res["text"][1] += 1; res["text"][0] += glyph_at(a, hdr + 24 * (m // view.CPL), view.TX0 + 16 * (m % view.CPL), "main") == i
    for r in range(N):
        y = hdr + wy0 + r * 24
        for j in range(H):
            b, jj = divmod(j, bw); e = int(h[r, j]); blk = a[y + b * view.CELL_H:y + (b + 1) * view.CELL_H, view.HX + jj * view.CELL_W:view.HX + (jj + 1) * view.CELL_W]
            res["heat"][1] += 1; res["heat"][0] += bool((blk[:, :, 0] == tr[e]).all() and (blk[:, :, 1] == tg[e]).all() and (blk[:, :, 2] == tb[e]).all() and (blk[:, :, 3] == 255).all())
        for k in range(CTX): res["ctx"][1] += 1; res["ctx"][0] += glyph_at(a, y, view.CTX_X0 + 16 * k, "ctx") == ctxids[r][k]
        res["actual"][1] += 1; res["actual"][0] += glyph_at(a, y, view.AX, "main") == info["act"][r]
        g = glyph_at(a, y, view.PX, "pred"); res["pred"][1] += 1; res["pred"][0] += g == int(pred[r]); got_pred += build.VOC[g] if g >= 0 else "?"
    out = {k: f"{v[0]}/{v[1]}" for k, v in res.items()}; out["decoded_prediction"] = got_pred; out["reference_prediction"] = "".join(build.VOC[i] for i in pred)
    out["all_exact"] = all(v[0] == v[1] for v in res.values()) and got_pred == out["reference_prediction"]
    return out

def main():
    text, trunc = gen.sanitize(sys.argv[1]); M = gen.load_model(); svg, info = gen.make(text, trunc, M); info["text"] = text
    p = os.path.join(tempfile.mkdtemp(prefix="dec_"), "t.svg"); open(p, "w").write(svg)
    a, dt = harness.render(p, view.DW, info["vh"], img=True); out = decode(a, info, M); out["render_s"] = round(dt, 2); out["input"] = text
    print(json.dumps(out))
    import shutil; shutil.rmtree(os.path.dirname(p), ignore_errors=True)
if __name__ == "__main__": main()
