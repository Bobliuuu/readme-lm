#!/usr/bin/env python3
"""decode.py "<input text>" SVG_FILE : render the generated SVG filter network in headless Chrome at 1x pixel ratio (as a
README <img> would) and READ BACK the next-character predictions the browser's filter engine drew (each predicted glyph is matched
exactly against the glyph atlas). Nothing is computed here: this only reads pixels. Prints
{"input","windows","pred","unmatched","render_s"}; in 'pred' a space is shown as a middle dot."""
import sys, os, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build, gen, harness
text, trunc = gen.sanitize(sys.argv[1]); path = sys.argv[2]
svg, N, c, L, ctx, hdr = gen.make(text, trunc)
m = re.search(r'<svg[^>]*?width="(\d+)"[^>]*?height="(\d+)"', svg) or re.search(r'<svg[^>]*?height="(\d+)"[^>]*?width="(\d+)"', svg)
W, H = int(m.group(1)), int(m.group(2))
a, dt = harness.render(path, W, H, img=True)
atlas = np.load(os.path.join(HERE, "model_atlas.npz"))["atlas"]
pred, bad = "", 0
for r in range(N):
    blk = a[hdr + r * 24: hdr + r * 24 + 24, L["PX"]: L["PX"] + 16, 0]
    ids = [i for i in range(32) if (blk == atlas[:, 16 * i:16 * i + 16]).all()]
    if ids: ch = build.VOC[ids[0]]; pred += "·" if ch == " " else ch
    else: pred += "?"; bad += 1
print(json.dumps({"input": text, "windows": N, "pred": pred, "unmatched": bad, "render_s": round(dt, 2)}, ensure_ascii=False))
