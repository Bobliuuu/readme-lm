#!/usr/bin/env python3
"""Greedy generation where EVERY next character is computed by the SVG filter network in headless Chrome.
  python3 demos/svg/generate.py "<prompt>" -n 60 > out.json   ->  {"input", "generated", "steps", "render_s"}
Per step: the last CTX characters become one window row (one-hot pixels), the browser evaluates the whole network (convolve, ReLU LUT,
convolve, feMorphology argmax) and the argmax id pixel is read back (the same id pixel the glyph lookup draws from).  Stops at n
characters (cap 80) or at a sentence end followed by a space."""
import argparse, json, os, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build, gen, harness, svgnn

def step_id(M, window_ids, path):
    c, L = build.build_q8(M, [window_ids], [0], 1, debug="id", disp=False)
    open(path, "w").write(svgnn.svg_wrap(c, "".join(c.p), L["W"], 1, 0, "", "step"))
    a, dt = harness.render(path, L["W"], 1)
    return int(a[0, V1, 0]), dt
V1 = 31      # argmax id column

def generate(prompt, n=60, M=None, log=None):
    M = M or gen.load_model(); CTX = int(M["ctx"]); idx = {c: i for i, c in enumerate(build.VOC)}
    text, _ = gen.sanitize(prompt); n = min(n, 80); s = " " * CTX + text; out = ""; total = 0.0
    d = tempfile.mkdtemp(prefix="genstep_"); p = os.path.join(d, "s.svg")
    try:
        while len(out) < n:
            i, dt = step_id(M, [idx[ch] for ch in s[-CTX:]], p); total += dt
            ch = build.VOC[i]; s += ch; out += ch
            if log: log(ch)
            if len(out) >= 2 and out[-1] == " " and out[-2] in ".?": break
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)
    return text, out.rstrip(), len(out), total

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("-n", type=int, default=60); a = ap.parse_args()
    t0 = time.time(); text, out, steps, rs = generate(a.prompt, a.n)
    print(json.dumps({"input": text, "generated": out, "steps": steps, "render_s": round(time.time() - t0, 2)}))
if __name__ == "__main__": main()
