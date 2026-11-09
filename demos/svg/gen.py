#!/usr/bin/env python3
"""Generate an SVG whose filter graph IS a neural network (no JavaScript).  The viewer's browser does every multiply-accumulate
(feConvolveMatrix), every ReLU/lookup (feComponentTransfer), the argmax (feMorphology) and the glyph drawing (feDisplacementMap).
This script only turns the input text into pixels (one-hot context codes) and writes the fixed network around them.

  python3 demos/svg/gen.py "<input text>" -o demos/svg/latest.svg      -> prints one JSON line {"input","windows","bytes"}

The input is untrusted (a public issue title): it is reduced to the 32-symbol vocabulary [a-z .,-'?] and only ever used as pixel
values / glyph-map indices; no input character is written into the SVG as text or markup."""
import argparse, json, os, re, sys, unicodedata
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build, svgnn

VOC = build.VOC
MAXLEN = 96                    # chars -> MAXLEN+1 windows; keeps render << 15 s at 1x DPR (see REPORT.md) and the file << 3 MB

def sanitize(s):
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = "".join(ch if ch in VOC else " " for ch in s if not unicodedata.combining(ch))   # vocabulary only (anything else -> space)
    s = re.sub(r" +", " ", s).strip()
    trunc = len(s) > MAXLEN
    return s[:MAXLEN].rstrip(), trunc

def make(text, trunc=False, M=None, RH=24, debug=False):
    M = M if M is not None else dict(np.load(os.path.join(HERE, "model.npz")))
    CTX, H = int(M["ctx"]), M["W1"].shape[2]
    idx = {c: i for i, c in enumerate(VOC)}
    s = " " * CTX + text; N = len(text) + 1
    ctx = [[idx[ch] for ch in s[j:j + CTX]] for j in range(N)]
    act = [idx[text[j]] if j < len(text) else idx[" "] for j in range(N)]
    c, L = build.build_q8(M, ctx, act, RH, debug=debug)
    hdr, DW = 58, L["DW"]
    t = lambda x, y, s, sz=10, fill="#9aa4b2": f'<text x="{x}" y="{y}" font-family="monospace" font-size="{sz}" fill="{fill}">{s}</text>'
    texts = [t(8, 17, "SVG neural net: next-character prediction computed by your browser's filter engine", 11, "#e6edf3"),
             t(8, 33, f"{H} ReLU units, {CTX}-character context, 8-bit integers, no code. Input = pixels.", 10),
             t(L["ctx_x"][0], 52, "context", 9), t(L["HX"], 52, f"hidden layer, {H} ReLU units (brighter = more active)", 9),
             t(L["PX"], 52, "predicted", 9), t(L["AX"], 52, "actual", 9)]
    cap = [f"{N} windows; each row slides a {CTX}-character window over the input. Per row the browser runs:",
           f"one-hot pixels -> {H} feConvolveMatrix neurons ({CTX*32} taps) -> feComponentTransfer ReLU/requantise LUT",
           f"-> 32 feConvolveMatrix logits ({H} taps) -> feMorphology argmax -> feDisplacementMap glyph atlas.",
           "Pure filter graph: weights = kernel numbers, activations = 8-bit pixels (bit-exact vs an integer reference).",
           "Computes correctly only in Chrome-engine browsers at 1x device-pixel ratio (100% zoom, non-Retina)."]
    if trunc: cap.append(f"(input truncated to the first {MAXLEN} characters)")
    vh = hdr + c.H + 22 + 14 * len(cap); c.vh = vh; y0 = hdr + c.H + 16
    texts += [t(8, y0 + 14 * i, s_, 10, "#f0b429" if (trunc and i == len(cap) - 1) else "#9aa4b2") for i, s_ in enumerate(cap)]
    svg = svgnn.svg_wrap(c, "".join(c.p), L["W"] if debug else DW, vh, hdr, "".join(texts), "SVG filter neural network")
    return svg, N, c, L, ctx, hdr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("-o", "--out", default=os.path.join(HERE, "latest.svg"))
    a = ap.parse_args()
    text, trunc = sanitize(a.text)
    svg, N, c, L, ctx, hdr = make(text, trunc)
    with open(a.out, "w") as f: f.write(svg)
    print(json.dumps({"input": text, "windows": N, "bytes": len(svg.encode())}))
if __name__ == "__main__": main()
