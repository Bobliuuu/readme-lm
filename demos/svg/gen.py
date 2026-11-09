#!/usr/bin/env python3
"""Generate an SVG whose filter graph IS a neural network (no JavaScript).  The viewer's browser does every multiply-accumulate
(feConvolveMatrix), every ReLU/lookup (feComponentTransfer), the argmax (feMorphology) and the glyph drawing (feDisplacementMap).
This script only turns the input text into pixels (one-hot context codes, glyph indices) and writes the fixed network around them.

  python3 demos/svg/gen.py "<full text>" -o demos/svg/latest.svg      -> prints one JSON line {"input","windows","bytes"}

The input is untrusted (a public issue title): it is reduced to the 32-symbol vocabulary [a-z .,-'?] and only ever used as pixel
values / glyph-map indices; no input character is written into the SVG as text or markup.  The picture shows the whole text big and
the network re-predicting each of the last <= 12 characters live."""
import argparse, json, os, re, sys, unicodedata
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build, view

VOC = build.VOC
MAXLEN = 96

def sanitize(s):
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = "".join(ch if ch in VOC else " " for ch in s if not unicodedata.combining(ch))   # vocabulary only (anything else -> space)
    s = re.sub(r" +", " ", s).strip()
    return s[:MAXLEN].rstrip(), len(s) > MAXLEN

def load_model(): return dict(np.load(os.path.join(HERE, "model.npz")))

def make(text, trunc=False, M=None, debug=False):
    M = M if M is not None else load_model()
    c, info = view.build_live(M, text, debug)
    H, CTX = info["H"], info["CTX"]; rows = info["rows"]; hdr = view.HDR
    foot = view.footer_lines(trunc, MAXLEN); vh = hdr + rows + 16 + 16 * len(foot) + 12; info["vh"] = vh
    t = lambda x, y, s, sz=10, fill="#8b949e", w="normal": f'<text x="{x}" y="{y}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" font-size="{sz}" font-weight="{w}" fill="{fill}">{s}</text>'
    ty = hdr + info["TR"]
    texts = [t(28, 30, "README-LM", 13, "#58a6ff", "bold"), t(110, 30, f"a {CTX}-character-context neural net, drawn in SVG filters", 11),
             t(view.CTX_X0, ty + 20, "how your browser just computed it", 10, "#58a6ff", "bold"),
             t(view.CTX_X0, ty + 40, "context", 9), t(view.HX, ty + 40, f"{H} hidden ReLU units (brighter = more active)", 9), t(view.PX, ty + 40, "predicted", 9), t(view.AX, ty + 40, "actual", 9)]
    texts += [t(view.CTX_X0, hdr + rows + 18 + 16 * i, s, 10, "#f0b429" if (trunc and i == len(foot) - 1) else "#8b949e") for i, s in enumerate(foot)]
    body = "".join(c.p); W = info["L"]["W"]; vw = W if debug else view.DW
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{vw}" height="{vh}" viewBox="0 0 {vw} {vh}"><title>SVG filter neural network</title>'
           f'<defs><filter id="F" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{rows}" color-interpolation-filters="sRGB">{body}</filter></defs>'
           f'<rect width="{vw}" height="{vh}" fill="#010409"/><rect x="6" y="6" width="{view.DW-12}" height="{vh-12}" rx="14" fill="{view.CARD}" stroke="#30363d"/>'
           f'<g transform="translate(0,{hdr})"><rect width="{W}" height="{rows}" fill="#000" filter="url(#F)"/></g>{"".join(texts)}</svg>')
    return svg, info

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("-o", "--out", default=os.path.join(HERE, "latest.svg"))
    a = ap.parse_args(); text, trunc = sanitize(a.text)
    svg, info = make(text, trunc)
    with open(a.out, "w") as f: f.write(svg)
    print(json.dumps({"input": text, "windows": info["N"], "bytes": len(svg.encode())}))
if __name__ == "__main__": main()
