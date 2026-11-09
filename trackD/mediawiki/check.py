#!/usr/bin/env python3
"""Render README.mediawiki with WikiCloth (and via github-markup) and compare every live cell
with an independent numpy/python-int forward pass from model.npz."""
import subprocess, re, sys, time, numpy as np
from html.parser import HTMLParser
here = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
M = dict(np.load(f"{here}/model.npz"))
H, D = M['W1'].shape[0], M['E'].shape[1]; NB = M['V'].shape[0]
VOC = "abcdefghijklmnopqrstuvwxyz .,-'?!_"[:32]

def fwd(c1, c2):
    x = np.concatenate([M['E'][c1], M['E'][c2]]).astype(np.int64)
    h = np.maximum(0, M['W1'].astype(np.int64) @ x + M['b1'])
    z = M['V'].astype(np.int64) @ h + M['b2']
    b = (z > 0).astype(int)
    return h, z, b, int("".join(map(str, b)), 2)

class T(HTMLParser):
    def __init__(s): super().__init__(); s.tables=[]; s.row=None; s.cell=None
    def handle_starttag(s, t, a):
        if t == 'table': s.tables.append([])
        if t == 'tr': s.row = []; s.tables[-1].append(s.row)
        if t in ('td', 'th'): s.cell = ''
    def handle_endtag(s, t):
        if t in ('td', 'th') and s.cell is not None: s.row.append(s.cell.strip()); s.cell = None
    def handle_data(s, d):
        if s.cell is not None: s.cell += d

def render(mode):
    out = f"{here}/out_{mode}.html"
    args = ["ruby", f"{here}/render.rb", f"{here}/README.mediawiki", out] + (["github"] if mode == "github" else [])
    t = time.time(); r = subprocess.run(args, capture_output=True, text=True); dt = time.time()-t
    assert r.returncode == 0, r.stderr
    return open(out).read(), dt, r.stderr.strip()

def check(mode):
    html, dt, info = render(mode)
    p = T(); p.feed(html)
    t = [tb for tb in p.tables if tb and tb[0] and tb[0][0] == 'n' ]
    assert "Expression error" not in html and "{{#" not in re.sub(r"<code>.*?</code>", "", html), "unevaluated/err markup present"
    # section 1
    rows = [r for r in t[0][1:]]
    ok = 0
    for r in rows:
        a, b = r[1][0], r[1][1]
        a = ' ' if a == '␣' else a; b = ' ' if b == '␣' else b
        ref = VOC[fwd(VOC.index(a), VOC.index(b))[3]]; ref = '␣' if ref == ' ' else ref
        live = r[3].replace('_', '␣')
        ok += (live == ref)
    print(f"[{mode}] section1: {ok}/{len(rows)} live predictions == numpy  (render+parse {dt:.2f}s, html {len(html)} B)")
    # section 2 (hidden + logits)
    m = re.search(r'Inside the network: input <code>(..)</code>', html)
    pair = m.group(1); a = ' ' if pair[0]=='␣' else pair[0]; b = ' ' if pair[1]=='␣' else pair[1]
    h, z, bits, _ = fwd(VOC.index(a), VOC.index(b))
    hl = [float(r[1]) for r in [tb for tb in p.tables if tb and tb[0][0]=='unit'][0][1:]]
    zt = [tb for tb in p.tables if tb and tb[0][0]=='bit'][0][1:]
    zl = [float(r[1]) for r in zt]; bl = [int(r[3]) for r in zt]
    herr = max(abs(np.array(hl)-h)); zerr = max(abs(np.array(zl)-z)); berr = sum(np.array(bl)!=bits)
    print(f"[{mode}] section2 ({pair}): max|h err|={herr}, max|z err|={zerr}, bit mismatches={berr}, hidden exact {sum(np.array(hl)==h)}/{H}, logits exact {sum(np.array(zl)==z)}/{NB}")
    return ok == len(rows) and herr == 0 and zerr == 0 and berr == 0

good = check("wikicloth") & check("github")
print("ALL EXACT" if good else "MISMATCH"); sys.exit(0 if good else 1)
