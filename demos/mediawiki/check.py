#!/usr/bin/env python3
"""Local verification: generate pages for several inputs, render with WikiCloth and GitHub::Markup,
compare every live cell with an independent python-int forward pass from model.npz.
Usage: python3 check.py            (reuses verify_html() in check_live.py)"""
import os, re, subprocess, sys, time, tempfile, json
import numpy as np
from html.parser import HTMLParser
here = os.path.dirname(os.path.abspath(__file__))
M = {k: v for k, v in np.load(f"{here}/model.npz").items()}
H, D = M['W1'].shape[0], M['E'].shape[1]; NB = M['V'].shape[0]
VOC = "abcdefghijklmnopqrstuvwxyz .,-'?"

def fwd(c1, c2):
    x = [int(v) for v in M['E'][c1]] + [int(v) for v in M['E'][c2]]
    h = [max(0, sum(int(M['W1'][j][i]) * x[i] for i in range(2*D)) + int(M['b1'][j])) for j in range(H)]
    z = [sum(int(M['V'][k][j]) * h[j] for j in range(H)) + int(M['b2'][k]) for k in range(NB)]
    b = [1 if v > 0 else 0 for v in z]
    return h, z, b, int("".join(map(str, b)), 2)

class T(HTMLParser):
    def __init__(s): super().__init__(convert_charrefs=True); s.tables = []; s.row = None; s.cell = None
    def handle_starttag(s, t, a):
        if t == 'table': s.tables.append([])
        if t == 'tr' and s.tables: s.row = []; s.tables[-1].append(s.row)
        if t in ('td', 'th'): s.cell = ''
    def handle_endtag(s, t):
        if t in ('td', 'th') and s.cell is not None and s.row is not None: s.row.append(s.cell.strip()); s.cell = None
    def handle_data(s, d):
        if s.cell is not None: s.cell += d

def dec(ch): return ' ' if ch == '␣' else ch
def enc(ch): return '␣' if ch == ' ' else ch

def verify_html(html, label):
    """returns (ok_cells, total_cells, notes)"""
    p = T(); p.feed(html)
    bad = []
    if "Expression error" in html: bad.append("Expression error present")
    if "{{#" in re.sub(r"<code>.*?</code>", "", html, flags=re.S): bad.append("unevaluated {{# outside code")
    t1 = [tb for tb in p.tables if tb and tb[0] and tb[0][0] == 'n']
    ok = tot = 0
    wins = []
    if t1:
        for r in t1[0][1:]:
            a, b = dec(r[1][0]), dec(r[1][1])
            ref = enc(VOC[fwd(VOC.index(a), VOC.index(b))[3]])
            tot += 1; ok += (r[3] == ref); wins.append((a, b))
    else: bad.append("no window table")
    if wins:
        h, z, bits, c = fwd(VOC.index(wins[-1][0]), VOC.index(wins[-1][1]))
        ut = [tb for tb in p.tables if tb and tb[0][0] == 'unit']
        bt = [tb for tb in p.tables if tb and tb[0][0] == 'bit']
        if ut and bt:
            hl = [r[1] for r in ut[0][1:]]
            zl = [r[1] for r in bt[0][1:]]; bl = [r[2] for r in bt[0][1:]]
            def eq(s, v):
                try: return float(s) == v
                except ValueError: return False
            for s, v in zip(hl, h): tot += 1; ok += eq(s, v)
            for s, v in zip(zl, z): tot += 1; ok += eq(s, v)
            for s, v in zip(bl, bits): tot += 1; ok += eq(s, v)
            m = re.search(r"Decoded next character \(live\): (\S+)", " ".join(html.split()))
            tot += 1
            txt = re.sub(r"<[^>]+>", "", re.search(r"Decoded next character \(live\):(.*?)</", html, re.S).group(1)).strip() if "Decoded next" in html else ""
            ok += (txt == enc(VOC[c]))
        else: bad.append("no worked-example tables")
    return ok, tot, bad, "".join(w[0] for w in wins) + (wins[-1][1] if wins else "")

def render(mode, src, out):
    args = ["ruby", f"{here}/render.rb", src, out] + (["github"] if mode == "github" else [])
    t = time.time(); r = subprocess.run(args, capture_output=True, text=True); dt = time.time() - t
    assert r.returncode == 0, r.stderr
    return open(out).read(), dt

def indep_sanitize(s):
    s = "".join(c for c in " ".join(s.lower().split()) if c in VOC)
    return " ".join(s.split())

if __name__ == "__main__":
    inputs = {
        "short": "the quick brown fox",
        "long-truncated": "the quick brown fox jumps over the lazy dog. " * 30,
        "all-vocab": "abcdefghijklmnopqrstuvwxyz .,-'?",
        "hostile1": "{{#expr:1}}", "hostile2": "[[x]] [http://e.x y]", "hostile3": "<script>alert(1)</script>",
        "hostile4": "|}\n== x ==\n{| class=a\n|-\n|", "hostile5": "{{#switch:1|1=<b>x</b>}} ''it's'' '''b''' ~~~~ __TOC__ &#60;",
        "single-char": "a", "empty": "!!!"[0:0] + "{}#", "mixed-case-ws": "  Hello\t\tWORLD\n\n Test  ",
    }
    scratch = tempfile.mkdtemp()
    allok = True; tt = ttot = 0
    try:
        for name, inp in inputs.items():
            src = f"{scratch}/{name}.mediawiki"
            r = subprocess.run([sys.executable, f"{here}/gen.py", inp, "-o", src], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            meta = json.loads(r.stdout)
            page = open(src, encoding="utf-8").read()
            # source hygiene: only whitelisted parser functions / tags, no links/html from input
            funcs = set(re.findall(r"\{\{(#?\w+)", page))
            hyg = funcs <= {"#expr", "#ifeq", "#switch"} and "[[" not in page and "http" not in page and "<script" not in page.lower()
            tags = set(re.findall(r"</?(\w+)", page))
            hyg = hyg and tags <= {"code", "b"} and meta["bytes"] == len(page.encode()) and meta["bytes"] <= 420000
            want = meta["input"]
            for mode in ("wikicloth", "github"):
                html, dt = render(mode, src, f"{scratch}/o_{mode}.html")
                ok, tot, bad, got = verify_html(html, name)
                exp_in = want if len(want) >= 2 else want
                in_match = (got == want)
                tt += ok; ttot += tot
                good = ok == tot and not bad and in_match and hyg
                allok &= good
                print(f"{name:15s} [{mode:9s}] windows={meta['windows']:3d} bytes={meta['bytes']:6d} cells {ok}/{tot} exact, "
                      f"input echoed==sanitized:{in_match} hygiene:{hyg} render {dt:.2f}s {bad or ''}")
            sanity = indep_sanitize(inp)
            if not (want == sanity or sanity.startswith(want) or len(sanity) < 2):
                print("  note: generator sanitize differs from independent sanitize:", repr(want), repr(sanity)); allok = False
    finally:
        subprocess.run(["rm", "-rf", scratch])
    print(f"TOTAL exact cells {tt}/{ttot} = {100*tt/ttot:.2f}%")
    print("ALL EXACT" if allok and tt == ttot else "MISMATCH"); sys.exit(0 if allok and tt == ttot else 1)
