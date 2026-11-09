#!/usr/bin/env python3
"""Exactness + safety checks for demos/svg.  Renders generated SVGs in headless Chrome (Playwright, DPR 1, embedded through <img>
like a README would), decodes the PIXELS and compares them with a pure-numpy integer reference.
  python3 demos/svg/check.py [stage1|stage2|safety|all]"""
import os, re, sys, json, time, tempfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build, svgnn, gen, harness, train
ATLAS = np.load(os.path.join(HERE, "model_atlas.npz"))["atlas"]
MW = dict(np.load(os.path.join(HERE, "..", "..", "trackD", "mediawiki", "model.npz")))
TMP = tempfile.mkdtemp(prefix="svgchk_")
IDX = {c: i for i, c in enumerate(build.VOC)}
def wr(svg, name="t.svg"):
    p = os.path.join(TMP, name); open(p, "w").write(svg); return p

def glyph_ok(a, y0, x0, ident):
    blk = a[y0:y0 + 24, x0:x0 + 16]; ref = ATLAS[:, 16 * ident:16 * ident + 16]
    return bool((blk[:, :, 0] == ref).all() and (blk[:, :, 1] == ref).all() and (blk[:, :, 2] == ref).all() and (blk[:, :, 3] == 255).all())

def decode_picture(a, hdr, L, N, RH, h_ref, pred_ref, ctx, act, cw, hx=None, nh=None, hmax=255, gain=1):
    """compare every pixel of the demo picture: heat cells, predicted glyph, context glyphs, actual glyph"""
    hr, hg, hb = svgnn.heat_tables(gain); hx = hx or L["HX"]; nh = nh or h_ref.shape[1]
    heat_ok = heat_n = gl_ok = gl_n = pr_ok = 0
    for r in range(N):
        y0 = hdr + r * RH
        for j in range(nh):
            e = min(int(h_ref[r, j]), hmax); blk = a[y0:y0 + RH, hx + j * cw:hx + (j + 1) * cw]
            ok = (blk[:, :, 0] == hr[e]).all() and (blk[:, :, 1] == hg[e]).all() and (blk[:, :, 2] == hb[e]).all(); heat_ok += bool(ok); heat_n += 1
        pr_ok += glyph_ok(a, y0, L["PX"], int(pred_ref[r]))
        for k, X in enumerate(L["ctx_x"]): gl_ok += glyph_ok(a, y0, X, ctx[r][k]); gl_n += 1
        gl_ok += glyph_ok(a, y0, L["AX"], act[r]); gl_n += 1
    return dict(heat=f"{heat_ok}/{heat_n}", pred_glyph=f"{pr_ok}/{N}", ctx_actual_glyph=f"{gl_ok}/{gl_n}")

def check_stage2(text, trunc=False, verbose=True):
    M = dict(np.load(os.path.join(HERE, "model.npz"))); CTX = int(M["ctx"]); out = {}
    svg, N, c, L, ctx, hdr = gen.make(text, trunc, M)
    act = [IDX[text[j]] if j < len(text) else IDX[" "] for j in range(N)]
    h, q, pred = build.ref_q8(M, ctx)
    p = wr(svg); vw = L["DW"]; vh = c.vh
    a, dt = harness.render(p, vw, vh, img=True)
    out.update(text=text, windows=N, bytes=len(svg.encode()), prims=c.n, render_s=round(dt, 2))
    out.update(decode_picture(a, hdr, L, N, 24, h, pred, ctx, act, L['CW'], gain=build.HEAT_GAIN_Q8))
    # raw computation columns: hidden codes, logits, id (debug renders)
    for mode in ("id", "lg"):
        svg2, *_ = gen.make(text, trunc, M, debug=mode); a2, dt2 = harness.render(wr(svg2, "d.svg"), L["W"], vh)
        rows = [hdr + r * 24 + 5 for r in range(N)]
        if mode == "id": out["hidden_codes"] = f"{int((a2[rows][:, L['HB']:L['HB']+h.shape[1], 0] == h).sum())}/{h.size}"; out["argmax_id"] = f"{int((a2[rows, 31, 0] == pred).sum())}/{N}"
        else: out["logit_codes"] = f"{int((a2[rows][:, 0:32, 0] == q).sum())}/{q.size}"
    out["pred_text"] = "".join(build.VOC[i] for i in pred); out["all_exact"] = all(v.split("/")[0] == v.split("/")[1] for k, v in out.items() if isinstance(v, str) and "/" in v)
    if verbose: print(json.dumps(out))
    return out

def forbidden(svg):
    low = svg.lower(); bad = []
    for pat in ("<script", "script", "onload", "onerror", "onclick", "javascript:", "href=\"http", "href='http", "xlink:href=\"http", "<foreignobject", "<style", "<iframe", "<a ", "<use", "@import", "url(http", "src="):
        if pat in low: bad.append(pat)
    for m in re.finditer(r'(?:xlink:)?href="([^"]*)"', svg):
        if not m.group(1).startswith("data:image/png;base64,"): bad.append("href->" + m.group(1)[:30])
    if re.search(r'https?://', svg.replace("http://www.w3.org/2000/svg", "")): bad.append("http(s) url")
    return bad

HOSTILE = ["</svg><script>alert(1)</script>", "\"><image href=x onerror=alert(1)>", "&lt;svg onload=alert(1)&gt;", "{{x}}", "<img src=//evil.example/x.png>",
           "javascript:alert(1)", "'; DROP TABLE x;--", "\x00\x1f‮ rtl 😀 éè NFKD", "<![CDATA[ ]]> <!-- --> ]]>", "  \n\t multiple   spaces\r\nand newlines  ", "", "a" * 500]
def check_safety():
    res = []; ok_all = True
    for t in HOSTILE:
        s, tr = gen.sanitize(t); svg, N, *_ = gen.make(s, tr); bad = forbidden(svg)
        charset_ok = all(ch in build.VOC for ch in s) and "  " not in s
        # the only non-markup text in the file is the fixed caption: input chars never appear as text nodes
        texts = " ".join(re.findall(r">([^<]*)<", svg)); leak = bool(len(s) >= 6 and s in texts)
        res.append(dict(input=repr(t)[:40], sanitized=s[:40], windows=N, bytes=len(svg), forbidden=bad, charset_ok=charset_ok, input_in_text_nodes=leak))
        ok_all &= (not bad) and charset_ok and not leak
        # well-formed XML
        import xml.dom.minidom; xml.dom.minidom.parseString(svg)
    # CLI contract
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(HERE, "gen.py"), "the quick brown fox", "-o", os.path.join(TMP, "cli.svg")], capture_output=True, text=True)
    cli = json.loads(r.stdout.strip().splitlines()[-1]); assert set(cli) == {"input", "windows", "bytes"} and cli["input"] == "the quick brown fox", cli
    for r_ in res: print(json.dumps(r_))
    print("safety:", "PASS" if ok_all else "FAIL", "| cli:", cli)
    return ok_all

def check_stage1():
    """MediaWiki toy as a filter graph: ALL 1024 two-char contexts, compute-only (hidden/logit/bit/id columns), then pictures (predicted glyph)."""
    ctx = [[a, b] for a in range(32) for b in range(32)]; act = [0] * len(ctx)
    Hh, Z, bits, ids = build.ref_mw(MW, ctx); res = {}
    for mode in ("hid", "z", "bits", "id"):
        c, L = build.build_mw(MW, ctx, act, 1, debug=mode, disp=False)
        a, dt = harness.render(wr(svgnn.svg_wrap(c, "".join(c.p), L["W"], c.H, 0, "", "t")), L["W"], c.H)
        g = a[:, :, 0].astype(np.int64)
        if mode == "hid": got = g[:, L["HB"]:L["HB"] + 12] + g[:, L["HB"] + 12:L["HB"] + 24]; res["hidden relu(W1 x+b1) exact"] = f"{int((got == Hh).sum())}/{Hh.size}"
        if mode == "z": got = g[:, L["ZB"]:L["ZB"] + 5]; res["logit codes clamp(z,0,255) exact"] = f"{int((got == np.clip(Z, 0, 255)).sum())}/{Z.size}"; res["logit sign (z>0) exact"] = f"{int(((got > 0) == (Z > 0)).sum())}/{Z.size}"
        if mode == "bits": got = g[:, L["ZB"]:L["ZB"] + 5]; res["bits exact"] = f"{int((got == bits).sum())}/{bits.size}"
        if mode == "id": res["decoded char id exact"] = f"{int((g[:, L['ZB'] + 5] == ids).sum())}/{len(ids)}"
        res[f"render_s[{mode}] (1024 rows)"] = round(dt, 2)
    # pictures in chunks (RH=24): predicted glyph pixels == atlas glyph of the numpy/MediaWiki prediction; heat + context glyphs too
    pic = dict(heat=0, heat_n=0, pred=0, pred_n=0, cg=0, cg_n=0); dts = []; CH = 64
    for s0 in range(0, 1024, CH):
        cx = ctx[s0:s0 + CH]; n = len(cx); c, L = build.build_mw(MW, cx, act[:n], 24)
        a, dt = harness.render(wr(svgnn.svg_wrap(c, "".join(c.p), L["DW"], c.H, 0, "", "t"), "p.svg"), L["DW"], c.H, img=True); dts.append(dt)
        d = decode_picture(a, 0, L, n, 24, Hh[s0:s0 + CH], ids[s0:s0 + CH], cx, act[:n], 16, nh=12)
        for k, key in (("heat", "heat"), ("pred_glyph", "pred"), ("ctx_actual_glyph", "cg")):
            x, y = map(int, d[k].split("/")); pic[key] += x; pic[key + "_n"] += y
    res["picture: heat cells exact"] = f"{pic['heat']}/{pic['heat_n']}"; res["picture: predicted glyph exact"] = f"{pic['pred']}/{pic['pred_n']}"; res["picture: context/actual glyphs"] = f"{pic['cg']}/{pic['cg_n']}"
    res["picture render_s per 64 windows (max)"] = round(max(dts), 2)
    # cross-check with the MediaWiki renderings: WikiCloth live column and numpy column of README.mediawiki
    from html.parser import HTMLParser
    class T(HTMLParser):
        def __init__(s): super().__init__(); s.tables = []; s.row = None; s.cell = None
        def handle_starttag(s, t, a):
            if t == "table": s.tables.append([])
            if t == "tr": s.row = []; s.tables[-1].append(s.row)
            if t in ("td", "th"): s.cell = ""
        def handle_endtag(s, t):
            if t in ("td", "th") and s.cell is not None: s.row.append(s.cell.strip()); s.cell = None
        def handle_data(s, d):
            if s.cell is not None: s.cell += d
    html = open(os.path.join(HERE, "..", "..", "trackD", "mediawiki", "out_wikicloth.html")).read(); p = T(); p.feed(html)
    rows = [r for r in [tb for tb in p.tables if tb and tb[0] and tb[0][0] == "n"][0][1:]]
    sp = lambda ch: " " if ch in ("␣", "_") else ch
    pairs = [(IDX[sp(r[1][0])], IDX[sp(r[1][1])], sp(r[3]), sp(r[4])) for r in rows]
    cx = [[a, b] for a, b, _, _ in pairs]; c, L = build.build_mw(MW, cx, [0] * len(cx), 24)
    a, dt = harness.render(wr(svgnn.svg_wrap(c, "".join(c.p), L["DW"], c.H, 0, "", "t"), "m.svg"), L["DW"], c.H, img=True)
    svg_pred = []
    for r in range(len(cx)):
        match = [i for i in range(32) if glyph_ok(a, r * 24, L["PX"], i)]; svg_pred.append(build.VOC[match[0]] if len(match) == 1 else "?")
    res["MediaWiki rows (WikiCloth live == numpy col)"] = f"{sum(p_[2] == p_[3] for p_ in pairs)}/{len(pairs)}"
    res["SVG glyph pred == WikiCloth live pred (rows of README.mediawiki)"] = f"{sum(s_ == p_[2] for s_, p_ in zip(svg_pred, pairs))}/{len(pairs)}"
    res["SVG glyph pred == numpy ref char (all 1024 ctx, from pixels)"] = res["picture: predicted glyph exact"]
    for k, v in res.items(): print(f"  {k}: {v}")
    ok = all(v.split("/")[0] == v.split("/")[1] for v in res.values() if isinstance(v, str) and "/" in v)
    print("stage1:", "PASS" if ok else "FAIL"); return ok

INPUTS = ["the quick brown fox", "what is qwentpu?", "how do i install it", "hello, world", "", "a" * 130 + " the end", "the repository is bobliuuu allgit before depending on it"]
def check_stage2_all():
    ok = True
    for t in INPUTS:
        s, tr = gen.sanitize(t); r = check_stage2(s, tr); ok &= r["all_exact"]
    print("stage2:", "PASS" if ok else "FAIL"); return ok

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"; ok = True
    if what in ("safety", "all"): ok &= check_safety()
    if what in ("stage1", "all"): ok &= check_stage1()
    if what in ("stage2", "all"): ok &= check_stage2_all()
    import shutil; shutil.rmtree(TMP, ignore_errors=True)
    print("ALL PASS" if ok else "SOMETHING FAILED"); sys.exit(0 if ok else 1)
