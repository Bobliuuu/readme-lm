"""The live picture (stage 2 demo): dark card, the whole text drawn big through the glyph atlas, then a panel with one row per
window (last <= 12): context glyphs, heat grid of the 224 hidden activations, predicted glyph (computed live), actual glyph (data).
All text that depends on the input is drawn through feDisplacementMap glyph lookups; the SVG has no input-derived text nodes."""
import numpy as np
import build, svgnn
from svgnn import Ctx, png_uri, glyph_cells, atlas_node, atlas_x, gcode, GW, GH
VOC = build.VOC; V = 32
DW = 640; BG = (13, 17, 23); CARD = "#0d1117"
INK = {"main": (230, 237, 243), "ctx": (139, 148, 158), "pred": (88, 166, 255)}
CPL = 36; TX0 = 32; CTX_X0 = 32; HX = 168; CELL_W = 5; CELL_H = 6; PX = 476; AX = 544; HDR = 48; GAP = 48; MAXROWS = 12

def color_atlas(gray, tag):
    ink = np.array(INK[tag], np.float64); bg = np.array(BG, np.float64); a = gray.astype(np.float64)[:, :, None] / 255.0
    return np.round(bg + (ink - bg) * a).astype(np.uint8)

def heat_tables(gain=3):
    stops = [(0, (22, 29, 40)), (85, (31, 72, 125)), (170, (88, 166, 255)), (255, (214, 236, 255))]
    t = np.zeros((256, 3))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        for u in range(a, b + 1): t[u] = np.array(ca) + (np.array(cb) - np.array(ca)) * (u - a) / (b - a)
    t = np.round(t).astype(np.int64)[np.minimum(np.arange(256) * gain, 255)]; return t[:, 0], t[:, 1], t[:, 2]

def geometry(text, M):
    CTX, H = M["W1"].shape[0], M["W1"].shape[2]; T = len(text)
    lines = max(1, -(-T // CPL)); TR = GW * 0 + GH * lines; wy0 = TR + GAP
    P = list(range(max(0, T - (MAXROWS - 1)), T + 1)); N = len(P)
    return dict(CTX=CTX, H=H, T=T, lines=lines, TR=TR, wy0=wy0, P=P, N=N, rows=wy0 + GH * N, band=H // 4)

def build_live(M, text, debug=False):
    g = geometry(text, M); CTX, H, T, N, TR, wy0, rows, P = (g[k] for k in ("CTX", "H", "T", "N", "TR", "wy0", "rows", "P"))
    assert H % 4 == 0 and (H // 4) * CELL_W <= 300
    L = build.layout_q8(CTX, H); L["DW"] = DW; L["CB"] = DW + 16; L["HB"] = L["CB"] + L["n1"] + 8; L["W"] = L["HB"] + max(L["n1"], H) + 8
    c = Ctx(L["W"], rows); idx = {ch: i for i, ch in enumerate(VOC)}
    padded = " " * CTX + text
    ctxids = [[idx[ch] for ch in padded[p:p + CTX]] for p in P]; act = [idx[text[p]] if p < T else idx[" "] for p in P]
    oh = np.zeros((rows, L["n1"]), np.uint8)
    for r in range(N):
        for k in range(CTX): oh[wy0 + r * GH:wy0 + (r + 1) * GH, k * V + ctxids[r][k]] = 1
    hid, idn, lg, pre = build.compute_q8(c, M, L, oh)
    # ---- baked glyph map (data): R = gather code for the glyph id, G = 128
    mp = np.zeros((rows, DW, 3), np.uint8)
    def cell(y, X, ident): mp[y:y + GH, X:X + GW, 0] = gcode(ident, X); mp[y:y + GH, X:X + GW, 1] = 128
    for m, ch in enumerate(text): cell(GH * (m // CPL), TX0 + GW * (m % CPL), idx[ch])
    for r in range(N):
        y = wy0 + r * GH
        for k in range(CTX): cell(y, CTX_X0 + GW * k, ctxids[r][k])
        cell(y, AX, act[r])
    mi = c.image(mp, x=0, y=0); gray = np.load(__file__.replace("view.py", "model_atlas.npz"))["atlas"]
    AT = lambda X, tag: atlas_node(c, color_atlas(gray, tag), atlas_x(X), tag)
    parts = []
    nfit = sum(1 for m in range(CPL) if TX0 + GW * m <= 500)
    parts.append(glyph_cells(c, AT(TX0, "main"), mi, TX0, GW * nfit, 0, TR))
    for m in range(nfit, CPL): X = TX0 + GW * m; parts.append(glyph_cells(c, AT(X, "main"), mi, X, GW, 0, TR))
    parts.append(glyph_cells(c, AT(CTX_X0, "ctx"), mi, CTX_X0, GW * CTX, wy0, GH * N))
    parts.append(glyph_cells(c, AT(AX, "main"), mi, AX, GW, wy0, GH * N))
    # ---- predicted glyph: id pixel (col 31) -> 16 one-column copies -> map pixel -> atlas gather
    cols = [c.offset(idn, PX + i - (V - 1), 0, x=PX + i, y=wy0, w=1, h=GH * N) for i in range(GW)]
    pm0 = c.merge_at(cols, PX, wy0, GW, GH * N)
    pm = c.cmatrix(pm0, [[4, 0, 0, 0, ((atlas_x(PX) + 510 - PX) // 4) / 255.0], [0, 0, 0, 0, 128 / 255.0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]], x=PX, y=wy0, w=GW, h=GH * N)
    parts.append(glyph_cells(c, AT(PX, "pred"), pm, PX, GW, wy0, GH * N))
    # ---- heat grid: 224 units folded into 4 bands of 56 cells (5 x 6 px each), gradient palette, band mask by feComposite 'in'
    bw = g["band"]; tr, tg, tb = heat_tables()
    for b in range(4):
        cols = [c.offset(hid, HX + j * CELL_W + u - (L["HB"] + b * bw + j), 0, x=HX + j * CELL_W + u, y=wy0, w=1, h=GH * N) for j in range(bw) for u in range(CELL_W)]
        hm = c.merge_at(cols, HX, wy0, bw * CELL_W, GH * N); hl = c.lut_rgb(hm, tr, tg, tb, x=HX, y=wy0, w=bw * CELL_W, h=GH * N)
        mask = np.zeros((GH * N, bw * CELL_W, 4), np.uint8); 
        for r in range(N): mask[r * GH + b * CELL_H:r * GH + (b + 1) * CELL_H, :, 3] = 255
        mk = c.image(mask, x=HX, y=wy0)
        parts.append(c.prim(f'<feComposite in="{hl}" in2="{mk}" operator="in" x="{HX}" y="{wy0}" width="{bw*CELL_W}" height="{GH*N}" RESULT/>'))
    if debug: parts += {'lg': [hid, lg], 'id': [hid, idn], 'p': [pre]}[debug]; c.merge(parts)
    else: c.merge_at(parts, 8, 0, DW - 16, rows, color=CARD)
    info = dict(g, L=L, ctxids=ctxids, act=act, c=c, idx=idx, hdr=HDR)
    return c, info

def footer_lines(trunc, maxlen):
    f = ["Computed live by your browser's SVG filters: convolve, ReLU table, argmax, glyph lookup.",
         "Blue = network prediction, white = actual next character. Chrome, 100% zoom, non-Retina."]
    if trunc: f.append(f"(input truncated to the first {maxlen} characters)")
    return f
