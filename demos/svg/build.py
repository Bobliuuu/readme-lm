"""Graph builders: stage 1 (MediaWiki toy, 2-char ctx, 12 ReLU, 5 bit outputs) and stage 2 (q8 char MLP, CTX chars, H ReLU, 32 logits, argmax)."""
import numpy as np
from svgnn import *

VOC = "abcdefghijklmnopqrstuvwxyz .,-'?"; V = 32
HEAT_CW = 5
HEAT_GAIN_Q8 = 3          # display-only brightness gain of the hidden-activation heat strip (hidden codes are <= 104)
def heat_cw(H): return max(2, min(6, 384 // H))

# ================================================================== stage 2
def layout_q8(CTX, H):
    L = {"CW": heat_cw(H)}
    L["ctx_x"] = [16 * (k + 1) for k in range(CTX)]
    L["HX"] = 16 * (CTX + 2); L["PX"] = 16 * ((L["HX"] + H * L["CW"]) // 16 + 2); L["AX"] = L["PX"] + 56
    L["DW"] = L["AX"] + 40
    L["CB"] = 16 * (-(-L["DW"] // 16) + 1)                 # computation area starts right of the picture
    L["n1"] = CTX * V; L["HB"] = L["CB"] + L["n1"] + 8
    L["W"] = L["HB"] + max(L["n1"], H) + 8
    return L

def compute_q8(c, M, L, oh):
    """the network itself. oh: (rows, ctx*32) uint8 one-hot input pixels (code 1) = the only input data. Returns (hidden codes, argmax id, logits, preact) images."""
    W1, b1, W2, b2 = M["W1"].astype(np.int64), M["b1"].astype(np.int64), M["W2"].astype(np.int64), M["b2"].astype(np.int64)
    d1, d2, lut = int(M["d1"]), int(M["d2"]), M["lut"].astype(np.int64); CTX, H = W1.shape[0], W1.shape[2]; rows = c.rows; CB = L["CB"]
    x = c.merge([c.image(oh, x=CB, y=0)])
    E1 = W1.transpose(2, 0, 1).reshape(H, CTX * V)
    p = linear(c, x, CB, L["n1"], E1, d1, 0, 128, b1, list(range(L["HB"], L["HB"] + H)))       # one convolve per hidden neuron, 8-bit code zp 128
    hid = c.merge([c.lut(p, lut, x=L["HB"], y=0, w=H, h=rows)])                                  # ReLU + requantise (feComponentTransfer)
    lg = linear(c, hid, L["HB"], H, W2, d2, 0, 128, b2, list(range(V)))                          # one convolve per logit; codes at cols 0..31
    idn = argmax_row(c, lg, V)                                                                   # predicted id at col 31
    return hid, idn, lg, p

def build_q8(M, ctxids, actual, RH=24, debug=False, disp=True, only=None):
    W1, b1, W2, b2 = M["W1"].astype(np.int64), M["b1"].astype(np.int64), M["W2"].astype(np.int64), M["b2"].astype(np.int64)
    d1, d2, lut = int(M["d1"]), int(M["d2"]), M["lut"].astype(np.int64); CTX, H = W1.shape[0], W1.shape[2]
    N = len(ctxids); L = layout_q8(CTX, H); rows = N * RH; c = Ctx(L["W"], rows); CB = L["CB"]
    oh = np.zeros((rows, L["n1"]), np.uint8)
    for r in range(N):
        for k in range(CTX): oh[r * RH:(r + 1) * RH, k * V + ctxids[r][k]] = 1
    hid, idn, lg, p = compute_q8(c, M, L, oh)
    parts = display(c, L, N, RH, ctxids, actual, hid, idn, L["HB"], H, V - 1, cw=L["CW"], only=only, gain=HEAT_GAIN_Q8) if disp else []
    if debug: parts += {'lg': [hid, lg], 'id': [hid, idn], 'p': [p]}[debug]; c.dbg_p_col = None
    c.merge(parts or [idn])
    return c, L

def display(c, L, N, RH, ctxids, actual, hid, idn, hcol0, nh, idcol, cw=HEAT_CW, hx=None, ctx_x=None, px=None, ax=None, only=None, gain=1):
    """picture: context glyphs (baked map -> atlas), heat strip of hidden codes, predicted glyph (computed id -> map -> atlas)."""
    rows = c.rows; CTX = len(ctxids[0]); hx = hx or L["HX"]; ctx_x = ctx_x or L["ctx_x"]; px = px or L["PX"]; ax = ax or L["AX"]
    atlas = np.load(__file__.replace("build.py", "model_atlas.npz"))["atlas"]
    on = lambda k: only is None or k in only
    AT = lambda X: atlas_node(c, atlas, atlas_x(X))
    parts = []
    if on('ctx') or on('act'):
        mp = np.zeros((rows, L["DW"], 3), np.uint8); mp[:, :, 1] = 128
        for r in range(N):
            for k in range(CTX):
                X = ctx_x[k]; mp[r * RH:(r + 1) * RH, X:X + 16, 0] = gcode(ctxids[r][k], X)
            mp[r * RH:(r + 1) * RH, ax:ax + 16, 0] = gcode(actual[r], ax)
        mi = c.image(mp, x=0, y=0)
        if on('ctx'): parts.append(glyph_cells(c, AT(ctx_x[0]), mi, ctx_x[0], 16 * CTX))
        if on('act'): parts.append(glyph_cells(c, AT(ax), mi, ax, 16))
    if on('pred'):
        # predicted id (col idcol) -> 16 one-column feOffset copies (no feTile: Skia re-renders the upstream chain per tile repeat)
        # -> map pixel R = 4*id + k, G = 128 -> atlas gather
        cols = [c.offset(idn, px + i - idcol, 0, x=px + i, y=0, w=1, h=rows) for i in range(16)]
        pm0 = c.merge_at(cols, px, 0, 16, rows)
        pm = c.cmatrix(pm0, [[4, 0, 0, 0, ((atlas_x(px) + 510 - px) // 4) / 255.0], [0, 0, 0, 0, 128 / 255.0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]], x=px, y=0, w=16, h=rows)
        parts.append(glyph_cells(c, AT(px), pm, px, 16))
    if on('heat'):
        cols = [c.offset(hid, hx + j * cw + u - (hcol0 + j), 0, x=hx + j * cw + u, y=0, w=1, h=rows) for j in range(nh) for u in range(cw)]
        hm = c.merge_at(cols, hx, 0, nh * cw, rows); tr, tg, tb = heat_tables(gain); parts.append(c.lut_rgb(hm, tr, tg, tb, x=hx, y=0, w=nh * cw, h=rows))
    return parts

# ---- numpy reference for stage 2 (exact integer mirror)
def ref_q8(M, ctxids):
    import train as T
    acc1, p, h, q = T.int_forward(M, np.asarray(ctxids)); return h, q, q.argmax(1)

# ================================================================== stage 1 (MediaWiki toy)
def layout_mw():
    L = {"ctx_x": [16, 32], "HX": 64}; L["PX"] = 16 * ((64 + 12 * 16) // 16 + 2); L["AX"] = L["PX"] + 56; L["DW"] = L["AX"] + 32
    L["CB"] = 16 * (-(-L["DW"] // 16) + 1); CB = L["CB"]
    L["IDC"] = CB; L["XB"] = CB + 8; L["HB"] = CB + 40; L["ZB"] = CB + 100; L["W"] = L["ZB"] + 40
    return L

def build_mw(M, ctxids, actual, RH=24, debug=False, disp=True):
    E, W1, b1, Vv, b2 = [np.asarray(M[k], np.int64) for k in ("E", "W1", "b1", "V", "b2")]
    N = len(ctxids); L = layout_mw(); rows = N * RH; c = Ctx(L["W"], rows); XB, HB, ZB = L["XB"], L["HB"], L["ZB"]
    # ---- input data: one id pixel per context char (code floor((k+.5)*255/32): a 32-entry discrete table indexes it exactly)
    uk = np.floor((np.arange(32) + 0.5) * 255 / 32).astype(np.uint8)
    outs = []
    for k in range(2):
        col = np.repeat(uk[np.array([cx[k] for cx in ctxids])], RH)[:, None]
        idimg = c.image(col, x=L["IDC"] + k, y=0)
        tl = c.tile(idimg, XB + 6 * k, 0, 6, rows)
        for j in range(6):          # embedding gather: one 32-entry LUT per embedding channel, code = E + 128
            outs.append(c.lut(tl, E[:, j] + 128, x=XB + 6 * k + j, y=0, w=1, h=rows))
    x = c.merge(outs)
    # ---- hidden layer, pre-activation p = W1 x + b1 in [-253, 275] (9 bits): two convolves per unit
    #      A = clamp(p+128) -> ReLU LUT -> 0..127 ;  B = clamp(p-127) (0..255) ;  relu(p) = A' + B exactly
    A = linear(c, x, XB, 12, W1, 1, 128, 128, b1, list(range(HB, HB + 12)))
    hA = c.lut(A, [max(v - 128, 0) for v in range(256)], x=HB, y=0, w=12, h=rows)
    B = linear(c, x, XB, 12, W1, 1, 128, 0, b1 - 127, list(range(HB + 12, HB + 24)))
    hB = c.offset(B, 0, 0, x=HB + 12, y=0, w=12, h=rows)
    hid = c.merge([hA, hB])
    # ---- output layer: z = V h + b2 ; logit code clamp(z) (z>0 <=> code>=1) ; bit LUT ; decode id = sum bit*2^k (one more convolve)
    E2 = np.concatenate([Vv, Vv], 1)
    z = linear(c, hid, HB, 24, E2, 1, 0, 0, b2, list(range(ZB, ZB + 5)))
    bits = c.merge([c.lut(z, [0] + [1] * 255, x=ZB, y=0, w=5, h=rows)])
    idn = linear(c, bits, ZB, 5, np.array([[16, 8, 4, 2, 1]]), 1, 0, 0, np.zeros(1, np.int64), [ZB + 5])
    parts = []
    if disp:
        # heat strip: relu(p) clipped to 8 bits = A' + B (saturating add), 12 cells of 16 px
        hB_s = c.offset(hB, -12, 0, x=HB, y=0, w=12, h=rows); hs = c.merge([add_u8(c, hA, hB_s, 0, x=HB, y=0, w=12, h=rows)])
        parts = display(c, L, N, RH, ctxids, actual, hs, idn, HB, 12, ZB + 5, cw=16)
    if debug: parts += {'hid': [hid], 'z': [z], 'id': [idn], 'bits': [bits]}[debug]
    c.merge(parts or [idn])
    return c, L

def ref_mw(M, ctxids):
    E, W1, b1, Vv, b2 = [np.asarray(M[k], np.int64) for k in ("E", "W1", "b1", "V", "b2")]
    X = np.concatenate([E[np.array([cx[0] for cx in ctxids])], E[np.array([cx[1] for cx in ctxids])]], 1)
    P = X @ W1.T + b1; Hh = np.maximum(P, 0); Z = Hh @ Vv.T + b2; bits = (Z > 0).astype(np.int64)
    return Hh, Z, bits, bits @ np.array([16, 8, 4, 2, 1])
