"""Model plan (parameters derived from the Q8 spec/params) + integer numpy mirror of the SVG pipeline + SVG assembly."""
import json, sys, math
import numpy as np
from ops_lib import *
sys.path.insert(0, "/Users/jhome/readme-lm/trackD/qat8")
import q8_ref as Q
PARAMS = "/Users/jhome/readme-lm/trackD/qat8/q8_params.npz"

def wreal(Z, name):
    cc = Z[name + "_c"].astype(np.float64); s = Z[name + "_s"].astype(np.float64)
    return (cc.reshape(cc.shape[0], -1, 32) * s[..., None]).reshape(cc.shape)

class Plan:
    def __init__(s, T, params=PARAMS, wmax=127):
        s.T = T; s.m = Q.Q8Ref(params, maxT=max(T, 64)); s.Z = np.load(params, allow_pickle=False); s.sp = s.m.sp; s.wmax = wmax
        s.norms = {}; s.lins = {}; s.att = {}; s.rope = {}; s.res = {}
        sp = s.sp; fs = lambda k: (sp[k + ".s"], sp[k + ".z"])
        s.fs = fs
        s.emb = s.m.emb_tab.astype(np.int64)
    # ---- parameter makers ----
    def lin(s, key, Wr, sin, sout, zin, zout):
        w = Wr * sin / sout; E, d = quantize_weights(w, s.wmax)
        s.lins[key] = dict(E=E, d=d, zin=int(zin), zout=int(zout)); return s.lins[key]
    def norm_base(s, n, inn, gain):
        sp = s.sp; sx, zx = s.fs(inn); ssq = sp[n + ".sq.s"]; sy, zy = s.fs(n); sr, zr = sp[n + ".sr"], sp[n + ".zr"]; C = int(sp[n + ".C"]); mm = sp[n + ".mm"]
        e = int(round(mm * 1000)); d = 1000
        return dict(sq=Q.lut_sq(sx, zx, ssq), rsq=Q.lut_rsq(C, e / d, ssq, sr, zr), C=C, e=e, d=d, zx=int(zx), zr=int(zr), zy=int(zy), S2=255,
                    gain=np.asarray(gain, np.float64), sx=sx, sr=sr, sy=sy)
    def norm_calibrate(s, key, base, Xlist):
        """choose S1 (<=600) so that t=(x-zx)(r8-zr)/S1 fits +-127 for the calibration data; gq from gain."""
        mx = 1
        for X in Xlist:
            ssum = np.asarray(base["sq"], np.int64)[X].sum(1); ms = np.clip(nudge_round(ssum * base["e"] - base["C"] * base["e"], base["d"]), 0, 255)
            r8 = np.asarray(base["rsq"], np.int64)[ms]
            mx = max(mx, int(np.abs((X - base["zx"]) * (r8[:, None] - base["zr"])).max()))
        S1 = int(min(600, max(8, math.ceil(mx / 127.0))))
        g = np.clip(np.round(base["S2"] * S1 * base["gain"] * base["sx"] * base["sr"] / base["sy"]), 0, 255).astype(np.int64)
        P = dict(base); P["S1"] = S1; P["gq"] = g; P["fold"] = True; P["sy2"] = S1 * base["sx"] * base["sr"]; P["clipped"] = int((np.round(base["S2"] * S1 * base["gain"] * base["sx"] * base["sr"] / base["sy"]) > 255).sum()); P["mxprod"] = mx
        s.norms[key] = P; return P
    def att_params(s, l):
        p = f"L{l}."; sp = s.sp; g = lambda k: sp[p + k]
        Sp = int(round(g("pr.s") / (g("qr.s") * g("kr.s")))); spr = Sp * g("qr.s") * g("kr.s")           # product scale actually realised (integer divisor)
        sf, zf = g("f.s"), int(g("f.z")); ssc, zsc = g("sc.s"), int(g("sc.z"))
        msc = spr * Q.ISD / ssc; e_sc, d_sc, er = best_ed(msc)
        Maf, Mbf = ssc / sf, spr * Q.ISD / sf                                      # fine stage: dd = (m-zsc)*Maf - sum(pr-128)*Mbf + ZF
        best = None
        for d in range(1, 4000):
            e1, e2 = int(round(Mbf * d)), int(round(Maf * d))
            if e1 < 1 or e1 + e2 > 255: continue
            err = max(abs(e1 / d - Mbf) / Mbf, abs(e2 / d - Maf) / Maf)
            if best is None or err < best[0]: best = (err, e1, e2, d)
        _, e1, e2, d_f = best
        D = int(g("D")); rec = np.clip(np.floor(255.0 / (np.maximum(np.arange(256), 1) * D) * 255 + 0.5), 0, 255).astype(np.int64)
        S_pv = int(round(255 * g("pv.s") / g("v.s"))); s_pv = S_pv * g("v.s") / 255; e_o, d_o, er2 = best_ed(1.0)
        P = dict(zq=int(g("qr.z")), zk=int(g("kr.z")), Sp=Sp, zsc=zsc, e_sc=e_sc, d_sc=d_sc, e1=e1, e2=e2, d_f=d_f, ZF=zf, exp=Q.lut_exp(sf), e_S=1, d_S=D, rec=rec,
                 zv=int(g("v.z")), S_pv=S_pv, e_o=e_o, d_o=d_o, zao=int(g("ao.z")), s_ao=s_pv, fine_err=best[0], coarse_err=er)
        s.att[l] = P; return P
    def rope_tabs(s, l, nm, T):
        sp = s.sp; ratio = sp[f"L{l}." + {"qr": "q", "kr": "k"}[nm] + ".s"] / sp[f"L{l}.{nm}.s"]
        R = int(math.floor(120 / max(ratio, 1e-9))); R = min(R, 120)
        f = 1.0 / (10000 ** (np.arange(0, 32, 2) / 32)); ang = np.arange(T)[:, None] * f[None]
        cosc = np.tile(np.round(R * np.cos(ang) * ratio), (1, 4)).astype(np.int64); sinc = np.tile(np.round(R * np.sin(ang) * ratio), (1, 4)).astype(np.int64)
        return dict(cosc=cosc, sinc=sinc, R=R, z=int(sp[f"L{l}." + {"qr": "q", "kr": "k"}[nm] + ".z"]), zo=int(sp[f"L{l}.{nm}.z"]))

def residual_scales(sa, sb, s1, D=256):
    return int(round(D * sa / s1)), int(round(D * sb / s1))

def build(T, calib_ids, wmax=127):
    """derive all parameters; calibration runs the integer mirror on calib_ids (list of id lists, each length T)."""
    pl = Plan(T, wmax=wmax); m = pl.m; sp = pl.sp; fs = pl.fs; Z = pl.Z
    states = [pl.emb[np.array(ids)] for ids in calib_ids]; inn = "emb"
    for l in range(3):
        p = f"L{l}."
        # n1
        base = pl.norm_base(p + "n1", inn, Z[f"g1_{l}"].astype(np.float64)); P1 = pl.norm_calibrate(p + "n1", base, states)
        n1 = [rmsnorm_ref(X, P1) for X in states]
        sn1, zn1 = P1["sy2"], 128
        for nm, a in (("q", 0), ("k", 128), ("v", 256)):
            Wr = wreal(Z, f"qkv{l}")[a:a + 128] * P1["gain"][None, :]; so, zo_ = fs(p + nm)
            L = pl.lin(p + nm, Wr, sn1, so, zn1, zo_)
        for nm in "qk": pl.rope[p + nm] = pl.rope_tabs(l, nm + "r", T)
        pl.att_params(l)
        # o
        sao, zao = pl.att[l]["s_ao"], pl.att[l]["zao"]; so_, zo_ = fs(p + "o"); pl.lin(p + "o", wreal(Z, f"o{l}"), sao, so_, zao, zo_)
        sx, zx = fs(inn); s1, z1 = fs(p + "r1"); pa, pb = residual_scales(sx, so_, s1)
        pl.res[p + "r1"] = dict(pa=pa, pb=pb, za=int(zx), zb=int(zo_), zo=int(z1), D=256)
        # forward calib through attention/o/res
        new = []
        for X, N1 in zip(states, n1):
            qx = linear_ref(N1, [128], *[pl.lins[p + "q"][k] for k in ("E", "d")], pl.lins[p + "q"]["zin"], pl.lins[p + "q"]["zout"], np.zeros(128, np.int64))
            kx = linear_ref(N1, [128], *[pl.lins[p + "k"][k] for k in ("E", "d")], pl.lins[p + "k"]["zin"], pl.lins[p + "k"]["zout"], np.zeros(128, np.int64))
            vx = linear_ref(N1, [128], *[pl.lins[p + "v"][k] for k in ("E", "d")], pl.lins[p + "v"]["zin"], pl.lins[p + "v"]["zout"], np.zeros(128, np.int64))
            R_ = pl.rope[p + "q"]; qr = rope_ref(qx, R_["cosc"], R_["sinc"], R_["z"], R_["zo"], R_["R"])
            R_ = pl.rope[p + "k"]; kr = rope_ref(kx, R_["cosc"], R_["sinc"], R_["z"], R_["zo"], R_["R"])
            ao = attention_ref(qr, kr, vx, pl.att[l], T)
            o = linear_ref(ao, [128], pl.lins[p + "o"]["E"], pl.lins[p + "o"]["d"], pl.lins[p + "o"]["zin"], pl.lins[p + "o"]["zout"], np.zeros(128, np.int64))
            r = pl.res[p + "r1"]; r1 = resadd_ref(X, o, r["pa"], r["pb"], r["za"], r["zb"], r["zo"], r["D"]); new.append(r1)
        states = new; inn1 = p + "r1"
        base = pl.norm_base(p + "n2", inn1, Z[f"g2_{l}"].astype(np.float64)); P2 = pl.norm_calibrate(p + "n2", base, states)
        n2 = [rmsnorm_ref(X, P2) for X in states]; sn2, zn2 = P2["sy2"], 128
        su, zu = fs(p + "u"); pl.lin(p + "u", wreal(Z, f"u{l}") * P2["gain"][None, :], sn2, su, zn2, zu)
        sg, zg = fs(p + "g"); pl.lins[p + "gelu"] = Q.lut_gelu(su, zu, sg, zg)
        sdn, zdn = fs(p + "dn"); pl.lin(p + "dn", wreal(Z, f"dn{l}"), sg, sdn, zg, zdn)
        s2, z2 = fs(p + "r2"); pa, pb = residual_scales(s1, sdn, s2); pl.res[p + "r2"] = dict(pa=pa, pb=pb, za=int(z1), zb=int(zdn), zo=int(z2), D=256)
        new = []
        for X, N2 in zip(states, n2):
            u = linear_ref(N2, [128], pl.lins[p + "u"]["E"], pl.lins[p + "u"]["d"], zn2, zu, np.zeros(512, np.int64)); g = np.asarray(pl.lins[p + "gelu"], np.int64)[u]
            dn = linear_ref(g, split_chunks(512), pl.lins[p + "dn"]["E"], pl.lins[p + "dn"]["d"], zg, zdn, np.zeros(128, np.int64))
            r = pl.res[p + "r2"]; new.append(resadd_ref(X, dn, r["pa"], r["pb"], r["za"], r["zb"], r["zo"], r["D"]))
        states = new; inn = p + "r2"
    base = pl.norm_base("nf", inn, Z["gf"].astype(np.float64)); Pn = pl.norm_calibrate("nf", base, states)
    snf, znf = Pn["sy2"], 128; slg, zlg = fs("lg"); pl.lin("head", wreal(Z, "emb") * Pn["gain"][None, :], snf, slg, znf, zlg)
    return pl

def forward_ref(pl, ids):
    """integer mirror of the SVG pipeline; returns dict of stage tensors (T rows) incl. 'logits' (T,86), 'argmax' (T,)"""
    T = pl.T; sp = pl.sp; fs = pl.fs; tr = {}; zero = lambda n: np.zeros(n, np.int64)
    X = pl.emb[np.array(ids)]; tr["emb"] = X
    for l in range(3):
        p = f"L{l}."; N1 = rmsnorm_ref(X, pl.norms[p + "n1"]); tr[p + "n1"] = N1
        def lin1(k, A, chunks=[128]): L = pl.lins[k]; return linear_ref(A, chunks, L["E"], L["d"], L["zin"], L["zout"], zero(L["E"].shape[0]))
        qx, kx, vx = lin1(p + "q", N1), lin1(p + "k", N1), lin1(p + "v", N1); tr[p + "q"], tr[p + "k"], tr[p + "v"] = qx, kx, vx
        R_ = pl.rope[p + "q"]; qr = rope_ref(qx, R_["cosc"], R_["sinc"], R_["z"], R_["zo"], R_["R"])
        R_ = pl.rope[p + "k"]; kr = rope_ref(kx, R_["cosc"], R_["sinc"], R_["z"], R_["zo"], R_["R"]); tr[p + "qr"], tr[p + "kr"] = qr, kr
        ao = attention_ref(qr, kr, vx, pl.att[l], T); tr[p + "ao"] = ao
        o = lin1(p + "o", ao); tr[p + "o"] = o; r = pl.res[p + "r1"]; R1 = resadd_ref(X, o, r["pa"], r["pb"], r["za"], r["zb"], r["zo"], r["D"]); tr[p + "r1"] = R1
        N2 = rmsnorm_ref(R1, pl.norms[p + "n2"]); tr[p + "n2"] = N2; u = lin1(p + "u", N2); g = np.asarray(pl.lins[p + "gelu"], np.int64)[u]; tr[p + "g"] = g
        L = pl.lins[p + "dn"]; dn = linear_ref(g, split_chunks(512), L["E"], L["d"], L["zin"], L["zout"], zero(128)); tr[p + "dn"] = dn
        r = pl.res[p + "r2"]; X = resadd_ref(R1, dn, r["pa"], r["pb"], r["za"], r["zb"], r["zo"], r["D"]); tr[p + "r2"] = X
    NF = rmsnorm_ref(X, pl.norms["nf"]); tr["nf"] = NF; L = pl.lins["head"]; lg = linear_ref(NF, [128], L["E"], L["d"], L["zin"], L["zout"], zero(86))
    tr["logits"] = lg; tr["argmax"] = lg.argmax(1); return tr

# ======================================================================= SVG assembly ==========================
GW, GH = 24, 36
ATT_UPTO = None
def build_svg(pl, ids, stop=None, gw=GW, gh=GH):
    """emit the whole token step as ONE filter graph; `stop` (stage name) ends the graph at that stage for verification.
    returns (svg, view=(x,y,w,h) cropping the stage tensor, ctx)"""
    T = pl.T
    ox = int(gw * 127.5 - gw * 85); W = max(660, ox + gw + 8, gw * (len(Q.VOC) + 1) + 8); H = attn_H(T)
    c = Ctx(W, H); c.rows = T; zero = lambda n: np.zeros(n, np.int64)
    def done(name, img, x=0, w=128, y=0, h=None):
        if stop == name:
            return ("view", img, (x, y, w, h or T))
    # embedding
    x = embed_lookup(c, ids, pl.emb, 0); r = done("emb", x)
    if r: return finish_view(c, r)
    for l in range(3):
        p = f"L{l}."
        n1 = rmsnorm(c, x, 0, 128, pl.norms[p + "n1"])
        r = done(p + "n1", n1)
        if r: return finish_view(c, r)
        def lin1(k, src, chunks=[(0, 128)], n_out=None):
            L = pl.lins[k]; n = L["E"].shape[0] if n_out is None else n_out
            return linear(c, src, chunks, L["E"], L["d"], L["zin"], L["zout"], zero(n), list(range(n)))
        def linA(k, src):   # output at head-strided columns
            L = pl.lins[k]; return linear(c, src, [(0, 128)], L["E"], L["d"], L["zin"], L["zout"], zero(128), ACOLS)
        q = linA(p + "q", n1)
        if stop == p + "q": return finish_view(c, ("view", q, (0, 0, WA, T)))
        k = linA(p + "k", n1); v = linA(p + "v", n1)
        R_ = pl.rope[p + "q"]; qr = rope_apply(c, q, ACOLS, R_["cosc"], R_["sinc"], R_["z"], R_["zo"], R_["R"], wspan=WA)
        if stop == p + "qr": return finish_view(c, ("view", qr, (0, 0, WA, T)))
        R_ = pl.rope[p + "k"]; kr = rope_apply(c, k, ACOLS, R_["cosc"], R_["sinc"], R_["z"], R_["zo"], R_["R"], wspan=WA)
        ao = attention(c, qr, kr, v, pl.att[l], T, upto=ATT_UPTO)
        if stop == p + "ao": return finish_view(c, ("view", ao, (0, 0, WA, T)))
        L = pl.lins[p + "o"]; Epad = np.zeros((128, WA), np.int64); Epad[:, ACOLS] = L["E"]
        o = linear(c, ao, [(0, WA)], Epad, L["d"], L["zin"], L["zout"], zero(128), list(range(128))); r = done(p + "o", o)
        if r: return finish_view(c, r)
        rr = pl.res[p + "r1"]; r1 = finish(c, resadd(c, x, o, rr["pa"], rr["pb"], rr["za"], rr["zb"], rr["zo"], rr["D"], x=0, y=0, w=128, h=T))
        r = done(p + "r1", r1)
        if r: return finish_view(c, r)
        n2 = rmsnorm(c, r1, 0, 128, pl.norms[p + "n2"]); r = done(p + "n2", n2)
        if r: return finish_view(c, r)
        sizes = split_chunks(512); starts = [0]; 
        for sz in sizes[:-1]: starts.append(starts[-1] + sz + 1)
        ucols = [j + j // sizes[0] for j in range(512)]; assert all(ucols[j] in range(starts[min(j // sizes[0], 2)], starts[min(j // sizes[0], 2)] + sizes[min(j // sizes[0], 2)]) for j in range(512))
        L = pl.lins[p + "u"]
        u = linear(c, n2, [(0, 128)], L["E"], L["d"], L["zin"], L["zout"], zero(512), ucols)
        gl = [c.lut(u, pl.lins[p + "gelu"], x=st, y=0, w=sz, h=T) for st, sz in zip(starts, sizes)]
        g = c.merge(gl)
        if stop == p + "g": return finish_view(c, ("view", g, (0, 0, 515, T)))
        L = pl.lins[p + "dn"]; chunks = [(st, sz) for st, sz in zip(starts, sizes)]
        dn = linear(c, g, chunks, L["E"], L["d"], L["zin"], L["zout"], zero(128), list(range(128)))
        r = done(p + "dn", dn)
        if r: return finish_view(c, r)
        rr = pl.res[p + "r2"]; x = finish(c, resadd(c, r1, dn, rr["pa"], rr["pb"], rr["za"], rr["zb"], rr["zo"], rr["D"], x=0, y=0, w=128, h=T))
        r = done(p + "r2", x)
        if r: return finish_view(c, r)
    nf = rmsnorm(c, x, 0, 128, pl.norms["nf"]); r = done("nf", nf)
    if r: return finish_view(c, r)
    L = pl.lins["head"]; lg = linear(c, nf, [(0, 128)], L["E"], L["d"], L["zin"], L["zout"], zero(86), list(range(86)))
    if stop == "logits": return finish_view(c, ("view", lg, (0, 0, 86, T)))
    X = argmax_row(c, lg, 86)
    if stop == "argmax": return finish_view(c, ("view", X, (127, 0, 1, T)))
    atlas = make_atlas(list(Q.VOC) + ["EOS"], gw, gh, fsize=int(gh * 0.72))
    gl, oxx = glyph_stage(c, X, T - 1, atlas, gw, gh, 0)
    return finish_view(c, ("view", gl, (oxx, 0, gw, gh)))

def finish_view(c, r):
    _, img, (x, y, w, h) = r
    c.prim(f'<feComponentTransfer in="{img}"/>')      # identity: makes the requested stage the filter output
    svg = c.g.svg()
    svg = svg.replace(f'width="{c.W}" height="{c.H}" viewBox="0 0 {c.W} {c.H}"', f'width="{w}" height="{h}" viewBox="{x} {y} {w} {h}"', 1)
    return svg, (x, y, w, h), c
