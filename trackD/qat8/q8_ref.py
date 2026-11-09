"""Pure-numpy INTEGER reference of the 8-bit-activation README-LM QA transformer (see q8_spec.md).
Every tensor is a uint8 code (stored in int64 arrays); all arithmetic is exact integer; rounding = round-half-up via
(acc*M + 2^31) >> 32 with M a 32-bit fixed-point multiplier. Usage:
    m = Q8Ref("q8_params.npz"); logits_u8 = m.forward_token(ids); ans = m.generate("what is qwentpu")"""
import json, re, numpy as np
VOC = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"()-_/=+*#<>[]!?\n"
EOS = len(VOC); V = len(VOC) + 1
D_MODEL, N_LAYER, N_HEAD, D_FF, GRP = 128, 3, 4, 512, 32
HD = D_MODEL // N_HEAD; SH = 32; SR_INV = 256; KB = 256; EPS = 1e-5
ISD = 1.0 / np.sqrt(HD)
import math
_erf = np.vectorize(math.erf)
U = np.arange(256, dtype=np.float64)
def rnd(x): return np.floor(np.asarray(x, np.float64) + 0.5)
def fxm(r): return np.floor(np.asarray(r, np.float64) * 2.0 ** SH + 0.5).astype(np.int64)
def clip8(x): return np.clip(x, 0, 255)
def rs(a): return (a + (1 << (SH - 1))) >> SH            # round-half-up right shift of an int64 accumulator
def gelu64(x): x = np.asarray(x, np.float64); return 0.5 * x * (1 + _erf(x / math.sqrt(2)))

# ---------------- LUT generators (float64, deterministic) ----------------
def lut_sq(sx, zx, s_sq): return clip8(rnd((U - zx) ** 2 * sx * sx / s_sq)).astype(np.int64)
def ms_real(ms8, C, mm, s_sq): return (np.asarray(ms8, np.float64) / mm + C) * s_sq / D_MODEL
def lut_rsq(C, mm, s_sq, sr, zr): return clip8(rnd(1.0 / np.sqrt(ms_real(U, C, mm, s_sq) + EPS) / sr) + zr).astype(np.int64)
def lut_exp(s_sc): return clip8(rnd(255.0 * np.exp(-U * s_sc))).astype(np.int64)
def lut_rec(Dv): return clip8(rnd(255.0 / (np.maximum(U, 1) * Dv) * SR_INV)).astype(np.int64)
def lut_gelu(su, zu, sg, zg): return clip8(rnd(gelu64((U - zu) * su) / sg) + zg).astype(np.int64)
def rope_tables(T):
    f = 1.0 / (10000 ** (np.arange(0, HD, 2) / HD)); a = np.arange(T, dtype=np.float64)[:, None] * f[None]; return np.cos(a), np.sin(a)

class Q8Ref:
    def __init__(s, path, maxT=512, noise=0.0, seed=0):
        s.noise = noise; s.rng = np.random.default_rng(seed)
        Z = np.load(path, allow_pickle=False); s.sp = json.loads(str(Z["spec"])); sp = s.sp; s.maxT = maxT
        def W(n):  # int codes (O,I) -> (O,G,32) float64 ; group scales float64 (O,G)
            c = Z[n + "_c"].astype(np.float64); o, i = c.shape; return c.reshape(o, i // GRP, GRP), Z[n + "_s"].astype(np.float64)
        def fs(n): return sp[n + ".s"], sp[n + ".z"]
        s.fs = fs; s.layers = []
        s.cos, s.sin = rope_tables(maxT)
        Wemb, Semb = W("emb")
        s0, z0 = fs("emb"); s.emb_tab = np.clip(rnd((Wemb * Semb[..., None]).reshape(V, -1) / s0) + 128, 0, 255).astype(np.int64)
        assert z0 == 128
        def lin_M(sw, sin, sout): return fxm(sw * sin / sout)           # (O,G) (sout scalar or (O,1))
        def norm_c(n, inn, g):
            sx, zx = fs(inn); ssq = sp[n + ".sq.s"]; sy, zy = fs(n); sr, zr = sp[n + ".sr"], sp[n + ".zr"]
            return dict(sq=lut_sq(sx, zx, ssq), rsq=lut_rsq(sp[n + ".C"], sp[n + ".mm"], ssq, sr, zr), C=int(sp[n + ".C"]), Mmm=int(fxm(sp[n + ".mm"])),
                        zr=int(zr), G=fxm(g.astype(np.float64) * sx * sr / sy), zx=zx, zy=zy)
        inn = "emb"; cs, sn = rope_tables(maxT)
        for l in range(N_LAYER):
            p = f"L{l}."; L = {}
            L["n1"] = norm_c(p + "n1", inn, Z[f"g1_{l}"].astype(np.float32)); sn1, zn1 = fs(p + "n1")
            Wq, Sq = W(f"qkv{l}"); sout = np.concatenate([np.full(128, fs(p + n)[0]) for n in "qkv"])[:, None]
            L["qkv"] = (Wq, lin_M(Sq, sn1, sout)); L["qz"] = [fs(p + n)[1] for n in "qkv"]
            for n, a in (("qr", "q"), ("kr", "k")):
                ratio = fs(p + a)[0] / fs(p + n)[0]; L[n + "C"] = fxm(s.cos * ratio); L[n + "S"] = fxm(s.sin * ratio); L[n + "zi"] = fs(p + a)[1]
            sqr, zqr = fs(p + "qr"); skr, zkr = fs(p + "kr"); spr = sp[p + "pr.s"]; ssc, zsc = fs(p + "sc"); sv, zv = fs(p + "v"); spv = sp[p + "pv.s"]; sao, zao = fs(p + "ao")
            L.update(zqr=zqr, zkr=zkr, Mp=int(fxm(sqr * skr / spr)), Msc=int(fxm(spr * ISD / ssc)), zsc=zsc, exp=lut_exp(sp[p + "f.s"]), Maf=int(fxm(ssc / sp[p + "f.s"])), Mbf=int(fxm(spr * ISD / sp[p + "f.s"])), ZF=int(sp[p + "f.z"]), D=int(sp[p + "D"]), MD=int(fxm(1.0 / sp[p + "D"])),
                     rec=lut_rec(sp[p + "D"]), MP=int(fxm(1.0 / SR_INV)), zv=zv, Mpv=int(fxm(sv / (255 * spv))), Mpb=int(fxm(spv / sao)))
            Wo, So = W(f"o{l}"); so_, zo = fs(p + "o"); L["o"] = (Wo, lin_M(So, sao, so_)); L["zao"] = zao; L["zo"] = zo
            sx, zx = fs(inn); s1, z1 = fs(p + "r1"); L["r1"] = (int(fxm(sx / s1)), int(fxm(so_ / s1)), z1, zx)
            L["n2"] = norm_c(p + "n2", p + "r1", Z[f"g2_{l}"].astype(np.float32)); sn2, zn2 = fs(p + "n2")
            Wu, Su = W(f"u{l}"); su, zu = fs(p + "u"); L["u"] = (Wu, lin_M(Su, sn2, su)); L["zn2"] = zn2
            sg, zg = fs(p + "g"); L["gelu"] = lut_gelu(su, zu, sg, zg); L["zu"] = zu; L["zg"] = zg
            Wd, Sd = W(f"dn{l}"); sdp, zdp = fs(p + "dp"); L["dn"] = (Wd, lin_M(Sd, sg, sdp)); assert zdp == 128
            sdn, zdn = fs(p + "dn"); L["Mdp"] = int(fxm(sdp / sdn)); L["zdn"] = zdn
            s2, z2 = fs(p + "r2"); L["r2"] = (int(fxm(s1 / s2)), int(fxm(sdn / s2)), z2, z1, zdn); inn = p + "r2"
            s.layers.append(L)
        s.nf = norm_c("nf", inn, Z["gf"].astype(np.float32)); snf, znf = fs("nf")
        s.head = (Wemb, lin_M(Semb, snf, fs("lg")[0])); s.znf = znf; s.zlg = fs("lg")[1]
        s.reset()
    def reset(s):
        s.K = np.zeros((N_LAYER, N_HEAD, s.maxT, HD), np.int64); s.Vc = np.zeros_like(s.K); s.n = 0
    # ----- primitives -----
    def nz(s, a):   # optional emulation of renderer convolve noise: uniform +-noise levels added before rounding
        return a if not s.noise else a + np.floor(s.rng.uniform(-s.noise, s.noise, np.shape(a)) * 2.0 ** SH).astype(np.int64)
    def lin(s, u, zin, Wm, groups=None):   # u (I,) codes -> int64 acc (O,) = sum_g M[o,g]*sum_j c*(u-zin)
        W, M = Wm; d = (u - zin).astype(np.float64).reshape(-1, GRP)
        if groups is not None: W, M = W[:, groups[0]:groups[1]], M[:, groups[0]:groups[1]]; d = d[groups[0]:groups[1]]
        S = np.einsum("ogk,gk->og", W, d).astype(np.int64); return s.nz((S * M).sum(1))
    @staticmethod
    def norm(x, c):
        S = int(c["sq"][x].sum()); ms8 = min(max(int(rs((S - c["C"]) * c["Mmm"])), 0), 255); r8 = int(c["rsq"][ms8])
        return clip8(rs((x - c["zx"]) * (r8 - c["zr"]) * c["G"]) + c["zy"])
    def step(s, tok, pos):
        H, D = N_HEAD, HD; x = s.emb_tab[tok].copy(); tr = s.trace = {}; tr['emb'] = x
        for l, L in enumerate(s.layers):
            n1 = s.norm(x, L["n1"]); tr[f'L{l}.n1'] = n1; sn1 = s.fs(f"L{l}.n1")[1]
            qkv = clip8(rs(s.lin(n1, sn1, L["qkv"])) + np.repeat(L["qz"], 128)).reshape(3, H, D)
            def rope(a, nm):
                C, S_, z = L[nm + "C"][pos], L[nm + "S"][pos], L[nm + "zi"]; a1, a2 = a[:, 0::2] - z, a[:, 1::2] - z; o = np.empty_like(a)
                o[:, 0::2] = a1 * C - a2 * S_; o[:, 1::2] = a1 * S_ + a2 * C
                return clip8(rs(o) + s.fs(f"L{l}." + nm)[1])
            tr[f'L{l}.q'] = qkv[0]; tr[f'L{l}.v'] = qkv[2]; qr = rope(qkv[0], "qr"); kr = rope(qkv[1], "kr"); tr[f'L{l}.qr'] = qr; s.K[l, :, pos] = kr; s.Vc[l, :, pos] = qkv[2]; t = pos + 1
            pr = clip8(rs((qr - L["zqr"])[:, None, :] * (s.K[l, :, :t] - L["zkr"]) * L["Mp"]) + 128)         # (H,t,D) 8-bit products
            sc = clip8(rs(s.nz((pr - 128).sum(-1) * L["Msc"])) + L["zsc"])                                              # (H,t) scores
            tr[f'L{l}.sc'] = sc; dd = clip8(rs(s.nz((sc.max(1, keepdims=True) - L["zsc"]) * L["Maf"] - (pr - 128).sum(-1) * L["Mbf"])) + L["ZF"]); e = L["exp"][dd]; Sg = e.sum(1)
            S8 = clip8(rs(Sg * L["MD"])); R8 = L["rec"][S8]; p8 = clip8(rs(e * R8[:, None] * L["MP"]))
            pv = np.clip(rs(p8[:, :, None] * (s.Vc[l, :, :t] - L["zv"]) * L["Mpv"]), -128, 127)                    # (H,t,D) 8-bit products
            ao = np.zeros((H, D), np.int64)
            for k0 in range(0, t, KB): ao += np.clip(rs(pv[:, k0:k0 + KB].sum(1) * L["Mpb"]), -128, 127)
            ao = clip8(ao + L["zao"]).reshape(-1); tr[f'L{l}.p8'] = p8; tr[f'L{l}.ao'] = ao
            o = clip8(rs(s.lin(ao, L["zao"], L["o"])) + L["zo"])
            tr[f'L{l}.o'] = o; Ma, Mb, z1, zx = L["r1"]; r1 = clip8(rs((x - zx) * Ma + (o - L["zo"]) * Mb) + z1)
            tr[f'L{l}.r1'] = r1; n2 = s.norm(r1, L["n2"]); tr[f'L{l}.n2'] = n2; u = clip8(rs(s.lin(n2, L["zn2"], L["u"])) + s.fs(f"L{l}.u")[1]); g = L["gelu"][u]
            p0 = clip8(rs(s.lin(g, L["zg"], L["dn"], (0, 8))) + 128); p1 = clip8(rs(s.lin(g, L["zg"], L["dn"], (8, 16))) + 128)
            tr[f'L{l}.g'] = g; dn = clip8(rs(((p0 - 128) + (p1 - 128)) * L["Mdp"]) + L["zdn"])
            Ma, Mb, z2, z1_, zdn = L["r2"]; x = clip8(rs((r1 - z1_) * Ma + (dn - zdn) * Mb) + z2); tr[f'L{l}.dn'] = dn; tr[f'L{l}.r2'] = x
        nf = s.norm(x, s.nf); return clip8(rs(s.lin(nf, s.znf, s.head)) + s.zlg)

    def export_tables(s, path):
        """All constants an SVG generator needs (LUTs, integer multipliers, weight codes) -> npz."""
        d = {}; S = {}
        d["emb_tab"] = s.emb_tab.astype(np.uint8)
        def nrm(n, c):
            d[n + ".sq_lut"] = c["sq"].astype(np.uint8); d[n + ".rsq_lut"] = c["rsq"].astype(np.uint8); d[n + ".G"] = c["G"]
            d[n + ".scal"] = np.array([c["C"], c["Mmm"], c["zr"], c["zx"], c["zy"]], np.int64)   # C, Mmm, zr, zx, zy
        for l, L in enumerate(s.layers):
            p = f"L{l}."; nrm(p + "n1", L["n1"]); nrm(p + "n2", L["n2"])
            d[p + "qkv.M"] = L["qkv"][1]; d[p + "o.M"] = L["o"][1]; d[p + "u.M"] = L["u"][1]; d[p + "dn.M"] = L["dn"][1]
            d[p + "qr.C"], d[p + "qr.S"], d[p + "kr.C"], d[p + "kr.S"] = L["qrC"], L["qrS"], L["krC"], L["krS"]
            d[p + "exp_lut"] = L["exp"].astype(np.uint8); d[p + "rec_lut"] = L["rec"].astype(np.uint8); d[p + "gelu_lut"] = L["gelu"].astype(np.uint8)
            keys = "zqr zkr Mp Msc zsc Maf Mbf ZF D MD MP zv Mpv Mpb zao zo zn2 zu zg Mdp zdn qrzi krzi".split()
            S[p[:-1]] = {k: int(L[k]) for k in keys}; S[p[:-1]]["qz"] = [int(v) for v in L["qz"]]
            S[p[:-1]]["r1"] = [int(v) for v in L["r1"]]; S[p[:-1]]["r2"] = [int(v) for v in L["r2"]]   # (Ma, Mb, z_out, z_x, [z_dn])
        nrm("nf", s.nf); d["head.M"] = s.head[1]; S["head"] = {"znf": int(s.znf), "zlg": int(s.zlg)}; d["scalars_json"] = np.array(json.dumps(S))
        np.savez(path, **d)
    # ----- public API -----
    def forward_token(s, ids):
        """ids: list of token ids (context). Returns uint8 logits (V,) at the last position (full recompute)."""
        s.reset()
        for p, t in enumerate(ids): lg = s.step(t, p)
        return lg.astype(np.uint8)
    def generate(s, question, maxn=300):
        q = "".join(c for c in question.lower() if c in VOC); q = re.sub(r"\s+", " ", q).strip()[:140]
        toks = [VOC.index(c) for c in "q: " + q + "\na: "]; s.reset(); out = []; pos = 0
        for t in toks:
            lg = s.step(t, pos); pos += 1
        for _ in range(maxn):
            n = int(np.argmax(lg))
            if n == EOS: break
            out.append(VOC[n]); lg = s.step(n, pos); pos += 1
        s.maxpos = pos; return "".join(out)
