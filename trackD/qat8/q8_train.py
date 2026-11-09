"""Torch QAT model whose forward mirrors q8_ref.py (8-bit activations at every tensor boundary)."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from q8_common import *

def rq(x): return x + (torch.floor(x + 0.5) - x).detach()     # round-half-up, STE
def fq4(w):                                                    # 4-bit group quant, fp16 scales (as exported to README)
    o, i = w.shape; g = w.view(o, i // GRP, GRP)
    s = (g.detach().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7).half().float()
    q = torch.clamp(torch.floor(g / s + 0.5), -7, 7) * s
    return (g + (q - g).detach()).view(o, i)
def h16(x): return x + (x.half().float() - x).detach()

class ScoreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qc, kc, zq, zk, cp, cs):
        B, H, T, D = qc.shape; qz = qc - zq; kz = kc - zk; out = torch.empty(B, H, T, T, device=qc.device)
        Tc = max(1, int(3e7 // (B * H * T * D)))
        for t0 in range(0, T, Tc):
            pr = torch.clamp(torch.floor(qz[:, :, t0:t0 + Tc, None, :] * kz[:, :, None] * cp + 0.5), -128, 127)
            out[:, :, t0:t0 + Tc] = pr.sum(-1) * cs
        ctx.save_for_backward(qz, kz); ctx.c = cp * cs
        return out
    @staticmethod
    def backward(ctx, g):
        qz, kz = ctx.saved_tensors; c = ctx.c
        return torch.einsum("bhts,bhsd->bhtd", g, kz) * c, torch.einsum("bhts,bhtd->bhsd", g, qz) * c, None, None, None, None

class PVFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, v, zv, cpv, mpb):     # p codes (B,H,T,T) 0..255; v codes (B,H,T,D); out signed, units of ao scale
        B, H, T, D = v.shape; vz = v - zv; out = torch.empty(B, H, T, D, device=p.device)
        Tc = max(1, int(3e7 // (B * H * T * D)))
        for t0 in range(0, T, Tc):
            pc = p[:, :, t0:t0 + Tc]; tot = 0
            for k0 in range(0, T, KB):
                pr = torch.clamp(torch.floor(pc[..., k0:k0 + KB, None] * vz[:, :, None, k0:k0 + KB] * cpv + 0.5), -128, 127)
                tot = tot + torch.clamp(torch.floor(pr.sum(-2) * mpb + 0.5), -128, 127)
            out[:, :, t0:t0 + Tc] = tot
        ctx.save_for_backward(p, vz); ctx.c = cpv * mpb
        return out
    @staticmethod
    def backward(ctx, g):
        p, vz = ctx.saved_tensors; c = ctx.c
        return torch.einsum("bhtd,bhsd->bhts", g, vz) * c, torch.einsum("bhts,bhtd->bhsd", p, g) * c, None, None, None

def pctl(x, p):
    x = x.detach().float().flatten().cpu()
    if x.numel() > 2_000_000: x = x[torch.randint(0, x.numel(), (2_000_000,))]
    return float(np.quantile(x.numpy(), p))

ISD = 1.0 / math.sqrt(HD)
FW = 9.0   # fine exp window (nats)
PCT = dict(default=0.9999, x=0.99999, r1=0.99999, r2=0.99999, emb=1.0)

class Q8(nn.Module):
    def __init__(s, spec=None):
        super().__init__(); d, ff = D_MODEL, D_FF
        s.emb = nn.Parameter(torch.randn(V, d) * 0.02)
        s.blk = nn.ModuleList(nn.ModuleDict(dict(qkv=nn.Linear(d, 3 * d, bias=False), o=nn.Linear(d, d, bias=False),
                  u=nn.Linear(d, ff, bias=False), dn=nn.Linear(ff, d, bias=False))) for _ in range(N_LAYER))
        s.g1 = nn.ParameterList(nn.Parameter(torch.ones(d)) for _ in range(N_LAYER))
        s.g2 = nn.ParameterList(nn.Parameter(torch.ones(d)) for _ in range(N_LAYER))
        s.gf = nn.Parameter(torch.ones(d)); s.spec = {} if spec is None else spec; s.lutc = {}
        s.valid = None; s.cal_pct = {}; s.noq = set(); s.dbg = None; s.noise = 0.0
    @staticmethod
    def from_fp(sd):
        m = Q8(); m.emb.data = sd["emb"].clone()
        for l in range(N_LAYER):
            for n in ("qkv", "o", "u", "dn"): m.blk[l][n].weight.data = sd[f"blocks.{l}.{n}.w"].clone()
            m.g1[l].data = sd[f"blocks.{l}.g1"].clone(); m.g2[l].data = sd[f"blocks.{l}.g2"].clone()
        m.gf.data = sd["gf"].clone(); return m
    # ---- spec helpers ----
    def fs(s, n): return s.spec[n + ".s"], s.spec[n + ".z"]
    def sel(s, x): return x[s.valid]
    def nz(s, t): return t if not s.noise else t + (torch.rand_like(t) * 2 - 1) * s.noise
    def fmt(s, n, x, kind="sym", sel=None, noisy=False):
        if n + ".s" not in s.spec:
            xv = (sel or s.sel)(x); key = n.split(".")[-1]; p = PCT.get(key, PCT["default"])
            if kind == "sym": sc, z = pctl(xv.abs(), p) / 127, 128
            elif kind == "uns": sc, z = pctl(xv, p) / 255, 0
            else:
                lo, hi = pctl(xv, 1 - p), pctl(xv, p); sc = (hi - lo) / 255; z = int(np.clip(rnd(-lo / sc), 0, 255))
            s.spec[n + ".s"] = max(float(sc), 1e-12); s.spec[n + ".z"] = int(z)
        sc, z = s.fs(n)
        if s.dbg is not None: s.dbg[n] = torch.clamp(rq(x / sc) + z, 0, 255).detach()
        if s.noq and n.split('.')[-1] in s.noq: return x / sc + z
        return torch.clamp(rq(s.nz(x / sc) if noisy else x / sc) + z, 0, 255)
    def lut(s, key, build):
        if key not in s.lutc: s.lutc[key] = torch.tensor(build(), dtype=torch.float32, device=s.emb.device)
        return s.lutc[key]
    @staticmethod
    def lapply(code, table, smooth):
        y = table[torch.clamp(torch.floor(code + 0.5), 0, 255).long()]; sm = smooth(code); return y + (sm - sm.detach())
    # ---- ops ----
    def lin(s, u, inn, W, outn, kind="sym"):
        si, zi = s.fs(inn); acc = ((u - zi) * si) @ W.t(); return s.fmt(outn, acc, kind, noisy=True)
    def norm(s, u, inn, g, n, out, valid_sel=None):
        sx, zx = s.fs(inn); xs = (u - zx) * sx
        if n + ".sq.s" not in s.spec: s.spec[n + ".sq.s"] = pctl(s.sel(xs * xs), PCT["x"]) / 255
        ssq = s.spec[n + ".sq.s"]
        sqc = s.lapply(u, s.lut(("sq", n), lambda: lut_sq(sx, zx, ssq)), lambda c: torch.clamp(((c - zx) * sx) ** 2 / ssq, 0, 255))
        S = sqc.sum(-1)
        if n + ".C" not in s.spec:
            lo, hi = float(s.sel(S).min()), float(s.sel(S).max()); C = math.floor(lo * 0.9); s.spec[n + ".C"] = C; s.spec[n + ".mm"] = 255.0 / (hi * 1.1 - C)
            r = 1.0 / np.sqrt(ms_real(np.arange(256.0), C, s.spec[n + ".mm"], ssq) + EPS); sr = (r.max() - r.min()) / 255
            s.spec[n + ".sr"] = float(sr); s.spec[n + ".zr"] = int(-math.floor(r.min() / sr))
        C, mm, sr, zr = s.spec[n + ".C"], s.spec[n + ".mm"], s.spec[n + ".sr"], s.spec[n + ".zr"]
        ms8 = torch.clamp(rq((S - C) * mm), 0, 255)
        def sm(c): return torch.clamp((1.0 / torch.sqrt((c / mm + C) * ssq / D_MODEL + EPS)) / sr + zr, 0, 255)
        r8 = s.lapply(ms8, s.lut(("rsq", n), lambda: lut_rsq(C, mm, ssq, sr, zr)), sm)
        y = xs * ((r8 - zr) * sr)[..., None] * h16(g)
        return s.fmt(out, y)
    def forward(s, idx, lens=None):
        B, T = idx.shape; dev = idx.device; H, D = N_HEAD, HD
        s.valid = torch.ones(B, T, dtype=torch.bool, device=dev) if lens is None else (torch.arange(T, device=dev)[None] < lens[:, None])
        E = fq4(s.emb); u = s.fmt("emb", E[idx], "sym")
        cos, sin = rope_tables(T); cos = torch.tensor(cos, dtype=torch.float32, device=dev); sin = torch.tensor(sin, dtype=torch.float32, device=dev)
        causal = torch.tril(torch.ones(T, T, device=dev)); cmask = causal[None, None]
        inn = "emb"
        for l in range(N_LAYER):
            p = f"L{l}."; b = s.blk[l]
            n1 = s.norm(u, inn, s.g1[l], p + "n1", p + "n1")
            si, zi = s.fs(p + "n1"); acc = ((n1 - zi) * si) @ fq4(b["qkv"].weight).t()
            qkv = acc.view(B, T, 3, H, D)
            q = s.fmt(p + "q", qkv[:, :, 0], noisy=True); k = s.fmt(p + "k", qkv[:, :, 1], noisy=True); v = s.fmt(p + "v", qkv[:, :, 2], noisy=True)
            def rope(c, nm):
                sc, z = s.fs(p + nm); xr = (c - z) * sc; x1, x2 = xr[..., 0::2], xr[..., 1::2]; c_, s_ = cos[None, :, None], sin[None, :, None]
                o = torch.stack([x1 * c_ - x2 * s_, x1 * s_ + x2 * c_], -1).flatten(-2); return o
            if p + "qr.s" not in s.spec: s.fmt(p + "qr", rope(q, "q"))
            if p + "kr.s" not in s.spec: s.fmt(p + "kr", rope(k, "k"))
            qr = s.fmt(p + "qr", rope(q, "q")).transpose(1, 2); kr = s.fmt(p + "kr", rope(k, "k")).transpose(1, 2)   # B,H,T,D
            sqr, zqr = s.fs(p + "qr"); skr, zkr = s.fs(p + "kr")
            qv = s.valid[:, None, :, None] & (causal[None, None] > 0)
            if p + "pr.s" not in s.spec:
                ii = torch.randint(0, T, (T,), device=dev); pr_ = ((qr - zqr) * sqr * (kr[:, :, ii] - zkr) * skr)[s.valid[:, None, :].expand(B, H, T)]
                s.spec[p + "pr.s"] = pctl(pr_.abs(), 0.9999) / 127; s.spec[p + "pr.z"] = 128
            spr = s.spec[p + "pr.s"]
            if p + "sc.s" not in s.spec:
                raw = ScoreFn.apply(qr, kr, float(zqr), float(zkr), sqr * skr / spr, 1.0)
                real = raw * spr * ISD; xv = real[qv.expand(B, H, T, T)]
                lo, hi = pctl(xv, 1 - 0.9999), pctl(xv, 0.9999); scs = (hi - lo) / 255
                s.spec[p + "sc.s"] = float(scs); s.spec[p + "sc.z"] = int(np.clip(rnd(-lo / scs), 0, 255))
            ssc, zsc = s.fs(p + "sc")
            if p + "f.s" not in s.spec: s.spec[p + "f.s"] = FW / 255; s.spec[p + "f.z"] = int(rnd(ssc / (FW / 255)))
            sf, ZF = s.fs(p + "f")
            raw0 = ScoreFn.apply(qr, kr, float(zqr), float(zkr), sqr * skr / spr, 1.0)
            sc = torch.clamp(rq(s.nz(raw0 * (spr * ISD / ssc))) + zsc, 0, 255)
            if s.dbg is not None: s.dbg[p + 'sc'] = sc.detach().transpose(1, 2)
            m = torch.where(cmask > 0, sc, torch.full_like(sc, -1.0)).amax(-1, keepdim=True)
            dd = torch.clamp(rq(s.nz((m - zsc) * (ssc / sf) - raw0 * (spr * ISD / sf))) + ZF, 0, 255)
            e = s.lapply(dd, s.lut(("exp", p), lambda: lut_exp(sf)), lambda c: 255.0 * torch.exp(-c * sf)) * cmask
            Sg = e.sum(-1)
            if p + "D" not in s.spec: s.spec[p + "D"] = int(2 ** max(0, math.ceil(math.log2(max(1.0, pctl(Sg[s.valid[:, None, :].expand(B, H, T)], 0.995) / 255)))))
            Dv = s.spec[p + "D"]
            S8 = torch.clamp(rq(Sg / Dv), 0, 255)
            R8 = s.lapply(S8, s.lut(("rec", p), lambda: lut_rec(Dv)), lambda c: torch.clamp(255.0 / (torch.clamp(c, min=1) * Dv) * SR_INV, 0, 255))
            p8 = torch.clamp(rq(e * R8[..., None] / SR_INV), 0, 255) * cmask       # (B,H,T,T) e: scale to 255-based p
            sv, zv = s.fs(p + "v"); vh = v.transpose(1, 2)
            if p + "pv.s" not in s.spec:
                jj = torch.randint(0, T, (32,), device=dev)
                pv_ = (p8[:, :, jj] / 255)[..., None] * ((vh - zv) * sv)[:, :, None]   # B,H,32,T,D
                pv_ = pv_[s.valid[:, jj][:, None, :, None, None].expand_as(pv_)]
                s.spec[p + "pv.s"] = pctl(pv_.abs(), 0.9999) / 127; s.spec[p + "pv.z"] = 128
            spv = s.spec[p + "pv.s"]
            s.spec[p + "ao.s"] = spv; s.spec[p + "ao.z"] = 128
            sao, zao = s.fs(p + "ao")
            ao = torch.clamp(PVFn.apply(p8, vh, float(zv), sv / (255 * spv), spv / sao) + 128, 0, 255).transpose(1, 2).reshape(B, T, D_MODEL)
            assert zao == 128
            if s.dbg is not None: s.dbg[p + 'ao'] = ao.detach(); s.dbg[p + 'p8'] = p8.detach().transpose(1, 2)
            o = s.lin(ao, p + "ao", fq4(b["o"].weight), p + "o")
            sx, zx = s.fs(inn); so, zo = s.fs(p + "o")
            r1 = s.fmt(p + "r1", (u - zx) * sx + (o - zo) * so)
            n2 = s.norm(r1, p + "r1", s.g2[l], p + "n2", p + "n2")
            uu = s.lin(n2, p + "n2", fq4(b["u"].weight), p + "u")
            su, zu = s.fs(p + "u")
            if p + "g.s" not in s.spec: s.fmt(p + "g", F.gelu((uu - zu) * su), "asym")
            sg, zg = s.fs(p + "g")
            g = s.lapply(uu, s.lut(("gelu", p), lambda: lut_gelu(su, zu, sg, zg)), lambda c: torch.clamp(F.gelu((c - zu) * su) / sg + zg, 0, 255))
            Wd = fq4(b["dn"].weight); acc = []
            if p + "dp.s" not in s.spec:
                a0 = ((g[..., :256] - zg) * sg) @ Wd[:, :256].t(); s.fmt(p + "dp", a0)
            sdp, zdp = s.fs(p + "dp")
            parts = [torch.clamp(rq(s.nz(((g[..., h * 256:(h + 1) * 256] - zg) * sg) @ Wd[:, h * 256:(h + 1) * 256].t() / sdp)) + 128, 0, 255) for h in range(2)]
            assert zdp == 128
            dn = s.fmt(p + "dn", (parts[0] - 128 + parts[1] - 128) * sdp)
            sdn, zdn = s.fs(p + "dn"); s1, z1 = s.fs(p + "r1")
            u = s.fmt(p + "r2", (r1 - z1) * s1 + (dn - zdn) * sdn); inn = p + "r2"
        nf = s.norm(u, inn, s.gf, "nf", "nf")
        lg = s.lin(nf, "nf", E, "lg")
        sl, zl = s.fs("lg"); return (lg - zl) * sl, lg
