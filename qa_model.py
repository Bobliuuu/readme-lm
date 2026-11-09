"""Tiny QA transformer for README-LM. Char-level, RoPE, RMSNorm, tied embeddings,
4-bit group-quantized weights (QAT) so the whole model fits in a README."""
import math, torch, torch.nn as nn, torch.nn.functional as F

VOC = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"()-_/=+*#<>[]!?\n"
EOS = len(VOC); V = len(VOC) + 1
G = 32  # quant group

def fq(w, bits=4):
    """symmetric group fake-quant with straight-through estimator"""
    o, i = w.shape; g = w.view(o, i // G, G)
    qmax = 2 ** (bits - 1) - 1
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(g / s), -qmax, qmax) * s
    return (g + (q - g).detach()).view(o, i)

class QL(nn.Module):
    def __init__(s, i, o):
        super().__init__(); s.w = nn.Parameter(torch.randn(o, i) * i ** -0.5); s.qat = False
    def forward(s, x): return x @ (fq(s.w) if s.qat else s.w).t()

def rms(x, g): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g

def rope(x, T0=0):  # x: B,H,T,D
    B, H, T, D = x.shape
    f = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    a = torch.arange(T0, T0 + T, device=x.device).float()[:, None] * f[None]
    c, s = a.cos(), a.sin(); x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

class Block(nn.Module):
    def __init__(s, d, h, ff):
        super().__init__(); s.h = h
        s.qkv = QL(d, 3 * d); s.o = QL(d, d); s.u = QL(d, ff); s.dn = QL(ff, d)
        s.g1 = nn.Parameter(torch.ones(d)); s.g2 = nn.Parameter(torch.ones(d))
    def forward(s, x):
        B, T, D = x.shape; H = s.h
        q, k, v = s.qkv(rms(x, s.g1)).view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.dn(F.gelu(s.u(rms(x, s.g2))))

class QA(nn.Module):
    def __init__(s, d=128, L=3, h=4, ff=512):
        super().__init__(); s.d, s.L, s.h, s.ff = d, L, h, ff
        s.emb = nn.Parameter(torch.randn(V, d) * 0.02); s.qat = False
        s.blocks = nn.ModuleList(Block(d, h, ff) for _ in range(L)); s.gf = nn.Parameter(torch.ones(d))
    def set_qat(s, on):
        s.qat = on
        for m in s.modules():
            if isinstance(m, QL): m.qat = on
    def forward(s, idx):
        E = fq(s.emb) if s.qat else s.emb
        x = E[idx]
        for b in s.blocks: x = b(x)
        return rms(x, s.gf) @ E.t()
