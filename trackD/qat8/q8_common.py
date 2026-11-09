"""Shared definitions for the 8-bit-activation (SVG-filter) QA transformer: LUT builders (float64 numpy,
deterministic) used by BOTH the torch QAT model and the integer reference."""
import math, numpy as np
VOC = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"()-_/=+*#<>[]!?\n"
EOS = len(VOC); V = len(VOC) + 1
D_MODEL, N_LAYER, N_HEAD, D_FF, GRP = 128, 3, 4, 512, 32
HD = D_MODEL // N_HEAD
SH = 32                      # fixed-point shift for all multipliers
SR_INV = 256                 # reciprocal tensor R8 scale: R = R8/256
KB = 256                     # PV key-block size (convolve kernel limit)
EPS = 1e-5
ROPE_F = 1.0 / (10000 ** (np.arange(0, HD, 2) / HD))   # float64

def rnd(x):  # round half up
    return np.floor(np.asarray(x, np.float64) + 0.5)
def fxm(ratio):  # float ratio -> int fixed-point multiplier, shift SH
    return np.floor(np.asarray(ratio, np.float64) * 2.0 ** SH + 0.5).astype(np.int64)
_erf = np.vectorize(math.erf)
def gelu64(x): x = np.asarray(x, np.float64); return 0.5 * x * (1 + _erf(x / math.sqrt(2)))
def clip8(x): return np.clip(x, 0, 255)

# ---- LUTs: index = uint8 code, value = uint8 code (int64 arrays of length 256) ----
U = np.arange(256, dtype=np.float64)
def lut_sq(sx, zx, s_sq):            # square, unsigned out
    return clip8(rnd((U - zx) ** 2 * sx * sx / s_sq)).astype(np.int64)
def ms_real(ms8, C, mm, s_sq):       # mean of squares (real) represented by code ms8
    return (np.asarray(ms8, np.float64) / mm + C) * s_sq / D_MODEL
def lut_rsq(C, mm, s_sq, sr, zr):    # 1/sqrt(mean_sq+eps) -> r8 ; r = (r8-zr)*sr
    r = 1.0 / np.sqrt(ms_real(U, C, mm, s_sq) + EPS)
    return clip8(rnd(r / sr) + zr).astype(np.int64)
def lut_exp(s_sc):                   # dd=(max-score) code diff -> e8 = 255*exp(-dd*s_sc)
    return clip8(rnd(255.0 * np.exp(-U * s_sc))).astype(np.int64)
def lut_rec(Dv):                     # S8 (sum/D) -> R8 = 255/(S8*D) * 256
    return clip8(rnd(255.0 / (np.maximum(U, 1) * Dv) * SR_INV)).astype(np.int64)
def lut_gelu(su, zu, sg, zg):
    return clip8(rnd(gelu64((U - zu) * su) / sg) + zg).astype(np.int64)
def rope_tables(T):                  # float64 cos/sin (T, HD/2)
    a = np.arange(T, dtype=np.float64)[:, None] * ROPE_F[None]
    return np.cos(a), np.sin(a)
