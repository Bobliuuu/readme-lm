"""SVG filter op library for the README-LM QA transformer (no JS; filter primitives only).
Conventions (verified in headless Chrome, see PROBES.md / REPORT.md):
  * every tensor is a uint8 code stored in the R channel (G,B mirror R unless noted); A=255 everywhere.
  * layout: filter region W cols x H rows; rows are tokens (all tokens processed in parallel), cols are channels.
  * feConvolveMatrix: kernel written reversed; out(x)=sum_i K[i]*in[x0+i]; targetX selects window start (x0=x-targetX).
    Skia quantises kernel to 8 bits over [kmin,kmax]; EXACT only if kmax-kmin divides 255 -> a guard tap
    (value kmin+255 on a 0-valued guard pixel) forces range 255 (see linear()).
  * out = floor(S/d + bias*255 + 0.5): made deterministic by a +0.25/d nudge (no exact ties).
  * feComposite arithmetic: alpha_out=k1+k2+k3+k4 must be >=1 (else premultiplied colour is clipped): asserted.
All emit functions take a Ctx and return result names; *_ref are exact integer numpy mirrors."""
import numpy as np
from core import *

def f12(x): return f"{x:.12g}"

class Ctx:
    def __init__(s, W, H): s.g = Graph(W, H); s.W, s.H = W, H; s.rows = H; s.bgs = {}; s.cache = {}
    def prim(s, xml, res=None): return s.g.add(xml, res)
    def bgf(s, h=None):
        key = h or s.rows; wd = s.W if key == s.rows else min(s.W, 200)
        if key not in s.bgs: s.bgs[key] = s.prim(f'<feFlood flood-color="#000" flood-opacity="1" x="0" y="0" width="{wd}" height="{key}" RESULT/>')
        return s.bgs[key]
    def image(s, arr, x=0, y=0, res=None): return image_prim(s.g, arr, x, y, res)
    def merge(s, names, res=None, h=None):
        return s.prim("<feMerge RESULT>" + "".join(f'<feMergeNode in="{n}"/>' for n in [s.bgf(h)] + list(names)) + "</feMerge>", res)
    def offset(s, src, dx, dy=0, x=None, y=None, w=None, h=None):
        if dx == 0 and dy == 0 and x is None: return src
        key = (src, dx, dy, x, y, w, h)
        if key in s.cache: return s.cache[key]
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        r = s.prim(f'<feOffset in="{src}" dx="{dx}" dy="{dy}"{sub} RESULT/>'); s.cache[key] = r; return r
    def tile(s, src, x, y, w, h): return s.prim(f'<feTile in="{src}" x="{x}" y="{y}" width="{w}" height="{h}" RESULT/>')
    def lut(s, src, table, x=None, y=None, w=None, h=None, chans="R"):
        t = " ".join(f"{(int(v) + 0.5) / 255:.5f}" for v in table); f = f'type="discrete" tableValues="{t}"'
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        return s.prim(f'<feComponentTransfer in="{src}"{sub} RESULT>' + "".join(f"<feFunc{c} {f}/>" for c in chans) + "</feComponentTransfer>")
    def cmatrix(s, src, rows, x=None, y=None, w=None, h=None):
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        return s.prim(f'<feColorMatrix in="{src}" type="matrix" values="{" ".join(f12(v) for r in rows for v in r)}"{sub} RESULT/>')
    def arith(s, a, b, k1, k2, k3, k4, x=None, y=None, w=None, h=None):
        assert k1 + k2 + k3 + k4 >= 1 - 1e-9, f"alpha<1 ({k1+k2+k3+k4})"
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        return s.prim(f'<feComposite in="{a}" in2="{b}" operator="arithmetic" k1="{f12(k1)}" k2="{f12(k2)}" k3="{f12(k3)}" k4="{f12(k4)}"{sub} RESULT/>')

# ------------------------------------------------------------------ O4 add / O? product -------------
def add_u8(c, a, b, off=0, **sub):
    """clip(a + b + off) on codes (off integer levels)."""
    return c.arith(a, b, 0, 1, 1, off / 255.0, **sub)
def add_u8_ref(a, b, off=0): return np.clip(np.asarray(a, np.int64) + b + off, 0, 255)

def neg_u8(c, a, **sub):   # 255-a on all RGB channels, broadcast from R
    return c.cmatrix(a, [[-1, 0, 0, 0, 1], [-1, 0, 0, 0, 1], [-1, 0, 0, 0, 1], [0, 0, 0, 1, 0]], **sub)

def prod(c, a, b, za, zb, zout, S, **sub):
    """clip(round_nudged((a-za)*(b-zb)/S)+zout). a,b,out are code images. nudge +0.25/S levels (tie-free)."""
    k1 = 255.0 / S; k2 = -zb / S; k3 = -za / S; k4 = (za * zb / S + zout + 0.25 / S) / 255.0
    return c.arith(a, b, k1, k2, k3, k4, **sub)
def prod_ref(a, b, za, zb, zout, S):
    P = (np.asarray(a, np.int64) - za) * (np.asarray(b, np.int64) - zb)
    return np.clip(np.floor_divide(4 * P + 2 * S + 1, 4 * S) + zout, 0, 255)

# ------------------------------------------------------------------ O1 linear ---------------------------
def quantize_weights(w_real, wmax=127):
    """real per-input-code gain (n_out,n_in) -> (int kernel e, integer divisor d): out ~ sum e*(u-zin)/d."""
    d = max(1, int(np.floor(wmax / np.abs(w_real).max())))
    return np.round(w_real * d).astype(np.int64), d

def split_chunks(n, maxc=255):
    k = -(-n // maxc); base, rem = divmod(n, k); return [base + (1 if i < rem else 0) for i in range(k)]

def conv_neuron(c, src, n, x_out, win_start, taps, d, Bint, zp, region_h=None, guard=None):
    """one output neuron: semantic taps (len n) applied to src cols [win_start, win_start+n) (+guard col win_start+n which MUST be 0),
    result at column x_out (src is already shifted so that window start is win_start)."""
    taps = np.asarray(taps, np.int64); g = int(taps.min()) + 255 if guard is None else guard
    assert taps.max() <= g and min(g, taps.min()) <= g and (max(g, taps.max()) - min(g, taps.min())) in (255, 85, 51, 17, 15, 5, 3, 1), "kernel range must divide 255"
    t = x_out - win_start; assert 0 <= t <= n, (t, n)
    K = np.concatenate([taps, [g]])[::-1]
    bias = ((Bint + 0.25) / d + zp) / 255.0
    return c.prim(f'<feConvolveMatrix in="{src}" order="{n+1} 1" kernelMatrix="{" ".join(map(str, K.tolist()))}" divisor="{d}" bias="{f12(bias)}" '
                  f'targetX="{t}" targetY="0" edgeMode="none" preserveAlpha="true" x="{x_out}" y="0" width="1" height="{region_h or c.rows}" RESULT/>')

def linear(c, src, chunks, E, d, zin, zout, bacc, out_cols):
    """chunks: list of (col0, n) input windows (each followed by a 0-valued guard column). E: (n_out, sum n) int kernel;
    bacc: (n_out,) integer accumulator bias (acc units, added before /d). Returns image name (outputs at out_cols, rest black).
    Multi-chunk: each chunk requantised to codes with zero-point 128 then added (sequential saturating adds)."""
    n_out = E.shape[0]; K = len(chunks); parts = []; off = 0
    for ci, (c0, n) in enumerate(chunks):
        Ec = E[:, off:off + n]; off += n; zp = zout if K == 1 else 128; outs = []
        # group outputs so that each group shares one shifted copy of src (window must satisfy 0<=x-start<=n)
        j = 0
        while j < n_out:
            x0 = out_cols[j]; jj = j
            while jj < n_out and out_cols[jj] - x0 <= n: jj += 1
            assert c.W >= x0 + n + 1, f'filter region too narrow: window [{x0},{x0+n}] exceeds W={c.W} (Chrome mis-evaluates partial windows)'
            sh = c.offset(src, x0 - c0)            # window start moves to c0+dx = x0
            for jx in range(j, jj):
                Bint = int(bacc[jx]) * (1 if ci == 0 else 0) - zin * int(Ec[jx].sum())
                outs.append(conv_neuron(c, sh, n, out_cols[jx], x0, Ec[jx], d, Bint, zp))
            j = jj
        parts.append(c.merge(outs))
    acc = parts[0]
    for i in range(1, K):
        o = -128 if i < K - 1 else zout - 256 + 128 - 128 + 128 - 128   # see below
        o = -128 if i < K - 1 else (zout - 128) - 128
        acc = add_u8(c, acc, parts[i], o)
    return acc

def linear_ref(U, chunks_n, E, d, zin, zout, bacc):
    U = np.asarray(U, np.int64); K = len(chunks_n); off = 0; parts = []
    for ci, n in enumerate(chunks_n):
        S = (E[:, off:off + n][None] * (U[:, None, off:off + n] - zin)).sum(-1) + (bacc[None] if ci == 0 else 0); off += n
        zp = zout if K == 1 else 128
        parts.append(np.clip(np.floor_divide(4 * S + 2 * d + 1, 4 * d) + zp, 0, 255))
    acc = parts[0]
    for i in range(1, K): acc = add_u8_ref(acc, parts[i], -128 if i < K - 1 else (zout - 128) - 128)
    return acc

# ------------------------------------------------------------------ helpers ----------------------------
def rdiv(num4, den4):   # floor_divide on ints with sign safety
    return np.floor_divide(num4, den4)
def nudge_round(P, S):  # round-half-up of P/S with tie-free nudge, integers
    return np.floor_divide(4 * np.asarray(P, np.int64) + 2 * S + 1, 4 * S)

def finish(c, name, h=None):    # opaque over black
    return c.merge([name], h=h)

# ------------------------------------------------------------------ O2 LUT ------------------------------
def lut_op(c, src, table, x=None, w=None):
    return c.lut(src, table, x=x, w=w)
def lut_ref(U, table): return np.asarray(table, np.int64)[np.asarray(U, np.int64)]

# ------------------------------------------------------------------ O4 residual add with scale change ---
def resadd(c, a, b, pa, pb, za, zb, zo, D=256, **sub):
    """out=clip(round_nudged((pa*(a-za)+pb*(b-zb))/D)+zo); pa,pb ints (ratios*D). Needs (pa+pb)/D >= ~1 for alpha."""
    k2 = pa / D; k3 = pb / D; k4 = (zo - k2 * za - k3 * zb + 0.25 / D) / 255.0
    return c.arith(a, b, 0, k2, k3, k4, **sub)
def resadd_ref(a, b, pa, pb, za, zb, zo, D=256):
    P = pa * (np.asarray(a, np.int64) - za) + pb * (np.asarray(b, np.int64) - zb)
    return np.clip(nudge_round(P, D) + zo, 0, 255)

# ------------------------------------------------------------------ O3 RMSNorm --------------------------
def rmsnorm(c, x, xc0, n, P, gimg_arr=None):
    """x: opaque image with n data cols at [xc0,xc0+n) (guard col after). P: dict(sq,rsq tables, C, e, d, zx, zr, S1, zy, S2, gq(n,)).
    returns opaque image with the normalised vector at the same columns."""
    sq = c.lut(x, P["sq"], x=xc0, w=n)
    ms = conv_neuron(c, sq, n, xc0, xc0, [P["e"]] * n, P["d"], -P["C"] * P["e"], 0)
    r8 = c.lut(ms, P["rsq"])
    rb = c.tile(r8, xc0, 0, n, c.rows)
    t = prod(c, x, rb, P["zx"], P["zr"], 128, P["S1"], x=xc0, y=0, w=n, h=c.rows)
    if P.get("fold"): return finish(c, t)        # gain folded into the consumer's weights
    g = c.image(np.tile(np.asarray(P["gq"], np.uint8)[None], (c.rows, 1)), x=xc0, y=0)
    y = prod(c, t, g, 128, 0, P["zy"], P["S2"], x=xc0, y=0, w=n, h=c.rows)
    return finish(c, y)
def rmsnorm_ref(X, P):
    X = np.asarray(X, np.int64); S = np.asarray(P["sq"], np.int64)[X].sum(1)
    ms = np.clip(nudge_round(S * P["e"] - P["C"] * P["e"], P["d"]), 0, 255)
    r8 = np.asarray(P["rsq"], np.int64)[ms]
    t = prod_ref(X, r8[:, None], P["zx"], P["zr"], 128, P["S1"])
    if P.get("fold"): return t
    return prod_ref(t, np.asarray(P["gq"], np.int64)[None], 128, 0, P["zy"], P["S2"])

# ------------------------------------------------------------------ O5 embedding / argmax --------------
def embed_lookup(c, ids_arr, emb_codes, ec0, idcol=0):
    """token ids (H,) -> embedding codes (H,n) via per-channel LUTs on the (tiled) id pixel.
    id pixel code u_k=floor((k+.5)*255/V) so that a V-entry discrete table indexes exactly k (keeps tables tiny)."""
    V, n = emb_codes.shape
    uk = np.floor((np.arange(V) + 0.5) * 255 / V).astype(np.uint8)
    idimg = np.zeros((c.rows, 1), np.uint8); idimg[:, 0] = uk[np.asarray(ids_arr)]
    c.image(idimg, x=idcol, y=0, res="ids")
    tl = c.tile("ids", ec0, 0, n, c.rows)
    outs = []
    for j in range(n):
        t = " ".join(f"{(int(v) + 0.5) / 255:.5f}" for v in emb_codes[:, j])
        outs.append(c.prim(f'<feComponentTransfer in="{tl}" x="{ec0+j}" y="0" width="1" height="{c.rows}" RESULT><feFuncR type="discrete" tableValues="{t}"/></feComponentTransfer>'))
    return c.merge(outs)
def embed_ref(ids, emb_codes): return np.asarray(emb_codes, np.int64)[np.asarray(ids)]

def argmax_row(c, L, ncols_valid, W_pow2=128):
    """L: opaque image, logit codes in R at cols [0,ncols_valid) (cols >= ncols_valid must be 0).  Exact first-index argmax in ~10 primitives, no tournament (result id at col 127, R):
    m = dilate(L, rx=127) (row max broadcast to every col); eq=(m-L==0) ; J = idx where eq else 255 ; id = erode(J, rx=127) (min index).  """
    n = W_pow2; H = c.rows
    ramp = np.zeros((1, n), np.uint8); ramp[0] = np.arange(n)
    rampi = c.image(np.tile(ramp, (H, 1)), x=0, y=0)
    m = c.prim(f'<feMorphology in="{L}" operator="dilate" radius="{n-1} 0" x="0" y="0" width="{n}" height="{H}" RESULT/>')
    nL = neg_u8(c, L, x=0, y=0, w=n, h=H)
    d = add_u8(c, m, nL, -255, x=0, y=0, w=n, h=H)                       # m - L >= 0
    teq = [255 if v == 0 else 0 for v in range(256)]; tne = [0 if v == 0 else 255 for v in range(256)]
    eq = c.lut(d, teq, x=0, y=0, w=n, h=H); ne = c.lut(d, tne, x=0, y=0, w=n, h=H)
    Ie = c.arith(rampi, eq, 1, 0, 0, 0, x=0, y=0, w=n, h=H)               # idx*eq/255 = idx if eq else 0
    J = add_u8(c, Ie, ne, 0, x=0, y=0, w=n, h=H)
    wb = c.prim(f'<feFlood flood-color="#fff" flood-opacity="1" x="0" y="0" width="{2*n-1}" height="{H}" RESULT/>')
    Jw = c.prim(f'<feMerge x="0" y="0" width="{2*n-1}" height="{H}" RESULT><feMergeNode in="{wb}"/><feMergeNode in="{J}"/></feMerge>')   # 255 (neutral for min) outside the data cols
    return c.prim(f'<feMorphology in="{Jw}" operator="erode" radius="{n-1} 0" x="0" y="0" width="{2*n-1}" height="{H}" RESULT/>')   # id at col n-1
def argmax_ref(Lcodes):
    return np.asarray(Lcodes).argmax(-1)       # numpy: lowest index on ties

# ------------------------------------------------------------------ O6 RoPE ----------------------------
def rope_apply(c, x, cols, cosc, sinc, z, zo, R, wspan=None):
    """x: opaque image; logical channel j lives at physical column cols[j] (pairs (2i,2i+1) adjacent). cosc/sinc: (rows, n/2) ints = round(R*cos*ratio).
    out_even = x_e*C - x_o*S ; out_odd = x_e*S + x_o*C, result zero-point zo. partner image via 2 offsets + parity masks; u = x*C ; v = P*S ; out = u+v."""
    H = c.rows; cols = np.asarray(cols); n = len(cols); w = wspan or (int(cols.max()) + 2)
    par = np.zeros((1, w), np.uint8); par[0, cols[0::2]] = 255
    npar = np.zeros((1, w), np.uint8); npar[0, cols[1::2]] = 255
    Me = c.image(np.tile(par, (H, 1)), x=0, y=0); Mo = c.image(np.tile(npar, (H, 1)), x=0, y=0)
    Xr = c.offset(x, -1, 0); Xl = c.offset(x, 1, 0)             # Xr[c]=x[c+1], Xl[c]=x[c-1]
    A = c.arith(Xr, Me, 1, 0, 0, 0, x=0, y=0, w=w, h=H); B = c.arith(Xl, Mo, 1, 0, 0, 0, x=0, y=0, w=w, h=H)
    Pn = add_u8(c, A, B, 0, x=0, y=0, w=w, h=H)
    Cf = np.zeros((H, w), np.int64); Sf = np.zeros((H, w), np.int64)
    Cf[:, cols[0::2]] = cosc; Cf[:, cols[1::2]] = cosc; Sf[:, cols[0::2]] = -sinc; Sf[:, cols[1::2]] = sinc
    Ci = c.image((Cf + 128).astype(np.uint8), x=0, y=0); Si = c.image((Sf + 128).astype(np.uint8), x=0, y=0)
    u = prod(c, x, Ci, z, 128, 128, R, x=0, y=0, w=w, h=H)
    v = prod(c, Pn, Si, z, 128, 128, R, x=0, y=0, w=w, h=H)
    return finish(c, add_u8(c, u, v, zo - 256, x=0, y=0, w=w, h=H))
def rope_ref(X, cosc, sinc, z, zo, R):
    X = np.asarray(X, np.int64); H, n = X.shape
    P = np.empty_like(X); P[:, 0::2] = X[:, 1::2]; P[:, 1::2] = X[:, 0::2]
    # mask products: partner code (not zero-point shifted); at image edges (none: pairs inside)
    Cf = np.zeros((H, n), np.int64); Sf = np.zeros((H, n), np.int64)
    Cf[:, 0::2] = cosc; Cf[:, 1::2] = cosc; Sf[:, 0::2] = -sinc; Sf[:, 1::2] = sinc
    u = prod_ref(X, Cf + 128, z, 128, 128, R); v = prod_ref(P, Sf + 128, z, 128, 128, R)
    return np.clip(u + v + (zo - 256), 0, 255)

# ------------------------------------------------------------------ O8 glyph atlas ----------------------
def make_atlas(voc_chars, gw=16, gh=24, fsize=18):
    """PIL-rendered glyph strip (white ink on black), len(voc_chars) cells of gw x gh. Special names: '\\n' -> arrow, 'EOS' -> box."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=fsize)
    A = Image.new("L", (gw * len(voc_chars), gh), 0); d = ImageDraw.Draw(A)
    for i, ch in enumerate(voc_chars):
        ox = i * gw
        if ch == "\n":
            d.line([(ox + gw - 4, 4), (ox + gw - 4, gh // 2), (ox + 3, gh // 2)], fill=255, width=2); d.polygon([(ox + 2, gh // 2), (ox + 7, gh // 2 - 4), (ox + 7, gh // 2 + 4)], fill=255)
        elif ch == "EOS": d.rectangle([ox + 2, 3, ox + gw - 3, gh - 4], outline=255, width=2)
        elif ch != " ": 
            bb = d.textbbox((0, 0), ch, font=font); w, h = bb[2] - bb[0], bb[3] - bb[1]
            d.text((ox + (gw - w) // 2 - bb[0], (gh - h) // 2 - bb[1]), ch, font=font, fill=255)
    return np.array(A, np.uint8)

def glyph_stage(c, X, row, atlas, gw, gh, oy=0):
    """X: image whose pixel (0,row).G = token id (<=170). Displays glyph `id` of the atlas strip at (ox,oy). Uses feDisplacementMap:
    dx = (scale/255)*(R-127.5), scale=255*gw -> exact integer gather (verified).  Returns (result, ox)."""
    ox = 340 * gw // 8 if False else None
    scale = 255 * gw
    off = 85                       # code = id + off  -> dx = gw*(id+off-127.5)
    ox = gw * (127.5 - off)          # atlas x0 = ox - gw*(127.5-off) ... choose atlas at x=0 => ox = gw*(127.5-off)
    ox = int(ox); assert ox == gw * 127.5 - gw * off
    ycode = 128; dy = int(round(gw * (ycode - 127.5)))     # = +gw/2
    idpx = c.cmatrix(X, [[1, 0, 0, 0, off / 255.0], [0, 0, 0, 0, ycode / 255.0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0]], x=127, y=row, w=1, h=1)
    mp = c.tile(idpx, ox, oy, gw, gh)
    at = c.image(atlas, x=0, y=oy + dy, res=None)             # atlas row 0 must land where output row oy reads: out(y) reads atlas at y+dy
    out = c.prim(f'<feDisplacementMap in="{at}" in2="{mp}" scale="{scale}" xChannelSelector="R" yChannelSelector="G" x="{ox}" y="{oy}" width="{gw}" height="{gh}" RESULT/>')
    return out, ox

def shift_conv(c, src, w, vertical, nfix, x, y, wd, h, forward=False):
    """exact copy of src shifted by w>=0 pixels: backward (default) out[pos]=src[pos-w]; forward out[pos]=src[pos+w].
    One-tap convolve of FIXED order nfix (every shift requests the same inflated input rect: avoids the exponential re-evaluation
    that feOffset diamonds cause in tournaments).  Edge pixels (window not fully inside the region) are garbage."""
    K = [0] * nfix
    if forward: K[nfix - 1 - w] = 1; t = 0                    # semantic tap w, window start = pos
    else: K[w] = 1; t = nfix - 1                              # semantic tap nfix-1-w, window start = pos-(nfix-1)
    order = f"1 {nfix}" if vertical else f"{nfix} 1"; tx, ty = (0, t) if vertical else (t, 0)
    return c.prim(f'<feConvolveMatrix in="{src}" order="{order}" kernelMatrix="{" ".join(map(str, K))}" divisor="1" bias="0.00098039216" '
                  f'targetX="{tx}" targetY="{ty}" edgeMode="none" preserveAlpha="true" x="{x}" y="{y}" width="{wd}" height="{h}" RESULT/>')

# ------------------------------------------------------------------ generic window convolves --------------
def conv_all(c, src, vertical, n, e, d, Bint, zp, shift, x, w, y, h, last=0):
    """one primitive over a whole sub-rectangle: every output pixel = clip(floor((4*(e*sum_{i<n} in[pos+i-shift]+Bint)+2d+1)/(4d))+zp)
    shift = targetX/targetY (window start = pos-shift).  extra tap value `last` (0) makes the kernel range = e (e must divide 255)."""
    assert 255 % e == 0 or e == 0, e
    K = ([e] * n + [last])[::-1]; bias = ((Bint + 0.25) / d + zp) / 255.0
    order = f"1 {n+1}" if vertical else f"{n+1} 1"; tx, ty = (0, shift) if vertical else (shift, 0)
    return c.prim(f'<feConvolveMatrix in="{src}" order="{order}" kernelMatrix="{" ".join(map(str, K))}" divisor="{d}" bias="{f12(bias)}" '
                  f'targetX="{tx}" targetY="{ty}" edgeMode="none" preserveAlpha="true" x="{x}" y="{y}" width="{w}" height="{h}" RESULT/>')
def conv_all_ref(A, vertical, n, e, d, Bint, zp, shift):
    """A: 2D int array (full region). out[pos] = clip(round(...)) using window pos-shift..pos-shift+n-1 ; out of range -> 0 (invalid at edges)"""
    A = np.asarray(A, np.int64); out = np.zeros_like(A)
    if vertical:
        Hh = A.shape[0]
        for y in range(Hh):
            lo = y - shift
            if lo < 0 or lo + n > Hh: continue
            out[y] = e * A[lo:lo + n].sum(0)
    else:
        Ww = A.shape[1]
        for x in range(Ww):
            lo = x - shift
            if lo < 0 or lo + n > Ww: continue
            out[:, x] = e * A[:, lo:lo + n].sum(1)
    return np.clip(np.floor_divide(4 * (out + Bint) + 2 * d + 1, 4 * d) + zp, 0, 255)
def best_ed(ratio, dmax=3000):
    """e in divisors of 255, integer d minimising |e/d-ratio|/ratio"""
    best = None
    for e in (1, 3, 5, 15, 17, 51, 85, 255):
        d = max(1, int(round(e / ratio)))
        if d > dmax: continue
        err = abs(e / d - ratio) / ratio
        if best is None or err < best[0]: best = (err, e, d)
    return best[1], best[2], best[0]

# ------------------------------------------------------------------ O7 causal attention ------------------
HS = 34                                 # head stride in the attention layout: 32 data cols + 1 'm' col + 1 zero guard col
ACOLS = [HS * (j // 32) + j % 32 for j in range(128)]
WA = 4 * HS
def attn_H(T):
    m, P = pick_pitch(T); return T * P + 2 * P + 8
def conv_taps(c, src, taps, guard, d, Bint, zp, shift, x, w, y, h, vertical=False):
    """one primitive, arbitrary integer taps (+ one guard tap reading a zero pixel) : kernel range must divide 255."""
    taps = list(taps); K = (taps + [guard])[::-1]; lo, hi = min(taps + [guard]), max(taps + [guard]); assert hi - lo in (255, 85, 51, 17, 15, 5, 3, 1), (lo, hi)
    bias = ((Bint + 0.25) / d + zp) / 255.0
    order = f"1 {len(K)}" if vertical else f"{len(K)} 1"; tx, ty = (0, shift) if vertical else (shift, 0)
    return c.prim(f'<feConvolveMatrix in="{src}" order="{order}" kernelMatrix="{" ".join(map(str, K))}" divisor="{d}" bias="{f12(bias)}" '
                  f'targetX="{tx}" targetY="{ty}" edgeMode="none" preserveAlpha="true" x="{x}" y="{y}" width="{w}" height="{h}" RESULT/>')
def attention(c, Q, K, V, P_, T, upto=None):
    """Q,K,V: opaque images, head-strided layout (ACOLS: head h at cols 34h..34h+31), rows 0..T-1. Spec step 1.5/1.6 incl. the FINE max-subtracted
    score (one convolve over 32 products + the max pixel).  Stacked layout with pitch P (>=T, 2m | P-1): row r = t*P+u.  Row regroupings =
    2 exact feDisplacementMap gathers; max = tournament (one-tap shift convolves); sums/broadcasts = vertical ones-convolves.
    Output: ao codes, head-strided (junk in gap cols), rows 0..T-1.  Region height >= attn_H(T), width >= WA+40."""
    m, P = pick_pitch(T); TT = T * P; W, Hb = c.W, c.H; assert Hb >= TT + 2 * P and W >= WA + 40
    rr = np.arange(TT + P); tq, uk = rr // P, rr % P
    def col_raster(vals, name):
        c.image(np.asarray(vals, np.uint8).reshape(-1, 1), x=0, y=0, res=name + "_1"); return c.tile(name + "_1", 0, 0, WA, TT + P)
    def row_raster(cols_on, name):
        a = np.zeros((1, WA), np.uint8); a[0, cols_on] = 255
        c.image(a, x=0, y=0, res=name + "_1"); return c.tile(name + "_1", 0, 0, WA, TT + P)
    causal = col_raster(np.where(uk <= tq, 255, 0), "mcausal")
    endm = col_raster(np.where(uk == P - 1, 255, 0), "mend")
    hmask = row_raster([HS * h for h in range(4)], "hm")
    dmask = row_raster(ACOLS, "dm")
    Qs = gather_stretch(c, Q, T, P, m, TT, WA)
    Kp = c.merge([K], h=P); Vp = c.merge([V], h=P)
    Kt = c.tile(Kp, 0, 0, WA, TT); Vt = c.tile(Vp, 0, 0, WA, TT)
    if upto == 'qs': return Qs
    pr = prod(c, Qs, Kt, P_["zq"], P_["zk"], 128, P_["Sp"], x=0, y=0, w=WA, h=TT)
    prm = c.arith(pr, dmask, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)                                                       # products only in data cols
    sc = conv_all(c, prm, False, 32, P_["e_sc"], P_["d_sc"], -128 * 32 * P_["e_sc"], P_["zsc"], 0, 0, WA, 0, TT)     # coarse score, valid at head-start cols
    if upto == 'sc': return sc
    scm = c.arith(sc, causal, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)
    ry = (P - 1) // 2; assert P % 2 == 1
    M = c.prim(f'<feMorphology in="{scm}" operator="dilate" radius="0 {ry}" x="0" y="0" width="{WA}" height="{TT}" RESULT/>')    # window max over the block (valid at block-centre rows)
    if upto == 'max': return M
    ctrm = col_raster(np.where(uk == ry, 255, 0), "mctr")
    Mend = c.arith(M, ctrm, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)
    Mb = conv_all(c, Mend, True, P, 1, 1, 0, 0, ry, 0, WA, 0, TT)                                                    # broadcast max over block rows (centred window)
    Mh = c.arith(Mb, hmask, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)
    Msh = c.offset(Mh, 32, 0, x=0, y=0, w=WA + 40, h=TT)                                                              # m pixel -> col 34h+32
    X2 = add_u8(c, prm, Msh, 0, x=0, y=0, w=WA, h=TT)                                                                # [32 products | m | 0]
    e1, e2, d_f, ZF, zsc = P_["e1"], P_["e2"], P_["d_f"], P_["ZF"], P_["zsc"]
    dd = conv_taps(c, X2, [-e1] * 32 + [e2], 255 - e1, d_f, e1 * 128 * 32 - e2 * zsc, ZF, 0, 0, WA, 0, TT)           # fine max-subtracted score (valid at head cols)
    if upto == 'dd': return dd
    e = c.lut(dd, P_["exp"], x=0, y=0, w=WA, h=TT)
    em = c.arith(e, causal, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)
    S8 = conv_all(c, em, True, P, P_["e_S"], P_["d_S"], 0, 0, P - 1, 0, WA, 0, TT)
    R8 = c.lut(S8, P_["rec"], x=0, y=0, w=WA, h=TT)
    Rend = c.arith(R8, endm, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)
    Rb = conv_all(c, Rend, True, P, 1, 1, 0, 0, 0, 0, WA, 0, TT)
    p8 = prod(c, em, Rb, 0, 0, 0, 255, x=0, y=0, w=WA, h=TT)
    if upto == 'p8': return p8
    p8h = c.arith(p8, hmask, 1, 0, 0, 0, x=0, y=0, w=WA, h=TT)
    p8b = conv_all(c, p8h, False, 32, 1, 1, 0, 0, 31, 0, WA, 0, TT)
    pv = prod(c, p8b, Vt, 0, P_["zv"], 128, P_["S_pv"], x=0, y=0, w=WA, h=TT)
    ao = conv_all(c, pv, True, P, P_["e_o"], P_["d_o"], -128 * P * P_["e_o"], P_["zao"], P - 1, 0, WA, 0, TT)
    if upto == 'ao': return ao
    return gather_compact(c, ao, T, P, m, TT, WA)

def attention_ref(Q, K, V, P, T):
    Q = np.asarray(Q, np.int64); K = np.asarray(K, np.int64); V = np.asarray(V, np.int64); out = np.zeros((T, 128), np.int64)
    exp = np.asarray(P["exp"], np.int64); rec = np.asarray(P["rec"], np.int64); e1, e2, d_f = P["e1"], P["e2"], P["d_f"]
    for t in range(T):
        pr = prod_ref(Q[t][None], K[:T], P["zq"], P["zk"], 128, P["Sp"])
        hs = np.stack([pr[:, 32 * h:32 * h + 32].sum(1) for h in range(4)], 1)             # (T,4) raw code sums
        sc = np.clip(np.floor_divide(4 * (P["e_sc"] * hs - 128 * 32 * P["e_sc"]) + 2 * P["d_sc"] + 1, 4 * P["d_sc"]) + P["zsc"], 0, 255)
        valid = (np.arange(T) <= t)[:, None]; scm = np.where(valid, sc, 0)
        M = scm.max(0)
        acc = e2 * M[None] - e1 * hs + (e1 * 128 * 32 - e2 * P["zsc"])
        dd = np.clip(np.floor_divide(4 * acc + 2 * d_f + 1, 4 * d_f) + P["ZF"], 0, 255)
        e = np.where(valid, exp[dd], 0)
        S = e.sum(0); S8 = np.clip(np.floor_divide(4 * P["e_S"] * S + 2 * P["d_S"] + 1, 4 * P["d_S"]), 0, 255); R8 = rec[S8]
        p8 = prod_ref(e, R8[None], 0, 0, 0, 255)
        p8b = np.repeat(p8, 32, axis=1)
        pv = prod_ref(p8b, V[:T], 0, P["zv"], 128, P["S_pv"])
        sm = pv.sum(0)
        out[t] = np.clip(np.floor_divide(4 * (P["e_o"] * sm - 128 * T * P["e_o"]) + 2 * P["d_o"] + 1, 4 * P["d_o"]) + P["zao"], 0, 255)
    return out

# ------------------------------------------------------------------ exact row gathers via feDisplacementMap ----
# Findings (probes p9-p11): dy = K*G - 127.5*K exactly for even K (all 256 codes); odd/half-integer cases (K=1) are TIES whose rounding
# depends on position -> never used.  K=2m coarse + K=2 fine stage (odd displacements only) with a 1-row-shifted second copy for parity.
def pick_pitch(T):
    """K=2m coarse step, pitch P=1+K*ceil((T-1)/K) >= T (2m | P-1), displacement range (P*T+2P <= 255*m)."""
    for m in range(1, 200):
        K = 2 * m; P = 1 + K * (-(-(T - 1) // K))
        if P >= T and P * T + 2 * P <= 255 * m and P <= 250: return m, P
    raise ValueError("T too large for 2-stage gather")

def _tile_map(c, codes, nrows, default, h, tag, wd=128):
    a = np.zeros((nrows, 1, 3), np.uint8); a[:, 0, 0] = 128; a[:, 0, 1] = default
    for r, g in codes.items(): a[r, 0, 1] = g
    nm = f"{tag}_{c.g.n}"; c.image(a, x=0, y=0, res=nm); return c.tile(nm, 0, 0, wd, h)
def _disp(c, src, mp, scale, h, dxc, dy=0, wd=128):
    s = c.offset(src, dxc, dy, x=0, y=0, w=c.W, h=max(h, 1) + 2) if (dxc or dy) else src
    return c.prim(f'<feDisplacementMap in="{s}" in2="{mp}" scale="{scale}" xChannelSelector="R" yChannelSelector="G" x="0" y="0" width="{wd}" height="{h}" RESULT/>')

def gather_stretch(c, src, T, P, m, TT, wd=128):
    """out[r] = src[r // P] for r < T*P (src rows 0..T-1 at cols 0..127, opaque). Two coarse copies (parity) + one fine stage."""
    K = 2 * m
    code = lambda t: ((t - (P * t + m)) + 255 * m) // K
    assert all(0 <= code(t) <= 255 and ((t - (P * t + m)) + 255 * m) % K == 0 for t in range(T))
    mapA = _tile_map(c, {P * t + m: code(t) for t in range(T)}, TT, 0, TT, "gsA", wd)
    mapB = _tile_map(c, {P * t + m + 1: code(t) for t in range(T)}, TT, 0, TT, "gsB", wd)
    midA = _disp(c, src, mapA, 255 * K, TT, m, 0, wd)
    midB = _disp(c, src, mapB, 255 * K, TT, m, 1, wd)
    mid = c.merge([midA, midB], h=TT)
    G2 = np.zeros(TT, np.int64)
    for r in range(TT):
        t, u = divmod(r, P); za = P * t + m; z = za if ((za - r) % 2) else za + 1
        d2 = z - r; assert d2 % 2 != 0 and abs(d2) <= 255; G2[r] = (d2 + 255) // 2
    a = np.zeros((TT, 1, 3), np.uint8); a[:, 0, 0] = 128; a[:, 0, 1] = G2.astype(np.uint8)
    nm = f"gs2_{c.g.n}"; c.image(a, x=0, y=0, res=nm); m2 = c.tile(nm, 0, 0, wd, TT)
    return _disp(c, mid, m2, 510, TT, 1, 0, wd)

def gather_compact(c, src, T, P, m, TT, wd=128):
    """out[t] = src[P*t + P-1] (block-end rows) for t<T -> rows 0..T-1."""
    K = 2 * m; s = 1 if m % 2 == 0 else 0; cp = P - 1 + s
    delta = (cp - m) % K; assert delta % 2 == 1
    code = lambda t: (((P - 1) * t + cp - delta) + 255 * m)
    assert all(code(t) % K == 0 and 0 <= code(t) // K <= 255 for t in range(T)), (T, P, m)
    mp = _tile_map(c, {t + delta: code(t) // K for t in range(T)}, TT + 4, 0, TT + 4, "gc1", wd)
    sh = c.offset(src, 0, s, x=0, y=0, w=c.W, h=TT + 2) if s else src
    mid = _disp(c, sh, mp, 255 * K, TT + 4, m, 0, wd)
    hh = T + delta + 2                      # displacement can only read inside its own output rect
    a = np.zeros((hh, 1, 3), np.uint8); a[:, 0, 0] = 128; a[:, 0, 1] = (delta + 255) // 2
    nm = f"gc2_{c.g.n}"; c.image(a, x=0, y=0, res=nm); m2 = c.tile(nm, 0, 0, wd, hh)
    out = _disp(c, mid, m2, 510, hh, 1, 0, wd)
    return c.merge([out], h=T)
