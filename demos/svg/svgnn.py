"""SVG-filter neural-net builder (no JavaScript).  Every multiply-accumulate is an feConvolveMatrix, every nonlinearity /
lookup an feComponentTransfer, argmax is feMorphology, glyph drawing is feDisplacementMap.  Ops verified bit-exact in
Chrome in trackD/svgops (copied here so the generator is self-contained: numpy only).
Conventions: tensors are uint8 codes (gray, R=G=B, A=255); filter-region rows are windows (each window repeated RH rows so the
picture is RH px tall), columns are channels."""
import zlib, struct, base64, numpy as np

def png_uri(a):
    a = np.asarray(a, np.uint8)
    if a.ndim == 2: a = np.stack([a, a, a], -1)
    if a.shape[2] == 3: a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    def ch(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 9)) + ch(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()

def f12(x): return f"{x:.12g}"

class Ctx:
    def __init__(s, W, H): s.W, s.H, s.rows = W, H, H; s.p = []; s.n = 0; s.bg = None; s.cache = {}
    def prim(s, xml):
        s.n += 1; res = f"r{s.n}"; s.p.append(xml.replace("RESULT", f'result="{res}"')); return res
    def bgf(s):
        if s.bg is None: s.bg = s.prim(f'<feFlood flood-color="#000" flood-opacity="1" x="0" y="0" width="{s.W}" height="{s.H}" RESULT/>')
        return s.bg
    def image(s, arr, x=0, y=0):
        h, w = np.asarray(arr).shape[:2]
        return s.prim(f'<feImage href="{png_uri(arr)}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" RESULT/>')
    def merge(s, names):
        return s.prim("<feMerge RESULT>" + "".join(f'<feMergeNode in="{n}"/>' for n in [s.bgf()] + list(names)) + "</feMerge>")
    def merge_at(s, names, x, y, w, h, color="#000"):   # merge on a flood (default black) confined to a sub-rectangle
        fl = s.prim(f'<feFlood flood-color="{color}" flood-opacity="1" x="{x}" y="{y}" width="{w}" height="{h}" RESULT/>')
        return s.prim(f'<feMerge x="{x}" y="{y}" width="{w}" height="{h}" RESULT>' + "".join(f'<feMergeNode in="{n}"/>' for n in [fl] + list(names)) + "</feMerge>")
    def offset(s, src, dx, dy=0, x=None, y=None, w=None, h=None):
        if dx == 0 and dy == 0 and x is None: return src
        key = (src, dx, dy, x, y, w, h)
        if key in s.cache: return s.cache[key]
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        r = s.prim(f'<feOffset in="{src}" dx="{dx}" dy="{dy}"{sub} RESULT/>'); s.cache[key] = r; return r
    def tile(s, src, x, y, w, h): return s.prim(f'<feTile in="{src}" x="{x}" y="{y}" width="{w}" height="{h}" RESULT/>')
    def lut(s, src, table, x=None, y=None, w=None, h=None, chans="RGB"):
        t = " ".join(f"{(int(v) + 0.5) / 255:.5f}" for v in table); f = f'type="discrete" tableValues="{t}"'
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        return s.prim(f'<feComponentTransfer in="{src}"{sub} RESULT>' + "".join(f"<feFunc{c} {f}/>" for c in chans) + "</feComponentTransfer>")
    def lut_rgb(s, src, tr, tg, tb, x=None, y=None, w=None, h=None):
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        fn = lambda c, t: f'<feFunc{c} type="discrete" tableValues="{" ".join(f"{(int(v)+0.5)/255:.5f}" for v in t)}"/>'
        return s.prim(f'<feComponentTransfer in="{src}"{sub} RESULT>' + fn("R", tr) + fn("G", tg) + fn("B", tb) + "</feComponentTransfer>")
    def cmatrix(s, src, rows, x=None, y=None, w=None, h=None):
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        return s.prim(f'<feColorMatrix in="{src}" type="matrix" values="{" ".join(f12(v) for r in rows for v in r)}"{sub} RESULT/>')
    def arith(s, a, b, k1, k2, k3, k4, x=None, y=None, w=None, h=None):
        assert k1 + k2 + k3 + k4 >= 1 - 1e-9
        sub = "".join(f' {k}="{v}"' for k, v in (("x", x), ("y", y), ("width", w), ("height", h)) if v is not None)
        return s.prim(f'<feComposite in="{a}" in2="{b}" operator="arithmetic" k1="{f12(k1)}" k2="{f12(k2)}" k3="{f12(k3)}" k4="{f12(k4)}"{sub} RESULT/>')

def add_u8(c, a, b, off=0, **sub): return c.arith(a, b, 0, 1, 1, off / 255.0, **sub)
def neg_u8(c, a, **sub): return c.cmatrix(a, [[-1, 0, 0, 0, 1], [-1, 0, 0, 0, 1], [-1, 0, 0, 0, 1], [0, 0, 0, 1, 0]], **sub)

def conv_neuron(c, src, n, x_out, win_start, taps, d, Bint, zp):
    """one output neuron: taps (len n) over src cols [win_start, win_start+n), guard col win_start+n must be 0.
    out = clip(floor((4*(S+Bint)+2d+1)/(4d)) + zp) (round half up, tie-free via the +0.25 nudge); Skia needs kernel range | 255 -> guard tap."""
    taps = np.asarray(taps, np.int64); g = int(taps.min()) + 255
    assert int(taps.max()) <= g and abs(int(taps.min())) <= 255
    t = x_out - win_start; assert 0 <= t <= n
    K = np.concatenate([taps, [g]])[::-1]
    bias = ((Bint + 0.25) / d + zp) / 255.0
    return c.prim(f'<feConvolveMatrix in="{src}" order="{n+1} 1" kernelMatrix="{" ".join(map(str, K.tolist()))}" divisor="{d}" bias="{f12(bias)}" '
                  f'targetX="{t}" targetY="0" edgeMode="none" preserveAlpha="true" x="{x_out}" y="0" width="1" height="{c.rows}" RESULT/>')

def linear(c, src, c0, n, E, d, zin, zout, bacc, out_cols):
    """E (n_out, n) int kernel on src cols [c0, c0+n) (+ zero guard col).  Outputs at out_cols (consecutive, within n of the first)."""
    n_out = E.shape[0]; x0 = out_cols[0]; assert out_cols[-1] - x0 <= n and c.W >= x0 + n + 1
    sh = c.offset(src, x0 - c0); outs = []
    for j in range(n_out):
        outs.append(conv_neuron(c, sh, n, out_cols[j], x0, E[j], d, int(bacc[j]) - zin * int(E[j].sum()), zout))
    return c.merge(outs)
def linear_ref(U, E, d, zin, zout, bacc):
    S = (np.asarray(E, np.int64)[None] * (np.asarray(U, np.int64)[:, None, :] - zin)).sum(-1) + np.asarray(bacc, np.int64)[None]
    return np.clip(np.floor_divide(4 * S + 2 * d + 1, 4 * d) + zout, 0, 255)

def argmax_row(c, L, n):
    """exact first-index argmax over logit codes in R at cols [0,n): row max by dilate, equality by LUT, min index by erode; id lands at col n-1."""
    H = c.rows
    ramp = np.tile(np.arange(n, dtype=np.uint8)[None], (H, 1)); rampi = c.image(ramp, x=0, y=0)
    m = c.prim(f'<feMorphology in="{L}" operator="dilate" radius="{n-1} 0" x="0" y="0" width="{n}" height="{H}" RESULT/>')
    nL = neg_u8(c, L, x=0, y=0, w=n, h=H)
    d = add_u8(c, m, nL, -255, x=0, y=0, w=n, h=H)
    teq = [255 if v == 0 else 0 for v in range(256)]; tne = [0 if v == 0 else 255 for v in range(256)]
    eq = c.lut(d, teq, x=0, y=0, w=n, h=H); ne = c.lut(d, tne, x=0, y=0, w=n, h=H)
    Ie = c.arith(rampi, eq, 1, 0, 0, 0, x=0, y=0, w=n, h=H)
    J = add_u8(c, Ie, ne, 0, x=0, y=0, w=n, h=H)
    wb = c.prim(f'<feFlood flood-color="#fff" flood-opacity="1" x="0" y="0" width="{2*n-1}" height="{H}" RESULT/>')
    Jw = c.prim(f'<feMerge x="0" y="0" width="{2*n-1}" height="{H}" RESULT><feMergeNode in="{wb}"/><feMergeNode in="{J}"/></feMerge>')
    return c.prim(f'<feMorphology in="{Jw}" operator="erode" radius="{n-1} 0" x="0" y="0" width="{2*n-1}" height="{H}" RESULT/>')

# ---------------------------------------------------------------- glyphs / heat colours
GW, GH = 16, 24
DSTEP = 4          # displacement step per map code: scale = 255*DSTEP -> dx = DSTEP*(code-127.5): small scale keeps Skia's input rect small
AX0 = 10           # atlas x offset: dx = 16*id + AX0 - X must be = DSTEP*code - 510 (+even)
def atlas_x(X): return AX0 if X <= 500 else X - 506          # per-column atlas offset keeps dx = 16*id + Ax - X inside +-510
def gcode(idv, X): return 4 * idv + (atlas_x(X) + 510 - X) // 4   # map R code that gathers glyph `idv` into a cell at x=X (0..255 asserted below)
def make_atlas(voc, fsize=18):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=fsize); A = Image.new("L", (GW * len(voc), GH), 0); d = ImageDraw.Draw(A)
    for i, ch in enumerate(voc):
        if ch == " ": continue
        bb = d.textbbox((0, 0), ch, font=font); w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text((i * GW + (GW - w) // 2 - bb[0], (GH - h) // 2 - bb[1]), ch, font=font, fill=255)
    return np.array(A, np.uint8)

def heat_tables(gain=1):
    """256-entry colour ramp (black -> purple -> orange -> pale yellow), distinct colour per code (asserted)."""
    stops = [(0, (0, 0, 0)), (85, (70, 0, 120)), (170, (235, 90, 20)), (255, (255, 245, 140))]
    t = np.zeros((256, 3))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        for u in range(a, b + 1): t[u] = np.array(ca) + (np.array(cb) - np.array(ca)) * (u - a) / (b - a)
    t = np.round(t).astype(np.int64); t = t[np.minimum(np.arange(256) * gain, 255)]; return t[:, 0], t[:, 1], t[:, 2]
def heat_decode_table():
    r, g, b = heat_tables(); return {(int(r[u]), int(g[u]), int(b[u])): u for u in range(256)}

def glyph_cells(c, atlas_tile, mapimg, x, w, y=0, h=None):
    """draw glyphs: atlas_tile is the (vertically tiled) atlas result, placed at Ax=8; mapimg: R = id + 128 - X/16 per cell, G=128."""
    return c.prim(f'<feDisplacementMap in="{atlas_tile}" in2="{mapimg}" scale="{255*DSTEP}" xChannelSelector="R" yChannelSelector="G" x="{x}" y="{y}" width="{w}" height="{c.rows if h is None else h}" RESULT/>')

def atlas_node(c, atlas, ax=None, tag=""):
    """vertically tiled atlas at x=ax (cached per ax): tile row t holds glyph row (t-2) because displacement dy=+2 (G code 128)."""
    ax = AX0 if ax is None else ax; key = ("atlas", ax, tag)
    if key not in c.cache:
        img = c.image(np.roll(atlas, 2, axis=0), x=ax, y=0); c.cache[key] = c.tile(img, ax, 0, atlas.shape[1], c.rows)
    return c.cache[key]

def svg_wrap(c, body, vw, vh, hdr, texts, title, w_total=None):
    """assemble: filter graph body, filtered rect shifted down by hdr (integer translate), static text overlay."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{vw}" height="{vh}" viewBox="0 0 {vw} {vh}">'
            f'<title>{title}</title><defs><filter id="F" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" x="0" y="0" width="{c.W}" height="{c.H}" color-interpolation-filters="sRGB">{body}</filter></defs>'
            f'<rect width="{vw}" height="{vh}" fill="#0b0d12"/><g transform="translate(0,{hdr})"><rect width="{c.W}" height="{c.H}" fill="#000" filter="url(#F)"/></g>{texts}</svg>')
