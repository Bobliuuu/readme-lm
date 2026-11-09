import sys, numpy as np, os, io
sys.path.insert(0, "/Users/jhome/readme-lm/trackC/cssfont/c1"); sys.path.insert(0, ".")
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable, TTFont
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables._f_v_a_r import Axis
from sim import load

def build1(W, path, base=None, upem=1000):
    """one-sided axes [0,1]; row i -> glyph; delta int16 per (row, axis). W int array (R x C)."""
    R, C = W.shape; names = [".notdef"] + [f"r{i}" for i in range(R)]
    fb = FontBuilder(upem, isTTF=True)
    if R > 0xFFF0: raise ValueError
    cps = [0x4E00 + i for i in range(R)]  # CJK range, plenty
    fb.setupGlyphOrder(names); fb.setupCharacterMap({cp: f"r{i}" for i, cp in enumerate(cps)})
    e = TTGlyphPen(None).glyph(); fb.setupGlyf({n: e for n in names})
    fb.setupHorizontalMetrics({n: (30000, 0) for n in names}); fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "m", "styleName": "R"}); fb.setupOS2(); fb.setupPost()
    fb.font["fvar"] = fv = newTable("fvar"); fv.axes = []; fv.instances = []
    for j in range(C):
        a = Axis(); a.axisTag = f"a{j:03d}"; a.minValue, a.defaultValue, a.maxValue = 0, 0, 1; a.axisNameID = 256; fv.axes.append(a)
    rl = ot.VarRegionList(); rl.RegionAxisCount = C; rl.Region = []
    for j in range(C):
        reg = ot.VarRegion(); reg.VarRegionAxis = []
        for k in range(C):
            ra = ot.VarRegionAxis(); ra.StartCoord, ra.PeakCoord, ra.EndCoord = (0, 1, 1) if k == j else (0, 0, 0); reg.VarRegionAxis.append(ra)
        rl.Region.append(reg)
    rl.RegionCount = C
    vd = ot.VarData(); vd.VarRegionCount = C; vd.VarRegionIndex = list(range(C)); vd.Item = [list(map(int, W[i])) for i in range(R)]; vd.ItemCount = R; vd.NumShorts = C
    st = ot.VarStore(); st.Format = 1; st.VarRegionList = rl; st.VarData = [vd]; st.VarDataCount = 1
    hv = newTable("HVAR"); h = ot.HVAR(); h.Version = 0x00010000; h.VarStore = st
    m = ot.VarIdxMap(); m.Format = 0; m.mapping = {".notdef": 0, **{f"r{i}": i for i in range(R)}}; h.AdvWidthMap = m; h.LsbMap = None; h.RsbMap = None
    hv.table = h; fb.font["HVAR"] = hv
    gv = newTable("gvar"); gv.version = 1; gv.reserved = 0; gv.variations = {n: [] for n in names}; fb.font["gvar"] = gv
    fb.save(path)
    f = TTFont(path); f.flavor = "woff2"; f.save(path + ".woff2")
    return os.path.getsize(path), os.path.getsize(path + ".woff2")

if __name__ == "__main__":
    m = load(); os.makedirs("fonts", exist_ok=True); WB = 9; qmax = 2 ** (WB - 1) - 1
    def qi(W): return np.round(W / (np.abs(W).max(1, keepdims=True) / qmax)).astype(int)
    rows = []; tot = [0, 0]; nf = 0
    def add(name, W):
        global nf
        a, b = build1(W, f"fonts/{name}.ttf"); rows.append((name, W.shape, a, b)); tot[0] += a; tot[1] += b; nf += 1
    for l, B in enumerate(m["blocks"]):
        add(f"qkv{l}", qi(B["qkv"])); add(f"o{l}", qi(B["o"])); add(f"u{l}", qi(B["u"]))
        dn = qi(B["dn"]); add(f"dn{l}a", dn[:, :256]); add(f"dn{l}b", dn[:, 256:])
    E = qi(m["emb"]); add("head", E); add("embed_onehot", qi(m["emb"].T.copy()))
    print(f"{'font':14s} {'rows x axes':>12s} {'ttf B':>9s} {'woff2 B':>9s}")
    for n, s, a, b in rows: print(f"{n:14s} {str(s):>12s} {a:9d} {b:9d}")
    print("fonts:", nf, "total ttf", tot[0], "woff2", tot[1], "weights", sum(r[1][0]*r[1][1] for r in rows))
