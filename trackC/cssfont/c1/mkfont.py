"""Generate a variable font: glyph i = row i of integer matrix W, axis j = column j.
advance(glyph i) = BASE + sum_j W[i,j]*K * x_j  (x_j axis coord in [-1,1]) via HVAR."""
import sys, numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib import builder as vb

def build(W, path, K=16, BASE=1100, upem=1000, tags=None):
    R, C = W.shape
    tags = tags or [f"a{j:03d}" for j in range(C)]
    names = [".notdef"] + [f"r{i}" for i in range(R)]
    fb = FontBuilder(upem, isTTF=True)
    # one char per row: U+0041.. 'A'+i
    cmap = {0x41 + i: f"r{i}" for i in range(R)}
    fb.setupGlyphOrder(names); fb.setupCharacterMap(cmap)
    empty = TTGlyphPen(None).glyph()
    fb.setupGlyf({n: empty for n in names})
    fb.setupHorizontalMetrics({n: (BASE, 0) for n in names})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "mm", "styleName": "Regular"})
    fb.setupOS2(); fb.setupPost()
    fb.font["fvar"] = fv = newTable("fvar"); fv.axes = []; fv.instances = []
    from fontTools.ttLib.tables._f_v_a_r import Axis
    for t in tags:
        a = Axis(); a.axisTag = t; a.minValue, a.defaultValue, a.maxValue = -1, 0, 1; a.axisNameID = 256
        fv.axes.append(a)
    # regions: per axis a pos region (0,1,1) and neg region (-1,-1,0)
    regs = []
    for j, t in enumerate(tags):
        regs.append(vb.buildVarRegion({t: (0, 1, 1)}, tags))
        regs.append(vb.buildVarRegion({t: (-1, -1, 0)}, tags))
    rl = vb.buildVarRegionList([r.get_support() if hasattr(r,'get_support') else r for r in []], tags) if False else None
    # build region list directly
    rlist = ot.VarRegionList(); rlist.RegionAxisCount = len(tags); rlist.Region = []
    for j in range(C):
        for lo, pk, hi in ((0, 1, 1), (-1, -1, 0)):
            reg = ot.VarRegion(); reg.VarRegionAxis = []
            for k in range(C):
                ra = ot.VarRegionAxis()
                ra.StartCoord, ra.PeakCoord, ra.EndCoord = (lo, pk, hi) if k == j else (0, 0, 0)
                reg.VarRegionAxis.append(ra)
            rlist.Region.append(reg)
    rlist.RegionCount = len(rlist.Region)
    data = ot.VarData(); data.VarRegionCount = rlist.RegionCount; data.VarRegionIndex = list(range(rlist.RegionCount))
    data.Item = []
    for i in range(R):
        row = []
        for j in range(C):
            d = int(W[i, j]) * K
            row += [d, -d]
        data.Item.append(row)
    data.ItemCount = len(data.Item); data.NumShorts = rlist.RegionCount
    store = ot.VarStore(); store.Format = 1; store.VarRegionList = rlist; store.VarData = [data]; store.VarDataCount = 1
    hvar = newTable("HVAR"); h = ot.HVAR(); h.Version = 0x00010000; h.VarStore = store
    # map glyph i (gid i+1) -> (outer 0, inner i); .notdef -> row 0 (unused)
    m = ot.VarIdxMap(); m.Format = 0; m.mapping = {}  # set below
    idx = {".notdef": 0}
    for i in range(R): idx[f"r{i}"] = i
    m.mapping = {n: (0 << 16) | v for n, v in idx.items()}
    h.AdvWidthMap = m; h.LsbMap = None; h.RsbMap = None
    hvar.table = h; fb.font["HVAR"] = hvar
    # empty gvar so FreeType treats as variable
    gv = newTable("gvar"); gv.version = 1; gv.reserved = 0; gv.variations = {n: [] for n in names}; fb.font["gvar"] = gv
    fb.save(path)
    return tags

if __name__ == "__main__":
    rng = np.random.default_rng(0); W = rng.integers(-8, 8, (8, 8)); np.save("W.npy", W)
    build(W, "mm8.ttf")
