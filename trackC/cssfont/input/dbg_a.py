"""Tokenizer font for a char vocab: GSUB tags the k-th character as glyph P_k_c (chained contextual lookups anchored on a
start marker U+E000), HVAR axis POS (0..1) selects position k (triangle region peaked at k/N) -> advance = id(c)+1 for
the selected position, 0 for all others. So width of the text = token id (+1) of char number POS*N."""
import sys, numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables._f_v_a_r import Axis
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
VOC = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"()-_/=+*#<>[]!?\n"
def gname(c): return "c%02x" % ord(c)
def build(path, N=40, mode="tok"):
    chars = [c for c in VOC if c != "\n"]; lc = [c for c in chars if not c.isupper()]  # base glyphs: lowercase set
    ids = {c: VOC.index(c) for c in lc}
    base = [gname(c) for c in lc]; start = "start"
    pg = lambda k, c: f"p{k}_{ord(c):02x}"
    order = [".notdef", start] + base + ([pg(k, c) for k in range(1, N + 1) for c in lc] if mode in ("tok","dbg") else [])
    fb = FontBuilder(1000, isTTF=True); fb.setupGlyphOrder(order)
    cmap = {0xE000: start}
    for c in chars: cmap[ord(c)] = gname(c.lower())
    cmap[0xA0] = gname(" "); cmap[0x0A] = gname(" ")
    fb.setupCharacterMap(cmap); e = TTGlyphPen(None).glyph(); fb.setupGlyf({n: e for n in order})
    adv = {n: 0 for n in order}
    if mode == "dbg":
        for n in order:
            if n.startswith("p"): adv[n]=3
    if mode == "len":
        for n in base: adv[n] = 1
    fb.setupHorizontalMetrics({n: (adv[n], 0) for n in order}); fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "tok", "styleName": "R"}); fb.setupOS2(); fb.setupPost()
    if mode in ("tok","dbg"):
        fea = "languagesystem DFLT dflt;\n@Base = [" + " ".join(base) + "];\n"
        for k in range(1, N + 1):
            fea += f"@P{k} = [" + " ".join(pg(k, c) for c in lc) + "];\n"
        for k in range(1, N + 1):
            fea += f"\n"
        fea += "feature calt {\n"
        fea += "  sub @Base by @P1;\n"
        pass
        fea += "} calt;\n"
        # ensure same-feature lookups all fire: feaLib lays them out in order inside one feature
        addOpenTypeFeaturesFromString(fb.font, fea)
        fv = newTable("fvar"); fv.axes = []; fv.instances = []
        a = Axis(); a.axisTag = "POSN"; a.minValue, a.defaultValue, a.maxValue = 0, 0, 1; a.axisNameID = 256; fv.axes.append(a); fb.font["fvar"] = fv
        rl = ot.VarRegionList(); rl.RegionAxisCount = 1; rl.Region = []
        for k in range(1, N + 1):
            r = ot.VarRegion(); ra = ot.VarRegionAxis(); ra.StartCoord = (k - 1) / N; ra.PeakCoord = k / N; ra.EndCoord = min(1.0, (k + 1) / N) if k < N else 1.0
            r.VarRegionAxis = [ra]; rl.Region.append(r)
        rl.RegionCount = N
        st = ot.VarStore(); st.Format = 1; st.VarRegionList = rl; st.VarData = []
        mapping = {".notdef": 0, start: 0}
        # VarData k-1 holds one region (k); item j = lowercase char j. Items stored per (outer=k-1, inner=j)
        for k in range(1, N + 1):
            vd = ot.VarData(); vd.VarRegionCount = 1; vd.VarRegionIndex = [k - 1]; vd.Item = [[ids[c] + 1] for c in lc]; vd.ItemCount = len(lc); vd.NumShorts = 1; st.VarData.append(vd)
            for j, c in enumerate(lc): mapping[pg(k, c)] = ((k - 1) << 16) | j
        for n in base: mapping[n] = 0
        # glyphs that are not P glyphs must map somewhere harmless: outer0 inner0 gets delta for 'space' at pos1 only on region1... use dummy VarData
        dummy = ot.VarData(); dummy.VarRegionCount = 1; dummy.VarRegionIndex = [0]; dummy.Item = [[0]]; dummy.ItemCount = 1; dummy.NumShorts = 1; st.VarData.append(dummy); st.VarDataCount = len(st.VarData)
        for n in [".notdef", start] + base: mapping[n] = (N << 16) | 0
        hv = newTable("HVAR"); h = ot.HVAR(); h.Version = 0x00010000; h.VarStore = st
        m = ot.VarIdxMap(); m.Format = 0; m.mapping = mapping; h.AdvWidthMap = m; h.LsbMap = None; h.RsbMap = None; hv.table = h; fb.font["HVAR"] = hv
        gv = newTable("gvar"); gv.version = 1; gv.reserved = 0; gv.variations = {n: [] for n in order}; fb.font["gvar"] = gv
    fb.save(path)
if __name__ == "__main__":
    build("tok.ttf", 40, "tok"); build("len.ttf", 40, "len")
    import os; print(os.path.getsize("tok.ttf"), os.path.getsize("len.ttf"))
