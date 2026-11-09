"""verify: HVAR region over TWO axes multiplies them (q*k) -> attention score / p*v product in the font engine"""
import sys, numpy as np, subprocess
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables._f_v_a_r import Axis
from PIL import Image
D = 4
names = [".notdef", "g"]; fb = FontBuilder(1000, isTTF=True); fb.setupGlyphOrder(names); fb.setupCharacterMap({0x41: "g"})
e = TTGlyphPen(None).glyph(); fb.setupGlyf({n: e for n in names}); fb.setupHorizontalMetrics({n: (1500, 0) for n in names}); fb.setupHorizontalHeader(ascent=800, descent=-200)
fb.setupNameTable({"familyName": "p", "styleName": "R"}); fb.setupOS2(); fb.setupPost()
fv = newTable("fvar"); fv.axes = []; fv.instances = []
tags = [f"q{d:03d}" for d in range(D)] + [f"k{d:03d}" for d in range(D)]
for t in tags:
    a = Axis(); a.axisTag = t; a.minValue, a.defaultValue, a.maxValue = -1, 0, 1; a.axisNameID = 256; fv.axes.append(a)
fb.font["fvar"] = fv
# for each d, 4 quadrant regions so q*k is exact for both signs: (q+,k+),(q-,k-) -> +; (q+,k-),(q-,k+) -> -  [scalar = |q||k| in each quadrant]
rl = ot.VarRegionList(); rl.RegionAxisCount = len(tags); rl.Region = []; deltas = []
for d in range(D):
    for qs, ks in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        r = ot.VarRegion(); r.VarRegionAxis = []
        for ti in range(len(tags)):
            ra = ot.VarRegionAxis(); ra.StartCoord = ra.PeakCoord = ra.EndCoord = 0
            if ti == d: ra.StartCoord, ra.PeakCoord, ra.EndCoord = (0, 1, 1) if qs > 0 else (-1, -1, 0)
            if ti == D + d: ra.StartCoord, ra.PeakCoord, ra.EndCoord = (0, 1, 1) if ks > 0 else (-1, -1, 0)
            r.VarRegionAxis.append(ra)
        rl.Region.append(r); deltas.append(1000 * qs * ks)
rl.RegionCount = len(rl.Region)
vd = ot.VarData(); vd.VarRegionCount = rl.RegionCount; vd.VarRegionIndex = list(range(rl.RegionCount)); vd.Item = [deltas]; vd.ItemCount = 1; vd.NumShorts = rl.RegionCount
st = ot.VarStore(); st.Format = 1; st.VarRegionList = rl; st.VarData = [vd]; st.VarDataCount = 1
hv = newTable("HVAR"); h = ot.HVAR(); h.Version = 0x00010000; h.VarStore = st
m = ot.VarIdxMap(); m.Format = 0; m.mapping = {".notdef": 0, "g": 0}; h.AdvWidthMap = m; h.LsbMap = None; h.RsbMap = None; hv.table = h; fb.font["HVAR"] = hv
gv = newTable("gvar"); gv.version = 1; gv.reserved = 0; gv.variations = {n: [] for n in names}; fb.font["gvar"] = gv
fb.save("prod.ttf")
rng = np.random.default_rng(2); rows = ""; exp = []
for trial in range(6):
    q = rng.integers(-8, 9, D) / 8; k = rng.integers(-8, 9, D) / 8
    fvs = ",".join([f"'q{d:03d}' {q[d]}" for d in range(D)] + [f"'k{d:03d}' {k[d]}" for d in range(D)])
    rows += f'<div class=r style="font-variation-settings:{fvs}">A</div>'; exp.append(1500 + 1000 * float(q @ k))
html = f"<!doctype html><style>@font-face{{font-family:p;src:url(prod.ttf)}}html,body{{margin:0}}.r{{font:1000px p;line-height:1;height:6px;overflow:hidden;width:max-content;background:#f00;margin-bottom:4px}}</style>{rows}"
open("prod.html", "w").write(html)
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=4000,80", "--screenshot=prod.png", "file:///Users/jhome/readme-lm/trackC/cssfont/c4/prod.html"], capture_output=True, timeout=60)
im = np.array(Image.open("prod.png").convert("RGB")).astype(int)
meas = [((255 - im[i * 10 + 3, :, 1]) / 255).sum() for i in range(6)]
print("meas", [round(x, 1) for x in meas]); print("exp ", [round(x, 1) for x in exp]); print("PRODUCT-REGION", "PASS" if np.abs(np.array(meas) - exp).max() < 2 else "FAIL")
