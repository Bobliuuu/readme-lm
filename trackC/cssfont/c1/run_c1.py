import numpy as np, subprocess, sys, zlib, struct
from PIL import Image
W = np.load("W.npy"); rng = np.random.default_rng(1)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ok_all = True
for trial in range(3):
    v = rng.integers(-8, 8, 8)
    fvs = ", ".join(f'"a{j:03d}" {v[j]/8}' for j in range(8))
    rows = "\n".join(f'<div class=r>{chr(65+i)}</div>' for i in range(8))
    html = f"""<!doctype html><meta charset=utf-8><style>
@font-face{{font-family:mm;src:url(mm8.ttf)}}
html,body{{margin:0;background:#fff}}
.r{{font-family:mm;font-size:1000px;line-height:1;height:6px;overflow:hidden;width:max-content;background:#f00;margin-bottom:4px;font-variation-settings:{fvs}}}
</style>{rows}"""
    open("t.html", "w").write(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=2400,200", "--screenshot=out.png", "file:///Users/jhome/readme-lm/trackC/cssfont/c1/t.html"], capture_output=True, timeout=60)
    im = np.array(Image.open("out.png").convert("RGB")).astype(int)
    meas = []
    for i in range(8):
        y = i * 10 + 3
        red = (im[y, :, 0] > 200) & (im[y, :, 1] < 60)
        meas.append(int(red.sum()))
    exp = 1100 + 16 * (W @ v) / 8 * 1  # delta=W*16 units * x=v/8
    exp = exp.astype(float)
    err = np.abs(np.array(meas) - exp)
    dec = [(m - 1100) / 2 for m in meas]
    print("v", v.tolist()); print("meas", meas); print("exp ", exp.astype(int).tolist()); print("W@v", (W @ v).tolist(), "decoded", dec, "maxerr px", err.max())
    ok_all &= bool((np.round(dec) == W @ v).all())
print("C1", "PASS" if ok_all else "FAIL")
