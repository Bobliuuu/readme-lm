import sys, numpy as np, subprocess, time
sys.path.insert(0, ".")
from fontsize import build1
from PIL import Image
CH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
rng = np.random.default_rng(3)
for C in (256, 257):
    R = 128; W = rng.integers(-8, 8, (R, C)); build1(W, f"ax{C}.ttf")
    a = rng.integers(0, 17, C) / 16
    fvs = ",".join(f'"a{j:03d}" {a[j]}' for j in range(C))
    rows = "".join(f'<div class=r>{chr(0x4E00+i)}</div>' for i in range(R))
    html = f"""<!doctype html><meta charset=utf-8><style>@font-face{{font-family:m;src:url(ax{C}.ttf)}}html,body{{margin:0}}
.r{{font-family:m;font-size:1000px;line-height:1;height:1px;overflow:hidden;width:max-content;background:#f00;font-variation-settings:{fvs}}}</style>{rows}"""
    open("ax.html", "w").write(html)
    t = time.time()
    subprocess.run([CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=40000,140", "--screenshot=ax.png", "file:///Users/jhome/readme-lm/trackC/cssfont/c4/ax.html"], capture_output=True, timeout=120)
    im = np.array(Image.open("ax.png").convert("RGB")).astype(int)
    meas = np.array([((255 - im[i, :, 1]) / 255).sum() for i in range(R)])
    exp = 30000 + 16 * 0 + W @ a * 1  # delta int units * axis
    print(C, "axes: screenshot width", im.shape, "wall s", round(time.time() - t, 2), "mean abs err px", float(np.abs(meas - exp).mean()) if meas.max() > 0 else "no render", "meas0", meas[:3], "exp0", exp[:3])
