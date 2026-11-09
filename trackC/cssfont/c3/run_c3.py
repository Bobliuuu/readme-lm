import sys, numpy as np, subprocess
sys.path.insert(0, "../c1")
from mkfont import build
from PIL import Image
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
rng = np.random.default_rng(5)
W1 = rng.integers(-8, 8, (8, 4)); W2 = rng.integers(-8, 8, (4, 8))
build(W1, "f1.ttf"); build(W2, "f2.ttf")
P = 2400; BASE = 1100; SC = 64
allok = True
for trial in range(3):
    v = rng.integers(-8, 8, 4)
    h = W1 @ v; a = np.clip(np.maximum(h, 0) / SC, 0, 1)
    y_exp = 16 * (W2 @ a)           # width delta px
    fv1 = ", ".join(f'"a{j:03d}" {v[j]/8}' for j in range(4))
    tl = ", ".join(f"--t{i}" for i in range(8))
    css = f"""
@font-face{{font-family:f1;src:url(f1.ttf)}} @font-face{{font-family:f2;src:url(f2.ttf)}}
""" + "".join(f"@property --p{i}{{syntax:'<number>';inherits:true;initial-value:0}}\n@keyframes rd{i}{{from{{--p{i}:0}}to{{--p{i}:1}}}}\n" for i in range(8)) + f"""
html,body{{margin:0;background:#fff}}
body{{timeline-scope:{tl}}}
#sc{{overflow-x:scroll;overflow-y:hidden;width:{P}px;height:100px;scrollbar-width:none;position:relative}}
.row{{height:10px;white-space:nowrap;font-size:0}}
.row .A{{display:inline-block;font-family:f1;font-size:1000px;line-height:0;height:0;font-variation-settings:{fv1};vertical-align:top}}
.row .M{{display:inline-block;width:1px;height:10px;vertical-align:top}}
#cons{{animation-name:{", ".join(f"rd{i}" for i in range(8))};animation-duration:1s;animation-timing-function:linear;animation-fill-mode:both;animation-timeline:{tl};
 {"".join(f"--h{i}:calc(({P+1} * (1 - var(--p{i})) - {BASE}) / 2); --a{i}:clamp(0, max(0, var(--h{i})) / {SC}, 1);" for i in range(8))}
 --fv:{", ".join(f'"a{j:03d}" var(--a{j})' for j in range(8))}}}
.o{{font-family:f2;font-size:1000px;line-height:1;height:6px;overflow:hidden;width:max-content;background:#f00;margin-bottom:4px;font-variation-settings:var(--fv)}}
"""
    rows = "".join(f'<div class=row><span class=A>{chr(65+i)}</span><span class=M style="view-timeline:--t{i} x;background:#00f"></span></div>' for i in range(8))
    outs = "".join(f'<div class=o>{chr(65+i)}</div>' for i in range(4))
    open("t.html", "w").write(f"<!doctype html><meta charset=utf-8><style>{css}</style><div id=sc>{rows}</div><div id=cons>{outs}</div>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=2600,300", "--screenshot=out.png", "file:///Users/jhome/readme-lm/trackC/cssfont/c3/t.html"], capture_output=True, timeout=60)
    im = np.array(Image.open("out.png").convert("RGB")).astype(float)
    # layer1 marker positions (blue) at rows y=5+10*i
    w1 = []
    for i in range(8):
        r = im[5 + 10*i]; blue = np.where((r[:,2] > 200) & (r[:,0] < 60))[0]; w1.append(int(blue[0]) if len(blue) else -1)
    base_y = 100
    meas = []
    for i in range(4):
        r = im[base_y + i*10 + 3]; meas.append(((255 - r[:,1])/255).sum())
    print("v", v.tolist(), "h exp", h.tolist(), "h dec(L1 readout)", [(w - BASE)/2 for w in w1])
    print("L2 width exp", (BASE + y_exp).round(1).tolist(), "meas", [round(m,1) for m in meas])
    ok = np.abs(np.array(meas) - (BASE + y_exp)).max() < 3
    print("maxerr px", np.abs(np.array(meas) - (BASE + y_exp)).max(), ok); allok &= bool(ok)
print("C3", "PASS" if allok else "FAIL")
