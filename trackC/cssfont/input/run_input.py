import numpy as np, subprocess
from PIL import Image
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VOC = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"()-_/=+*#<>[]!?\n"
N = 40
prompt = "What  is\nQwenTPU?"    # double space + newline + uppercase: CSS whitespace collapsing should normalise
import re
norm = re.sub(r"\s+", " ", prompt.lower())  # what the model's pre-processing does
ids = [VOC.index(c) for c in norm]
rows = ""
for t in range(1, len(ids) + 3):
    rows += f'<div class=row><style contenteditable class=T style="font-variation-settings:\'POS \' {t/N}">{prompt}</style><span class=M></span></div>'
rows += f'<div class=row><style contenteditable class=L>{prompt}</style><span class=M></span></div>'
html = f"""<!doctype html><meta charset=utf-8><style>
@font-face{{font-family:tok;src:url(tok.ttf)}}@font-face{{font-family:len;src:url(len.ttf)}}
html,body{{margin:0;background:#fff}}.row{{height:6px;margin-bottom:4px;white-space:nowrap;font-size:0}}
.T,.L{{display:inline-block;font-size:1000px;line-height:0;height:0;white-space:normal;vertical-align:top;color:transparent}}
.T{{font-family:tok}} .L{{font-family:len}}
.T::before,.L::before{{content:"\\E000"}}
.M{{display:inline-block;width:1px;height:6px;background:#00f;vertical-align:top}}
</style>{rows}"""
open("t.html", "w").write(html)
H = (len(ids) + 4) * 10 + 20
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size=400,{H}", "--screenshot=out.png", "file:///Users/jhome/readme-lm/trackC/cssfont/input/t.html"], capture_output=True, timeout=60)
im = np.array(Image.open("out.png").convert("RGB")).astype(int)
got = []
for r in range(len(ids) + 3):
    row = im[r * 10 + 3]; b = np.where((row[:, 2] > 200) & (row[:, 0] < 60))[0]
    got.append(int(b[0]) if len(b) else -1)
print("normalised prompt:", repr(norm), "len", len(norm))
dec = [g - 1 for g in got[:-1]]   # id = width - 1 ; width 0 => -1 (beyond end)
print("expected ids:", ids); print("decoded ids :", dec[:len(ids)], "beyond end ->", dec[len(ids):])
print("len-font width (=char count):", got[-1])
ok = dec[:len(ids)] == ids and got[-1] == len(norm)
print("INPUT-TOKENIZER", "PASS" if ok else "FAIL")
