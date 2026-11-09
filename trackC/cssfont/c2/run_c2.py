import re, numpy as np, subprocess, math
from PIL import Image
from math import erf, sqrt, exp
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
S = 200; OFF = 4
sig = lambda z: 1/(1+exp(-z))
x1, x2 = -1.3, 0.7
s = [0.5, 1.5, -0.3, 1.0]; m = max(s); e = [exp(v-m) for v in s]; sm = [v/sum(e) for v in e]
xs = [0.6, -1.1, 0.3, 0.9, -0.4, 0.2]; ms = sum(v*v for v in xs)/6; rs = 1/sqrt(ms+1e-5)
th = 0.8*3  # pos*freq
cases = []  # (name, vars, css expr, expected)
cases.append(("relu(-1.3)", {"x": x1}, "max(0, var(--x))", max(0, x1)))
cases.append(("relu(0.7)", {"x": x2}, "max(0, var(--x))", max(0, x2)))
for x in (-2.0, -0.5, 0.5, 1.5):
    cases.append((f"gelu_sig({x})", {"x": x}, "var(--x) / (1 + exp(-1.702 * var(--x)))", x*sig(1.702*x)))
    cases.append((f"  vs exact erf gelu ({x})", {"x": x}, "var(--x) / (1 + exp(-1.702 * var(--x)))", 0.5*x*(1+erf(x/sqrt(2)))))
cases.append(("exp(1.2)", {"x": 1.2}, "exp(var(--x))", exp(1.2)))
cases.append(("pow(1.5,2.5)", {"x": 1.5}, "pow(var(--x), 2.5)", 1.5**2.5))
cases.append(("sqrt(2.25)", {"x": 2.25}, "sqrt(var(--x))", 1.5))
for i in range(4):
    # softmax_i with max-subtract
    css = "exp(var(--s%d) - max(var(--s0),var(--s1),var(--s2),var(--s3))) / (exp(var(--s0)-max(var(--s0),var(--s1),var(--s2),var(--s3))) + exp(var(--s1)-max(var(--s0),var(--s1),var(--s2),var(--s3))) + exp(var(--s2)-max(var(--s0),var(--s1),var(--s2),var(--s3))) + exp(var(--s3)-max(var(--s0),var(--s1),var(--s2),var(--s3))))" % i
    cases.append((f"softmax[{i}]", {f"s{k}": s[k] for k in range(4)}, css, sm[i]))
vv = {f"v{k}": xs[k] for k in range(6)}
cases.append(("rms rsqrt", vv, "1 / sqrt((var(--v0)* var(--v0) + var(--v1)*var(--v1)+var(--v2)*var(--v2)+var(--v3)*var(--v3)+var(--v4)*var(--v4)+var(--v5)*var(--v5))/6 + 0.00001)", rs))
cases.append(("rms out[1]", vv, "var(--v1) / sqrt((var(--v0)* var(--v0) + var(--v1)*var(--v1)+var(--v2)*var(--v2)+var(--v3)*var(--v3)+var(--v4)*var(--v4)+var(--v5)*var(--v5))/6 + 0.00001)", xs[1]*rs))
cases.append(("rope cos(2.4)", {"a": th}, "cos(var(--a) * 1rad)", math.cos(th)))
cases.append(("rope x1*c-x2*s", {"a": th, "p": 0.9, "q": -0.4}, "var(--p)*cos(var(--a)*1rad) - var(--q)*sin(var(--a)*1rad)", 0.9*math.cos(th)+0.4*math.sin(th)))
cases.append(("round(2.37,0.25)", {"x": 2.37}, "round(var(--x), 0.25)", 2.25))
cases.append(("mod(3.7,1.5)", {"x": 3.7}, "mod(var(--x), 1.5)", 3.7 % 1.5))
cases.append(("clamp(-1,x=2.5,1.2)", {"x": 2.5}, "clamp(-1, var(--x), 1.2)", 1.2))
cases.append(("tanh via exp (x=0.8)", {"x": 0.8}, "(exp(2 * var(--x)) - 1) / (exp(2 * var(--x)) + 1)", math.tanh(0.8)))
cases.append(("argmax-ish: max(4)", {f"s{k}": s[k] for k in range(4)}, "max(var(--s0),var(--s1),var(--s2),var(--s3))", max(s)))
props = set(k for c in cases for k in c[1])
reg = "".join(f"@property --{k}{{syntax:'<number>';inherits:false;initial-value:0}}\n" for k in props)
rows = ""
for i, (n, v, css, ex) in enumerate(cases):
    st = ";".join(f"--{k}:{val}" for k, val in v.items())
    css=re.sub(r"\)\s*([+\-*/])\s*(?=[\w(])", r") \1 ", css)
    rows += f'<div class=r style="{st};--y:calc({css})"></div>\n'
html = f"<!doctype html><style>{reg}@property --y{{syntax:'<number>';inherits:false;initial-value:0}}html,body{{margin:0}}.r{{height:6px;margin-bottom:4px;background:#f00;width:calc((var(--y) + {OFF}) * {S}px)}}</style>\n{rows}"
open("t.html", "w").write(html)
H = len(cases)*10+20
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size=2400,{H}", "--screenshot=out.png", "file:///Users/jhome/readme-lm/trackC/cssfont/c2/t.html"], capture_output=True, timeout=60)
im = np.array(Image.open("out.png").convert("RGB")).astype(float)
worst = 0; fails = 0
for i, (n, v, css, ex) in enumerate(cases):
    row = im[i*10+3]; cov = ((255-row[:,1])/255).sum()   # subpixel coverage
    val = cov/S - OFF
    err = abs(val-ex); tag = "(info)" if n.startswith("  vs") else ("OK" if err < 0.01 else "FAIL")
    if tag == "FAIL": fails += 1
    print(f"{n:28s} css={val:+.4f} np={ex:+.4f} err={err:.4f} {tag}")
print("C2", "PASS" if fails == 0 else "FAIL")
