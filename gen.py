#!/usr/bin/env python3
"""Sample from the README-LM neural net (numpy only) and write an animated
typing SVG. Usage: gen.py OUT.svg [seed]"""
import sys, html, numpy as np, random
ALPHA = " abcdefghijklmnopqrstuvwxyz"; K = 6
Z = np.load(__file__.rsplit("/", 1)[0] + "/lm_k6.npz") if "/" in __file__ else np.load("lm_k6.npz")

def probs(ctx, temp):
    e = Z["emb"][[ALPHA.index(c) for c in ctx]].reshape(-1)
    h = np.tanh(Z["w0"] @ e + Z["b0"]); h = np.tanh(Z["w1"] @ h + Z["b1"])
    l = (Z["w2"] @ h + Z["b2"]) / temp
    p = np.exp(l - l.max()); return p / p.sum()

def sample(seed_text, n, temp, rng):
    s = seed_text
    for _ in range(n):
        p = probs(s[-K:], temp); s += ALPHA[rng.choice(27, p=p)]
    return s

def wrap(s, w):
    lines, cur = [], ""
    for word in s.split(" "):
        if len(cur) + len(word) + (1 if cur else 0) > w: lines.append(cur); cur = word
        else: cur = (cur + " " + word) if cur else word
    return lines + ([cur] if cur else [])

def svg(prompt, text, temp, W=64, cw=9.0, lh=24):
    lines = wrap(text, W); N = sum(len(l) for l in lines)
    CPS = 14; typ = N / CPS; T = typ + 5   # seconds per loop
    H = 100 + lh * len(lines); width = int(W * cw + 56)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{H}" viewBox="0 0 {width} {H}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
         f'<rect width="{width}" height="{H}" rx="12" fill="#0d1117" stroke="#30363d"/>',
         '<circle cx="24" cy="22" r="6" fill="#ff5f56"/><circle cx="44" cy="22" r="6" fill="#ffbd2e"/><circle cx="64" cy="22" r="6" fill="#27c93f"/>',
         f'<text x="88" y="27" font-size="12" fill="#8b949e">README-LM · neural MLP (92K params) · {K}-char context · temp {temp}</text>',
         f'<text x="28" y="62" font-size="13" fill="#58a6ff">prompt&gt; {html.escape(prompt)}</text>']
    done = 0; xs, ys, kt = [], [], []
    xs.append(28); ys.append(100 - 14); kt.append(0.0)
    for i, ln in enumerate(lines):
        y = 100 + lh * i; n = len(ln)
        steps = ";".join(f"{c*cw:.1f}" for c in range(n + 1))
        kts = ";".join(f"{(done + c) / N * typ / T:.5f}" for c in range(n + 1))
        # clip width: 0 before start, stepping while typed, full after
        vals = "0;" + steps + ";" + f"{n*cw:.1f}"
        times = "0;" + kts + ";1"
        # first step value 0 at t=start, so prepend duplicates handled by discrete mode
        o.append(f'<clipPath id="c{i}"><rect x="28" y="{y-16}" height="{lh}" width="0">'
                 f'<animate attributeName="width" dur="{T:.2f}s" repeatCount="indefinite" calcMode="discrete" values="{vals}" keyTimes="{times}"/></rect></clipPath>')
        o.append(f'<text x="28" y="{y}" font-size="15" fill="#e6edf3" textLength="{n*cw:.1f}" lengthAdjust="spacing" clip-path="url(#c{i})" xml:space="preserve">{html.escape(ln)}</text>')
        for c in range(1, n + 1):
            xs.append(28 + c * cw); ys.append(y - 14); kt.append((done + c) / N * typ / T)
        done += n
    kt = kt[:len(xs)]
    xv = ";".join(f"{x:.1f}" for x in xs) + f";{xs[0]:.0f}"; yv = ";".join(f"{y}" for y in ys) + f";{ys[0]}"
    kv = ";".join(f"{k:.5f}" for k in kt) + ";1"
    o.append(f'<rect width="8" height="16" fill="#58a6ff"><animate attributeName="x" dur="{T:.2f}s" repeatCount="indefinite" calcMode="discrete" values="{xv}" keyTimes="{kv}"/>'
             f'<animate attributeName="y" dur="{T:.2f}s" repeatCount="indefinite" calcMode="discrete" values="{yv}" keyTimes="{kv}"/></rect></svg>')
    return "\n".join(o)

if __name__ == "__main__":
    out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    prompts = ["the model ", "a transformer ", "we compile ", "inside qwen "]
    prompt = prompts[seed % len(prompts)]
    pad = (" " * K + prompt)[-K:]
    text = sample(prompt, 230, 0.7, rng).strip()
    open(out, "w").write(svg(prompt.strip(), text, 0.7))
    print(text)
