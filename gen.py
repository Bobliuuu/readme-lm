#!/usr/bin/env python3
"""README-LM sampler. The model weights are read FROM the README text itself
(int8 base64 block between <!--LM-WEIGHTS-START/END--> markers). numpy only.
Usage: gen.py README.md OUT.svg SEED [PROMPT]   (prints the generated text)"""
import sys, re, io, base64, html, numpy as np

def load(readme):
    r = open(readme).read()
    b = re.search(r"```\n(.*?)\n```", r.split("<!--LM-WEIGHTS-START-->")[1], re.S).group(1).replace("\n", "")
    Q = np.load(io.BytesIO(base64.b64decode(b)))
    m = {"emb": Q["emb"].astype(np.float32), "voc": "".join(chr(c) for c in Q["voc"]), "k": int(Q["k"])}
    for n in "012":
        m["w" + n] = Q["q" + n].astype(np.float32) * Q["s" + n].astype(np.float32); m["b" + n] = Q["b" + n].astype(np.float32)
    m["params"] = sum(Q[x].size for x in ("emb", "q0", "q1", "q2", "b0", "b1", "b2"))
    return m

def sample(m, prompt, n, temp, rng, top_p=0.95):
    voc, K = m["voc"], m["k"]
    s = "".join(c for c in prompt if c in voc); ctx = [voc.index(c) for c in (" " * K + s)[-K:]]
    out = s
    for _ in range(n):
        e = m["emb"][ctx].reshape(-1)
        h = np.tanh(m["w0"] @ e + m["b0"]); h = np.tanh(m["w1"] @ h + m["b1"])
        l = (m["w2"] @ h + m["b2"]) / temp; p = np.exp(l - l.max()); p /= p.sum()
        o = np.argsort(-p); c = np.cumsum(p[o]); keep = o[: int((c < top_p).sum()) + 1]
        q = p[keep] / p[keep].sum(); i = int(rng.choice(keep, p=q))
        out += voc[i]; ctx = ctx[1:] + [i]
    return out

def wrap(s, w):
    lines, cur = [], ""
    for word in s.split(" "):
        if len(cur) + len(word) + (1 if cur else 0) > w: lines.append(cur); cur = word
        else: cur = (cur + " " + word) if cur else word
    return lines + ([cur] if cur else [])

def svg(prompt, text, temp, meta, W=64, cw=9.0, lh=24):
    lines = wrap(text, W); N = sum(len(l) for l in lines)
    CPS = 14; typ = N / CPS; T = typ + 5   # seconds per loop
    H = 100 + lh * len(lines); width = int(W * cw + 56)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{H}" viewBox="0 0 {width} {H}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">',
         f'<rect width="{width}" height="{H}" rx="12" fill="#0d1117" stroke="#30363d"/>',
         '<circle cx="24" cy="22" r="6" fill="#ff5f56"/><circle cx="44" cy="22" r="6" fill="#ffbd2e"/><circle cx="64" cy="22" r="6" fill="#27c93f"/>',
         f'<text x="88" y="27" font-size="12" fill="#8b949e">README-LM · {meta} · temp {temp}</text>',
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
    readme, out = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    m = load(readme); rng = np.random.default_rng(seed)
    prompts = ["The model ", "A transformer ", "We compile ", "Inside Qwen "]
    prompt = (sys.argv[4] if len(sys.argv) > 4 and sys.argv[4].strip() else prompts[seed % len(prompts)])
    prompt = "".join(c for c in prompt if c in m["voc"] and c != "\n")[:60]
    text = re.sub(r"\s+", " ", sample(m, prompt + (" " if not prompt.endswith(" ") else ""), 300, 0.7, rng)).strip()
    meta = f"neural MLP · {m['params']//1000}K params · {m['k']}-char context"
    open(out, "w").write(svg(prompt.strip(), text, 0.7, meta))
    print(text)
