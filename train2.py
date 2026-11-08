#!/usr/bin/env python3
"""Train the big README-LM: char MLP, K-char context, sized so int8+base64 fits a 500 KiB README."""
import glob, re, sys, time, numpy as np, torch, torch.nn as nn
import pydoc_data.topics as T
VOC = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:'\"()-_/=+*#<>[]!?\n"
V = len(VOC); I = {c: i for i, c in enumerate(VOC)}
K, E, H = 12, 16, 448

def clean(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S); t = re.sub(r"<[^>]*>", " ", t); t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
    return "".join(c for c in t if c in I)

def corpus():
    own = " ".join(open(f, errors="ignore").read() for f in sorted(glob.glob("data/*.md")))
    py = "\n".join(T.topics.values())
    py = re.sub(r"\n[ ]{4,}[^\n]*(?=\n)", "\n", py)  # drop indented code-ish lines
    return clean(py), clean(own)

class M(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, E)
        s.net = nn.Sequential(nn.Linear(K * E, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, V))
    def forward(s, x): return s.net(s.emb(x).flatten(1))

if __name__ == "__main__":
    epochs = int(sys.argv[1]); torch.manual_seed(0)
    py, own = corpus(); text = py + "\n" + (own + "\n") * 4
    print("py", len(py), "own", len(own), "total", len(text), flush=True)
    ids = torch.tensor([I[c] for c in text]); n = len(ids) - K
    X = torch.stack([ids[i:i + n] for i in range(K)], 1); Y = ids[K:K + n]
    cut = int(n * 0.97); perm0 = torch.randperm(n); tr, va = perm0[:cut], perm0[cut:]
    m = M(); print("params", sum(p.numel() for p in m.parameters()), flush=True)
    BS = 1024; steps = epochs * ((cut + BS - 1) // BS)
    opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    for ep in range(epochs):
        t0 = time.time(); p = tr[torch.randperm(cut)]
        for i in range(0, cut, BS):
            b = p[i:i + BS]; loss = nn.functional.cross_entropy(m(X[b]), Y[b], label_smoothing=0.02)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad(): vl = nn.functional.cross_entropy(m(X[va[:50000]]), Y[va[:50000]]).item()
        print(f"ep {ep} train {loss.item():.3f} val {vl:.3f} ppl {np.exp(vl):.2f} {time.time()-t0:.0f}s", flush=True)
        sd = {k: v.numpy() for k, v in m.state_dict().items()}
        np.savez("lm_big.npz", emb=sd["emb.weight"], w0=sd["net.0.weight"], b0=sd["net.0.bias"], w1=sd["net.2.weight"],
                 b1=sd["net.2.bias"], w2=sd["net.4.weight"], b2=sd["net.4.bias"], voc=np.array([ord(c) for c in VOC]), k=np.array(K))
