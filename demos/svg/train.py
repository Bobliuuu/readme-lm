#!/usr/bin/env python3
"""Stage-2 model: character MLP, CTX chars of context (one-hot, 32 symbols) -> H ReLU units -> 32 logits, trained with
8-bit activation QAT and integer kernels (static scales).  Mirrors exactly what the SVG filter graph computes:
  acc1[j] = sum_k W1[k, c_k, j] + b1[j]           (ints; one-hot input pixels are code 1, so this is a 129-tap convolve)
  p[j]    = clip(rnd(acc1/d1) + 128, 0, 255)       (8-bit pre-activation code; the convolve's own round+clamp)
  h[j]    = LUT1[p[j]] = clip(rnd(max(p-128,0)*m1), 0, 255)   (feComponentTransfer ReLU/requant LUT)
  acc2[v] = sum_j W2[v, j] * h[j] + b2[v]          (65-tap convolve)
  q[v]    = clip(rnd(acc2/d2) + 128, 0, 255)       (8-bit logit code)
  next    = argmax_v q[v], lowest index on ties    (feMorphology)
Usage: train.py CTX H out.npz [epochs]"""
import sys, json, re, glob, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
VOC = "abcdefghijklmnopqrstuvwxyz .,-'?"; V = len(VOC); assert V == 32
ROOT = "/Users/jhome/readme-lm"
def clean(s):
    s = s.lower(); s = "".join(ch if ch in VOC else " " for ch in s); return re.sub(r" +", " ", s).strip()
def load_text():
    parts = [clean(open(f"{ROOT}/corpus.txt").read())]
    for f in sorted(glob.glob(f"{ROOT}/qa/g*.jsonl")):
        for l in open(f):
            r = json.loads(l); parts += [clean(q) for q in r["qs"][:2]] + [clean(r["a"])]
    seen = set(); out = []
    for p in parts:
        if p and p not in seen: seen.add(p); out.append(p)
    return out

def int_forward(M, ids):
    """ids: (N, CTX) int -> (acc1, p, h, q) numpy int64. The integer reference."""
    W1, b1, W2, b2, d1, d2, lut = M["W1"].astype(np.int64), M["b1"].astype(np.int64), M["W2"].astype(np.int64), M["b2"].astype(np.int64), int(M["d1"]), int(M["d2"]), M["lut"].astype(np.int64)
    acc1 = b1[None] + sum(W1[k][ids[:, k]] for k in range(ids.shape[1]))
    p = np.clip(np.floor_divide(4 * acc1 + 2 * d1 + 1, 4 * d1) + 128, 0, 255)
    h = lut[p]
    acc2 = h @ W2.T + b2[None]
    q = np.clip(np.floor_divide(4 * acc2 + 2 * d2 + 1, 4 * d2) + 128, 0, 255)
    return acc1, p, h, q

def main():
    CTX, H, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; EP = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    texts = load_text(); torch.manual_seed(0); rng = np.random.default_rng(0)
    idx = {c: i for i, c in enumerate(VOC)}
    X, Y, G = [], [], []
    for gi, t in enumerate(texts):
        s = [idx[" "]] * CTX + [idx[c] for c in t] + [idx[" "]]
        for i in range(CTX, len(s)): X.append(s[i - CTX:i]); Y.append(s[i]); G.append(gi)
    X = np.array(X); Y = np.array(Y); G = np.array(G)
    hold = ((G % 10 == 0) & (G > 0)) | ((np.arange(len(G)) % 10 == 0) & (G == 0)); tr = ~hold
    print("texts", len(texts), "samples", len(X), "train", tr.sum(), "hold", hold.sum(), flush=True)
    d1, d2, m1 = (int(sys.argv[5]) if len(sys.argv)>5 else 8), (int(sys.argv[6]) if len(sys.argv)>6 else 256), (float(sys.argv[7]) if len(sys.argv)>7 else 2.0)
    rd = lambda x: x + (torch.round(x) - x).detach()
    cl = lambda x, lo, hi: x + (torch.clamp(x, lo, hi) - x).detach()
    W1 = (torch.randn(CTX, V, H) * 20).requires_grad_(); b1 = torch.zeros(H).requires_grad_()
    W2 = (torch.randn(V, H) * 12).requires_grad_(); b2 = torch.zeros(V).requires_grad_(); la = torch.tensor(0.1).requires_grad_()
    opt = torch.optim.Adam([W1, b1, W2, b2, la], lr=0.5)
    Xt = torch.tensor(X[tr]); Yt = torch.tensor(Y[tr]); BS = 4096; steps = EP * (len(Xt) // BS + 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.6, total_steps=steps)
    def fwd(x):
        w1 = rd(torch.clamp(W1, -127, 127)); a1 = rd(torch.clamp(b1, -4000, 4000)) + sum(w1[k][x[:, k]] for k in range(CTX))
        p = cl(rd(a1 / d1) + 128, 0, 255); h = cl(rd(torch.relu(p - 128) * m1), 0, 255)
        a2 = h @ rd(torch.clamp(W2, -127, 127)).T + rd(torch.clamp(b2, -30000, 30000))
        q = cl(rd(a2 / d2) + 128, 0, 255); return (q - 128) * la.abs()
    for ep in range(EP):
        perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(Xt), BS):
            b = perm[i:i + BS]; loss = torch.nn.functional.cross_entropy(fwd(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item() * len(b)
        if ep % 5 == 4 or ep == EP - 1: print("ep", ep, "loss", round(tot / len(Xt), 4), flush=True)
    lut = np.clip(np.floor(np.maximum(np.arange(256) - 128, 0) * m1 + 0.5), 0, 255).astype(np.int64)
    M = dict(W1=np.clip(np.round(W1.detach().numpy()), -127, 127).astype(np.int16), b1=np.round(b1.detach().numpy()).astype(np.int32),
             W2=np.clip(np.round(W2.detach().numpy()), -127, 127).astype(np.int16), b2=np.round(b2.detach().numpy()).astype(np.int32),
             d1=np.array(d1), d2=np.array(d2), lut=lut.astype(np.int16), ctx=np.array(CTX))
    _, _, h, q = int_forward(M, X); pred = q.argmax(1)
    a_tr = (pred[tr] == Y[tr]).mean(); a_ho = (pred[hold] == Y[hold]).mean()
    print(f"CTX={CTX} H={H} int acc train {a_tr:.4f} holdout {a_ho:.4f} | logit codes min/max {q.min()} {q.max()} | frac h>0 {(h>0).mean():.2f} | tie rate {(np.sort(q,1)[:,-1]==np.sort(q,1)[:,-2]).mean():.3f}")
    np.savez(out, **M, train_acc=a_tr, hold_acc=a_ho)
if __name__ == '__main__': main()
