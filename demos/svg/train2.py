#!/usr/bin/env python3
"""Generation-quality training (8-bit QAT, same integer network as train.py): memorise coherent sentences so greedy generation
from the first 7 characters recites them.   train2.py CTX H out.npz [steps] [d1 d2 m1]"""
import sys, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import textprep, train as T
VOC = textprep.VOC; IDX = {c: i for i, c in enumerate(VOC)}

def greedy(M, prompt, n=80):
    CTX = int(M["ctx"]); s = " " * CTX + prompt; gen = ""
    while len(gen) < n:
        ids = np.array([[IDX[c] for c in s[-CTX:]]]); nxt = VOC[int(T.int_forward(M, ids)[3].argmax(1)[0])]
        s += nxt; gen += nxt
        if len(gen) >= 2 and gen[-1] == " " and gen[-2] in ".?": break
    return gen
def recitation(M, sents, n=80):
    bad = []
    for t in sents:
        g = greedy(M, t[:7], n)
        if g.rstrip() != t[7:]: bad.append((t, t[:7] + g))
    return len(sents) - len(bad), bad

def main():
    CTX, H, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    d1, d2, m1 = (int(sys.argv[5]) if len(sys.argv) > 5 else 8), (int(sys.argv[6]) if len(sys.argv) > 6 else 96), (float(sys.argv[7]) if len(sys.argv) > 7 else 4.0)
    sents = textprep.sentences(); torch.manual_seed(0)
    X, Y = [], []
    for t in sents:
        s = [IDX[" "]] * CTX + [IDX[c] for c in t] + [IDX[" "]]
        for i in range(2 * CTX, len(s)): X.append(s[i - CTX:i]); Y.append(s[i])
    X = torch.tensor(X); Y = torch.tensor(Y); print("sentences", len(sents), "samples", len(X), flush=True)
    rd = lambda x: x + (torch.round(x) - x).detach(); cl = lambda x, lo, hi: x + (torch.clamp(x, lo, hi) - x).detach()
    W1 = (torch.randn(CTX, 32, H) * 20).requires_grad_(); b1 = torch.zeros(H).requires_grad_()
    W2 = (torch.randn(32, H) * 12).requires_grad_(); b2 = torch.zeros(32).requires_grad_(); la = torch.tensor(0.1).requires_grad_()
    opt = torch.optim.Adam([W1, b1, W2, b2, la], lr=0.6); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=0.01)
    def fwd(x):
        w1 = rd(torch.clamp(W1, -127, 127)); a1 = rd(torch.clamp(b1, -4000, 4000)) + sum(w1[k][x[:, k]] for k in range(CTX))
        p = cl(rd(a1 / d1) + 128, 0, 255); h = cl(rd(torch.relu(p - 128) * m1), 0, 255)
        a2 = h @ rd(torch.clamp(W2, -127, 127)).T + rd(torch.clamp(b2, -30000, 30000))
        return (cl(rd(a2 / d2) + 128, 0, 255) - 128) * la.abs()
    for st in range(STEPS):
        loss = torch.nn.functional.cross_entropy(fwd(X), Y); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if st % 500 == 499 or st == STEPS - 1:
            with torch.no_grad(): acc = (fwd(X).argmax(1) == Y).float().mean().item()
            print("step", st, "loss", round(loss.item(), 4), "train acc", round(acc, 4), flush=True)
    lut = np.clip(np.floor(np.maximum(np.arange(256) - 128, 0) * m1 + 0.5), 0, 255).astype(np.int64)
    M = dict(W1=np.clip(np.round(W1.detach().numpy()), -127, 127).astype(np.int16), b1=np.round(b1.detach().numpy()).astype(np.int32),
             W2=np.clip(np.round(W2.detach().numpy()), -127, 127).astype(np.int16), b2=np.round(b2.detach().numpy()).astype(np.int32),
             d1=np.array(d1), d2=np.array(d2), lut=lut.astype(np.int16), ctx=np.array(CTX))
    _, _, h, q = T.int_forward(M, X.numpy()); acc = (q.argmax(1) == Y.numpy()).mean()
    ok, bad = recitation(M, sents)
    print(f"CTX={CTX} H={H} int next-char acc {acc:.4f}  recitation {ok}/{len(sents)} = {ok/len(sents):.3f}  max h {h.max()} logit codes {q.min()}..{q.max()}")
    for t, g in bad[:6]: print("  MISS:", t, "\n   GOT:", g)
    np.savez(out, **M, recit=ok / len(sents))
if __name__ == "__main__": main()
