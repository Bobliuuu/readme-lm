import sys, time, json, random, numpy as np, torch, torch.nn.functional as F
from qa_model import *
import build_qa

dev = "mps" if torch.backends.mps.is_available() else "cpu"
I = {c: i for i, c in enumerate(VOC)}
MAXLEN = 448

def enc(q, a):
    p = [I[c] for c in "q: " + q + "\na: "]; r = [I[c] for c in a] + [EOS]
    return (p + r)[:MAXLEN], ([0] * len(p) + [1] * len(r))[:MAXLEN]

def batchify(pairs, idx):
    xs = [enc(*pairs[i][:2]) for i in idx]; L = max(len(x[0]) for x in xs)
    X = torch.full((len(xs), L), EOS); M = torch.zeros(len(xs), L)
    for j, (t, m) in enumerate(xs): X[j, :len(t)] = torch.tensor(t); M[j, :len(m)] = torch.tensor(m, dtype=torch.float)
    return X.to(dev), M.to(dev)

@torch.no_grad()
def generate(model, q, maxn=300):
    model.eval(); toks = [I[c] for c in "q: " + build_qa.norm_q(q) + "\na: "]; out = []
    for _ in range(maxn):
        lg = model(torch.tensor([toks[-MAXLEN:]], device=dev))[0, -1]
        n = int(lg.argmax())
        if n == EOS: break
        toks.append(n); out.append(VOC[n])
    model.train(); return "".join(out)

def score(pred, ref):  # char 4-gram F1
    g = lambda s: {s[i:i + 4] for i in range(max(1, len(s) - 3))}
    p, r = g(pred), g(ref); inter = len(p & r)
    return 0 if not inter else 2 * inter / (len(p) + len(r))

def evaluate(model, test, n=120):
    rng = random.Random(1); sub = rng.sample(test, min(n, len(test))); sc = []; ref_ok = []
    for q, a, repo in sub:
        out = generate(model, q)
        if a == "REFUSE": ref_ok.append(out.startswith(("I can only", "That is outside", "I only know")))
        else: sc.append(score(out, a))
    return (float(np.mean(sc)) if sc else 0, float(np.mean(ref_ok)) if ref_ok else 0)

if __name__ == "__main__":
    epochs = int(sys.argv[1]); holdout = float(sys.argv[2]); tag = sys.argv[3]
    d, L, ff = (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (128, 3, 512)
    torch.manual_seed(0); train, test = build_qa.make(holdout=holdout)
    print("train", len(train), "test", len(test), "device", dev, flush=True)
    model = QA(d=d, L=L, ff=ff).to(dev); P = sum(p.numel() for p in model.parameters()); print("params", P, flush=True)
    BS = 48; steps = epochs * (len(train) // BS)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=0.01, betas=(0.9, 0.98))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.1)
    step = 0
    for ep in range(epochs):
        model.set_qat(ep >= epochs // 2); t0 = time.time(); perm = list(range(len(train))); random.shuffle(perm)
        for i in range(0, len(perm) - BS + 1, BS):
            X, M = batchify(train, perm[i:i + BS])
            lg = model(X[:, :-1]); loss = (F.cross_entropy(lg.reshape(-1, V), X[:, 1:].reshape(-1), reduction="none") * M[:, 1:].reshape(-1)).sum() / M[:, 1:].sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); step += 1
        msg = f"ep {ep} loss {loss.item():.3f} qat {model.qat} {time.time()-t0:.0f}s"
        if test and (ep % 5 == 4 or ep == epochs - 1): msg += " holdout(f1,refuse) %.3f %.3f" % evaluate(model, test)
        print(msg, flush=True); torch.save(model.state_dict(), f"qa_{tag}.pt")
