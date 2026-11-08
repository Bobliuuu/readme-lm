#!/usr/bin/env python3
"""README-LM compiler: train a tiny neural char-LM (MLP, k-char context) and
compile its finite-context predictions into pure GitHub Markdown links."""
import glob, re, sys, json, collections, torch, torch.nn as nn

ALPHA = " abcdefghijklmnopqrstuvwxyz"
V = len(ALPHA)
I = {c: i for i, c in enumerate(ALPHA)}

def corpus():
    t = " ".join(open(f, errors="ignore").read() for f in sorted(glob.glob("data/*.md")))
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"<[^>]*>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t).lower()
    return re.sub(r"[^a-z]+", " ", t).strip()

class MLP(nn.Module):
    def __init__(s, k, e=24, h=256):
        super().__init__()
        s.k = k; s.emb = nn.Embedding(V, e)
        s.net = nn.Sequential(nn.Linear(k * e, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, V))
    def forward(s, x):
        return s.net(s.emb(x).flatten(1))

def train(text, k, epochs=40, seed=0):
    torch.manual_seed(seed)
    ids = torch.tensor([I[c] for c in text])
    n = len(ids) - k
    X = torch.stack([ids[i:i + n] for i in range(k)], 1)
    Y = ids[k:k + n]
    cut = int(n * 0.95)
    m = MLP(k); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-3, total_steps=epochs * ((cut + 511) // 512))
    for ep in range(epochs):
        perm = torch.randperm(cut)
        for i in range(0, cut, 512):
            b = perm[i:i + 512]
            loss = nn.functional.cross_entropy(m(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        val = nn.functional.cross_entropy(m(X[cut:]), Y[cut:]).item()
    return m, val

def sid(ctx): return "s-" + ctx.replace(" ", "_")
def show(c): return "␣" if c == " " else c

def compile_graph(m, k, topn, minp, maxstates, start):
    """BFS over states reachable via the top-n (>=minp) predictions."""
    seen, order, q, rows = {start}, [], collections.deque([start]), {}
    while q and len(order) < maxstates:
        ctx = q.popleft()
        with torch.no_grad():
            p = torch.softmax(m(torch.tensor([[I[c] for c in ctx]])), -1)[0]
        pr, ix = p.topk(topn)
        links = [(ALPHA[i], float(v)) for v, i in zip(pr, ix) if v >= minp] or [(ALPHA[ix[0]], float(pr[0]))]
        rows[ctx] = links; order.append(ctx)
        for c, _ in links:
            nx = (ctx + c)[1:]
            if nx not in seen: seen.add(nx); q.append(nx)
    keep = set(order)
    return order, {c: [(ch, pv) for ch, pv in rows[c] if (c + ch)[1:] in keep] or rows[c][:1] for c in order}, keep

def render(order, rows, keep, heading_level=3):
    out = []
    for ctx in order:
        links = " · ".join(f"[`{show(c)}` {round(p*100)}%](#{sid((ctx+c)[1:])})" for c, p in rows[ctx] if (ctx + c)[1:] in keep)
        out.append(f'<a name="{sid(ctx)}"></a>`{"".join(show(c) for c in ctx)}` → {links}<br>')
    return "\n".join(out)

if __name__ == "__main__":
    k, topn, minp, maxs = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    text = corpus()
    m, val = train(text, k)
    start = ("the " * k)[-k:] if k <= 4 else " " * k
    order, rows, keep = compile_graph(m, k, topn, minp, maxs, start)
    body = render(order, rows, keep)
    open(f"out/model_k{k}.md", "w").write(body)
    st = dict(k=k, topn=topn, minp=minp, val_loss=round(val, 3), val_ppl=round(2.718281828 ** val, 2),
              states=len(order), bytes=len(body.encode()), possible=V ** k, params=sum(p.numel() for p in m.parameters()))
    print(json.dumps(st))
    torch.save(m.state_dict(), f"out/model_k{k}.pt")
