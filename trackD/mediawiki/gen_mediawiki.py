#!/usr/bin/env python3
"""Train a tiny integer char-model and emit README.mediawiki whose every
'live' cell is computed by the renderer (WikiCloth #expr/#ifeq/#switch).

Model (all integer arithmetic, exact in float64):
  ctx (c1,c2) -> x = [E[c1];E[c2]]  (embedding gather = literals in the page)
  h_j = relu(sum_i W1[j,i]*x_i + b1_j)          H hidden units
  z_k = sum_j V[k,j]*h_j + b2_k                 NBITS output bit-logits
  bit_k = z_k > 0 ; next char = table[bits]     (binary-coded output)
"""
import sys, json, numpy as np, torch
H = int(sys.argv[1]) if len(sys.argv) > 1 else 8
D = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
R = 7
CORPUS = ("qwentpu is a tpu compiled into the weights of a stock qwen. "
          "it is a language model that does exact matrix multiplication. "
          "the model runs with no tpu hardware and no training. ")
VOC = "abcdefghijklmnopqrstuvwxyz .,-'?!_"[:32]
V = len(VOC); NB = 5
idx = {c: i for i, c in enumerate(VOC)}
ids = [idx[c] for c in CORPUS]
ctx = np.array([(ids[i], ids[i+1]) for i in range(len(ids)-2)])
tgt = np.array([ids[i+2] for i in range(len(ids)-2)])
bits = np.array([[(t >> (NB-1-k)) & 1 for k in range(NB)] for t in tgt], dtype=np.float32)

def train(seed=0, steps=6000):
    torch.manual_seed(seed)
    q = lambda x, r=R: x + (torch.round(torch.clamp(x, -r, r)) - x).detach()
    E = (torch.randn(V, D)*3).requires_grad_()
    W1 = (torch.randn(H, 2*D)*3).requires_grad_()
    b1 = torch.zeros(H).requires_grad_()
    Vv = (torch.randn(NB, H)*3).requires_grad_()
    b2 = torch.zeros(NB).requires_grad_()
    opt = torch.optim.Adam([E, W1, b1, Vv, b2], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    c = torch.tensor(ctx); y = torch.tensor(bits)
    def fwd():
        x = torch.cat([q(E)[c[:, 0]], q(E)[c[:, 1]]], 1)
        h = torch.relu(x @ q(W1).T + q(b1, 60))
        return (h @ q(Vv).T + q(b2, 600)) / 60.0
    for s in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(fwd(), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        acc = ((fwd() > 0).float() == y).all(1).float().mean().item()
    r = lambda t, m=R: np.clip(np.round(t.detach().numpy()), -m, m).astype(int)
    return acc, dict(E=r(E), W1=r(W1), b1=r(b1, 60), V=r(Vv), b2=r(b2, 600))

def ref_forward(M, c1, c2):
    x = list(M['E'][c1]) + list(M['E'][c2])
    h = [max(0, int(sum(int(M['W1'][j][i])*int(x[i]) for i in range(2*D)) + int(M['b1'][j]))) for j in range(H)]
    z = [int(sum(int(M['V'][k][j])*h[j] for j in range(H)) + int(M['b2'][k])) for k in range(NB)]
    b = [1 if v > 0 else 0 for v in z]
    return x, h, z, b, int(''.join(map(str, b)), 2)

# ---------- wikitext emission ----------
def hidden_expr(M, j, c1, c2):
    x = list(M['E'][c1]) + list(M['E'][c2])
    P = []; N = []
    for i in range(2*D):
        w = int(M['W1'][j][i]); xi = int(x[i])
        if w == 0 or xi == 0: continue
        (P if (w > 0) == (xi > 0) else N).append(f"{abs(w)}*{abs(xi)}")
    b = int(M['b1'][j])
    if b > 0: P.append(str(b))
    elif b < 0: N.append(str(-b))
    P = "+".join(P) or "0"; N = "+".join(N) or "0"
    return P, N

def relu_expr(P, N):
    return "{{#ifeq:{{#expr:%s>%s}}|1|{{#expr:%s-(%s)}}|0}}" % (P, N, P, N)

def logit_PN(M, k, c1, c2):
    P = []; N = []
    for j in range(H):
        v = int(M['V'][k][j])
        if v == 0: continue
        hp, hn = hidden_expr(M, j, c1, c2)
        (P if v > 0 else N).append(f"{abs(v)}*{relu_expr(hp, hn)}")
    b = int(M['b2'][k])
    if b > 0: P.append(str(b))
    elif b < 0: N.append(str(-b))
    return "+".join(P) or "0", "+".join(N) or "0"

def next_char_expr(M, c1, c2):
    parts = []
    for k in range(NB):
        P, N = logit_PN(M, k, c1, c2)
        parts.append("{{#expr:%s>%s}}" % (P, N))
    cases = "|".join(f"{''.join(str((t>>(NB-1-k))&1) for k in range(NB))}={('_' if VOC[t]==' ' else VOC[t])}" for t in range(V))
    return "{{#switch:" + "".join(parts) + "|" + cases + "}}"

def disp(ch): return {' ': '␣'}.get(ch, ch)

def main():
    best = None
    for seed in range(6):
        acc, M = train(seed)
        print("seed", seed, "train exact-next-char acc (quantised)", round(acc, 3), file=sys.stderr)
        if best is None or acc > best[0]: best = (acc, M)
    acc, M = best
    np.savez("model.npz", **M)
    # choose rows: first NROWS distinct contexts in text order
    seen = []; rows = []
    for (a, b), t in zip(ctx, tgt):
        if (a, b) in seen: continue
        seen.append((a, b)); rows.append((int(a), int(b), int(t)))
        if len(rows) == NROWS: break
    out = []
    w = out.append
    w("= A neural network computed by the page renderer =\n")
    w(f"This file is plain MediaWiki markup (<code>README.mediawiki</code>). There is no JavaScript, no GitHub Action and no program of ours: "
      f"every cell marked ''live'' below is evaluated by the markup renderer's parser functions (<code>#expr</code>, <code>#ifeq</code>, <code>#switch</code>) when this page is rendered. "
      f"The model is a ''character model'': given two characters it predicts the next one with a {2*D}-input, {H}-hidden-unit (ReLU), {NB}-output-bit integer network ({V} symbols; ␣ = space). "
      f"Trained offline (numpy/torch) on {len(CORPUS)} characters of this project's own README text; trained exact-match on that text: {acc*100:.0f}%.\n")
    w("== 1. Live next-character predictions ==")
    w("Columns: the two input characters, the true next character in the training text, the next character computed ''live by the renderer'' from the integer weights spelled out inside the cell, and the numpy reference (a literal, for comparison only). If the renderer did not compute, the live column would show raw <code><nowiki>{{#switch...}}</nowiki></code> text.\n")
    w('{| class="wikitable"\n! n !! input !! true next !! live (renderer) !! numpy ref')
    for n, (a, b, t) in enumerate(rows):
        ref = ref_forward(M, a, b)[4]
        w("|-")
        w(f"| {n+1} || <code>{disp(VOC[a])}{disp(VOC[b])}</code> || {disp(VOC[t])} || {next_char_expr(M, a, b)} || {disp(VOC[ref]) if VOC[ref]!=' ' else '␣'}")
    w("|}\n")
    # section 2 : full network for one input
    a, b, t = rows[1]
    xs, hs, zs, bs, r = ref_forward(M, a, b)
    w(f"== 2. Inside the network: input <code>{disp(VOC[a])}{disp(VOC[b])}</code> ==")
    w(f"Embedding vector (gathered literal) x = {xs}.\n")
    w("Hidden layer, h = ReLU(W1 x + b1). Each cell is a live <code>#ifeq/#expr</code> expression; the last column is the numpy value.\n")
    w('{| class="wikitable"\n! unit !! live h !! numpy h')
    for j in range(H):
        P, N = hidden_expr(M, j, a, b)
        w("|-"); w(f"| h{j+1} || {relu_expr(P, N)} || {hs[j]}")
    w("|}\n")
    w("Output layer: bit logits z = V h + b2 (the hidden units are recomputed inline inside every cell - there are no variables), then bit = [z &gt; 0].\n")
    w('{| class="wikitable"\n! bit !! live logit z !! numpy z !! live bit !! numpy bit')
    for k in range(NB):
        P, N = logit_PN(M, k, a, b)
        w("|-"); w("| z%d || {{#expr:%s-(%s)}} || %d || {{#expr:%s>%s}} || %d" % (k+1, P, N, zs[k], P, N, bs[k]))
    w("|}\n")
    w(f"Predicted next character (live): <b>{next_char_expr(M, a, b)}</b> &nbsp; (numpy: <b>{disp(VOC[r])}</b>; true: <b>{disp(VOC[t])}</b>)\n")
    w("== 3. What is and is not computed by the renderer ==")
    w("* '''Computed live''': every multiply-accumulate of both layers, the ReLU, the sign test of each output bit, and the bit-string to character decode.\n"
      "* '''Fixed in the text''': the trained integer weights (as literals), the choice of which two input characters each row uses, and the embedding gather (the two embedding vectors for the row's input are written as literals; there is no variable or loop to look them up).\n"
      "* A README has no viewer input, so each row is a constant expression; the demonstration is that a real network forward pass is evaluated by the renderer, not that it is interactive.\n")
    txt = "\n".join(out)
    open("README.mediawiki", "w").write(txt)
    print(f"acc={acc:.3f} rows={len(rows)} bytes={len(txt.encode())}")
main()
