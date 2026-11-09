#!/usr/bin/env python3
"""Compile real GPT-2 (weights read from gpt2/*.md shards) into a pure-Markdown link graph.
State = prompt + tokens so far. Links = top next tokens (prob). Best-first under a per-prompt node budget.
Usage: compile_gpt2.py OUTDIR [nodes_per_prompt] [depth]"""
import sys, os, json, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpt2"))
import run

dev = "mps" if torch.backends.mps.is_available() else "cpu"
Wn = run.load_weights(); tok = run.Tok(); W = {k: torch.tensor(v, device=dev) for k, v in Wn.items()}

@torch.no_grad()
def logits_batch(seqs):
    L = max(len(s) for s in seqs); X = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs): X[i, :len(s)] = torch.tensor(s)
    X = X.to(dev); x = W["wte"][X] + W["wpe"][:L][None]
    for l in range(12):
        p = f"h.{l}."; h = F.layer_norm(x, (768,), W[p + "ln1.g"], W[p + "ln1.b"], 1e-5)
        q, k, v = (a.view(len(seqs), L, 12, 64).transpose(1, 2) for a in (h @ W[p + "attn.w"] + W[p + "attn.b"]).split(768, -1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(len(seqs), L, 768)
        x = x + a @ W[p + "proj.w"] + W[p + "proj.b"]
        h = F.layer_norm(x, (768,), W[p + "ln2.g"], W[p + "ln2.b"], 1e-5)
        x = x + F.gelu(h @ W[p + "fc.w"] + W[p + "fc.b"], approximate="tanh") @ W[p + "mp.w"] + W[p + "mp.b"]
    x = F.layer_norm(x, (768,), W["lnf.g"], W["lnf.b"], 1e-5)
    last = torch.stack([x[i, len(s) - 1] for i, s in enumerate(seqs)])
    lg = (last @ W["wte"].T).cpu().numpy() / TEMP
    e = np.exp(lg - lg.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

PROMPTS = ["The capital of France is", "Question: What is the capital of France?\nAnswer:", "Question: Who wrote Hamlet?\nAnswer:", "To be, or not to be,",
           "Roses are red, violets are", "1, 2, 3, 4, 5,", "The Eiffel Tower is located in", "Water boils at 100 degrees",
           "The meaning of life is", "Once upon a time", "GitHub is", "Markdown is"]
TOPK, MINP, TEMP = 3, 0.08, 0.6

def vis(t):  # display token
    s = tok.decode([t]); return s.replace("\n", "⏎").replace(" ", "␣") if s.strip() == "" else s.replace("`", "'").replace("\n", "⏎").replace(" ", "␣", 1) if s.startswith(" ") else s.replace("`", "'").replace("\n", "⏎")
def text(ids): return tok.decode(ids).replace("`", "'").replace("\n", " ⏎ ")

def build(prompt, N, D):
    root = tok.encode(prompt); nodes = [dict(par=-1, tok=None, pp=1.0, ids=root, depth=0, kids=[])]; frontier = [0]; cov = []
    while frontier and len(nodes) < N:
        probs = np.concatenate([logits_batch([nodes[i]["ids"] for i in frontier[j:j + 64]]) for j in range(0, len(frontier), 64)])
        cand = []
        for fi, i in enumerate(frontier):
            p = probs[fi]; top = np.argsort(-p)[:TOPK]
            for t in top:
                t = int(t)
                if (p[t] >= MINP or t == top[0]) and "`" not in tok.decode([t]) and t != 50256: cand.append((nodes[i]["pp"] * float(p[t]), i, t, float(p[t])))
        cand.sort(reverse=True); nxt = []
        for pp, i, t, pt in cand:
            if len(nodes) >= N or nodes[i]["depth"] + 1 > D: continue
            nodes.append(dict(par=i, tok=t, pp=pp, ids=nodes[i]["ids"] + [t], depth=nodes[i]["depth"] + 1, kids=[])); nodes[i]["kids"].append((len(nodes) - 1, pt)); nxt.append(len(nodes) - 1)
        frontier = nxt
    return nodes


def compress(nodes):
    """chain contraction: keep root, branching nodes and leaves; a run of single-continuation states becomes one link"""
    kept = [i for i, n in enumerate(nodes) if i == 0 or len(n["kids"]) != 1]; ks = set(kept); links = {}
    for i in kept:
        L = []
        for c, pt in nodes[i]["kids"]:
            chars, p, cur = [c], pt, c
            while cur not in ks:
                (nc, npt), = nodes[cur]["kids"]; chars.append(nc); p *= npt; cur = nc
            L.append((cur, chars, p))
        links[i] = L
    return kept, links

def render(pi, prompt, nodes, D):
    kept, links = compress(nodes)
    out = [f"# GPT-2 graph {pi + 1}: `{prompt.replace(chr(10), ' ⏎ ')}`\n", "[⬅ all prompts](README.md)\n", f"{len(kept)} states ({len(nodes)} before chain contraction). Each line is one state: the text so far, then GPT-2's most likely continuations as links (probability shown; a link spans a run of tokens when GPT-2 had only one likely continuation). `␣` = space, `⏎` = newline. Click to append. Paths stop at depth {D} tokens or where the budget ran out.\n"]
    for i in kept:
        n = nodes[i]
        links_md = " · ".join(f"[`{''.join(vis(nodes[c]['tok']) for c in ch)}` {p*100:.0f}%](#n{t})" for t, ch, p in links[i]) or "*(end of compiled graph)*"
        out.append(f'<a name="n{i}"></a>**{text(n["ids"])}** → {links_md}<br>')
    return "\n".join(out) + "\n", len(kept)

if __name__ == "__main__":
    outd = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; D = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    os.makedirs(outd, exist_ok=True); idx = ["# GPT-2 in pure Markdown\n", "GPT-2 small (124M), compiled into link graphs. **No Action, no JavaScript, no server: you click.** Each prompt below opens a page whose states are the real GPT-2's top next-token predictions. Pick a token to extend the text; the next state shows the text so far.\n",
           "Honest limits: probabilities are GPT-2's at temperature 0.6 (sharpened, as in typical sampling); only the most probable branches (top-3, probability >= 8%) are compiled, up to a state budget, so you can only reach continuations GPT-2 rated likely; paths end at a depth limit. Compiled from the int8 weights stored in [`../../gpt2/`](../../gpt2/).\n", "| Prompt | States | P(text stays in graph for 8 tokens) | for 12 tokens |\n|---|---|---|---|"]
    stats = []
    for pi, pr in enumerate(PROMPTS):
        nodes = build(pr, N, D); md, nk = render(pi, pr, nodes, D); open(f"{outd}/p{pi + 1:02d}.md", "w").write(md)
        leaf_mass = sum(n["pp"] for n in nodes if n["depth"] == 8); m12 = sum(n["pp"] for n in nodes if n["depth"] == 12)
        stats.append(dict(prompt=pr, states_raw=len(nodes), states=nk, bytes=len(md.encode()), mass_d8=round(leaf_mass, 4), mass_d12=round(m12, 4), maxdepth=max(n["depth"] for n in nodes)))
        idx.append(f"| [{pr.replace(chr(10), ' ')}](p{pi + 1:02d}.md#n0) | {nk} (from {len(nodes)}) | {leaf_mass*100:.1f}% | {m12*100:.1f}% |"); print(stats[-1], flush=True)
    open(f"{outd}/README.md", "w").write("\n".join(idx) + "\n"); json.dump(stats, open(f"{outd}/stats.json", "w"), indent=1)
