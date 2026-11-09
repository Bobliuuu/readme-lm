#!/usr/bin/env python3
"""Compile the QA transformer (qa_final2.pt, 4-bit QAT weights) into pure-Markdown click-trees: one tree per menu question.
State = question + answer text so far. Usage: compile_qa.py OUTDIR [nodes_per_question]"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qa_model import QA, VOC, EOS, V
import build_qa

dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = QA().to(dev); m.load_state_dict(torch.load("qa_final2.pt", map_location=dev)); m.set_qat(True); m.eval()
I = {c: i for i, c in enumerate(VOC)}
TOPK, MINP = 3, 0.10

@torch.no_grad()
def probs(seqs):
    out = []
    for j in range(0, len(seqs), 256):
        ch = seqs[j:j + 256]; L = max(map(len, ch)); X = torch.full((len(ch), L), EOS)
        for i, s in enumerate(ch): X[i, :len(s)] = torch.tensor(s)
        lg = m(X.to(dev)); last = torch.stack([lg[i, len(s) - 1] for i, s in enumerate(ch)])
        out.append(torch.softmax(last, -1).cpu().numpy())
    return np.concatenate(out)

def build(q, N, maxd=200):
    pre = [I[c] for c in "q: " + q + "\na: "]
    nodes = [dict(par=-1, ch=None, pp=1.0, ids=pre, ans="", kids=[], end=False)]; frontier = [0]
    while frontier and len(nodes) < N:
        P = probs([nodes[i]["ids"] for i in frontier]); cand = []
        for fi, i in enumerate(frontier):
            p = P[fi]; top = np.argsort(-p)[:TOPK]
            if top[0] == EOS and p[EOS] >= 0.5: nodes[i]["end"] = True; continue
            for t in top:
                if t != EOS and (p[t] >= MINP or t == top[0]): cand.append((nodes[i]["pp"] * float(p[t]), i, int(t), float(p[t])))
        cand.sort(reverse=True); nxt = []
        for pp, i, t, pt in cand:
            if len(nodes) >= N or len(nodes[i]["ans"]) >= maxd: continue
            nodes.append(dict(par=i, ch=VOC[t], pp=pp, ids=nodes[i]["ids"] + [t], ans=nodes[i]["ans"] + VOC[t], kids=[], end=False)); nodes[i]["kids"].append((len(nodes) - 1, pt)); nxt.append(len(nodes) - 1)
        frontier = nxt
    return nodes

def esc(s): return s.replace("`", "'").replace("|", "/").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")
def vis(c): return "␣" if c == " " else esc(c)


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

def render_tree(qi, q, nodes):
    kept, links = compress(nodes); o = [f'<a name="q{qi}n0"></a>\n### {esc(q)}\n']
    lk = lambda i: " · ".join(f"[`{''.join(vis(nodes[c]['ch']) for c in ch)}` {p*100:.0f}%](#q{qi}n{t})" for t, ch, p in links[i])
    for i in kept:
        if i == 0: o.append("→ " + lk(0) + "<br>"); continue
        n = nodes[i]; back = f"[⬅](#q{qi}n{n['par']}) " if False else ""
        tail = " ■ *(end of answer)*" if n["end"] else (" → " + lk(i) if links[i] else " *(end of compiled graph)*")
        o.append(f'<a name="q{qi}n{i}"></a>**{esc(n["ans"])}**{tail}<br>')
    return "\n".join(o) + "\n", len(kept)

if __name__ == "__main__":
    outd = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 260; os.makedirs(outd, exist_ok=True)
    items, _ = build_qa.load(); qs = []; seen = set()
    for o in items:
        q = build_qa.norm_q(o["qs"][0])
        if q and q not in seen: seen.add(q); qs.append((o["repo"], q))
    qs.sort(key=lambda x: (x[0] != "profile", x[0] != "curated", x[0].lower(), x[1]))
    files, cur, size, fi, stats, menu = [], [], 0, 1, [], {}
    def flush():
        global cur, size, fi
        if cur: open(f"{outd}/q{fi:02d}.md", "w").write(f"# Questions, page {fi}\n\n[⬅ question menu](README.md)\n\n" + "\n".join(cur)); fi += 1; cur, size = [], 0
    for qi, (repo, q) in enumerate(qs):
        nodes = build(q, N); md, nk = render_tree(qi, q, nodes)
        if size + len(md.encode()) > 420_000: flush()
        menu.setdefault(repo, []).append((q, fi, qi)); cur.append(md); size += len(md.encode())
        complete = sum(1 for n in nodes if n["end"]); stats.append(dict(q=q, states=len(nodes), kept=nk, complete=complete))
        if qi % 50 == 0: print(qi, len(qs), q, len(nodes), flush=True)
    flush()
    idx = ["# Ask the QA transformer in pure Markdown\n", "Pick a question; follow the links to read the answer the 4-bit transformer would generate, one click per decision: a link spans the whole run of characters the model is confident about (probability = product along the run), and the states where it has real alternatives show them as separate links. The text so far is shown in each state. No Action, no JavaScript.\n"]
    for repo in sorted(menu, key=lambda r: (r != "profile", r != "curated", r.lower())):
        idx.append(f"\n## {repo}\n"); idx += [f"- [{esc(q)}](q{f:02d}.md#q{qi}n0)" for q, f, qi in menu[repo]]
    open(f"{outd}/README.md", "w").write("\n".join(idx) + "\n")
    tot = sum(s["states"] for s in stats); kept_tot = sum(s["kept"] for s in stats); done = sum(1 for s in stats if s["complete"])
    json.dump(dict(questions=len(qs), states_raw=tot, states_after_chain_contraction=kept_tot, answers_reaching_end=done), open(f"{outd}/stats.json", "w"))
    print("questions", len(qs), "raw states", tot, "after contraction", kept_tot, "trees where greedy path reaches end of answer:", done)
