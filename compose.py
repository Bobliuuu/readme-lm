import re, sys, build, torch, json
m = build.MLP(3); m.load_state_dict(torch.load("out/model_k3.pt"))
text = build.corpus()

def validate(md):
    names = set(re.findall(r'<a name="([^"]+)">', md))
    links = re.findall(r'\]\(#([^)]+)\)', md)
    bad = [l for l in links if l not in names]
    return len(names), len(links), bad

def sample(order, rows, keep, n=60):
    import random; random.seed(7); ctx = "he "; s = ctx
    for _ in range(n):
        ch = random.choices([c for c, _ in rows[ctx]], [p for _, p in rows[ctx]])[0]
        s += ch; ctx = (ctx + ch)[1:]
    return s

def make(topn, minp, maxs=100000):
    o, r, k = build.compile_graph(m, 3, topn, minp, maxs, "the")
    return o, r, k, build.render(o, r, k)

START = '[`t` `h` `e`](#s-the)'
if __name__ == "__main__":
    o, r, k, body = make(4, 0.12)
    n, l, bad = validate(body)
    print("profile model", n, "states", l, "links", len(bad), "broken", len(body.encode()), "bytes")
    print(sample(o, r, k))
    o2, r2, k2, body2 = make(5, 0.04)
    n2, l2, bad2 = validate(body2); print("repo model", n2, l2, len(bad2), len(body2.encode()))
    open("out/profile_model.md", "w").write(body); open("out/repo_model.md", "w").write(body2)
