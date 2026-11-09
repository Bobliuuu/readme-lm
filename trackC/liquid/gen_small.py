import numpy as np, json, os
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site"); D = S + "/_data"
rng = np.random.default_rng(1)
# L1: integer matvec in YAML
M = rng.integers(-9, 10, (4, 4)); x = rng.integers(-9, 10, 4)
with open(D + "/l1.yml", "w") as f:
    f.write("M:\n" + "".join(f"  - [{', '.join(map(str, r))}]\n" for r in M) + f"x: [{', '.join(map(str, x))}]\n")
# L4: MLP 4->8->3, floats, JSON
W1 = rng.normal(0, .5, (8, 4)).round(4); b1 = rng.normal(0, .2, 8).round(4); W2 = rng.normal(0, .5, (3, 8)).round(4); b2 = rng.normal(0, .2, 3).round(4)
X = rng.normal(0, 1, (5, 4)).round(4)
json.dump(dict(W1=W1.tolist(), b1=b1.tolist(), W2=W2.tolist(), b2=b2.tolist(), X=X.tolist()), open(D + "/l4.json", "w"))
# L3 toy LM: vocab 6, tied-ish bigram RNN. h' = relu(A h + E[tok]); logits = O h' ; greedy
V, H = 6, 5
for seed in range(200):
    r2=np.random.default_rng(seed); E = r2.integers(-3, 4, (V, H)); A = r2.integers(-1, 2, (H, H)); O = r2.integers(-3, 4, (V, H))
    h=np.zeros(H,int); o=[]
    for t in [1,2]: h=np.maximum(A@h+E[t],0)
    for _ in range(12):
        n=int(np.argmax(O@h)); o.append(n)
        if n==0: break
        h=np.maximum(A@h+E[n],0)
    if len(set(o))>=3 and o[-1]==0: break
E_ = None
json.dump(dict(E=E.tolist(), A=A.tolist(), O=O.tolist(), prompt=[1, 2], eos=0, steps=12), open(D + "/l3.json", "w"))
ref = {}
ref["l1"] = (M @ x).tolist()
h1 = np.maximum(W1 @ X.T + b1[:, None], 0); ref["l4"] = (W2 @ h1 + b2[:, None]).T.tolist()
h = np.zeros(H, int); toks = [1, 2]; out = []
for t in toks: h = np.maximum(A @ h + E[t], 0)
for s in range(12):
    n = int(np.argmax(O @ h)); out.append(n)
    if n == 0: break
    h = np.maximum(A @ h + E[n], 0)
ref["l3"] = out
json.dump(ref, open(os.path.dirname(S) + "/ref_small.json", "w"))
print(ref["l1"], out)
