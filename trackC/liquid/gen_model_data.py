import re, io, base64, json, numpy as np, os
root = os.path.dirname(os.path.abspath(__file__)); D = root + "/site5/_data"
b = re.search(r"```\n(.*?)\n```", open("/Users/jhome/readme-lm/qa/weights.md").read(), re.S).group(1).replace("\n", "")
Z = np.load(io.BytesIO(base64.b64decode(b)))
def unpack(n):
    p = Z[n + "_q"]; o = p.shape[0]; q = np.empty((o, p.shape[1] * 2), np.int64); q[:, 0::2] = (p & 15).astype(np.int64) - 8; q[:, 1::2] = (p >> 4).astype(np.int64) - 8
    return q, Z[n + "_s"].astype(np.float32)
def dq(n):
    q, s = unpack(n); return (q.reshape(q.shape[0], -1, 32) * s[..., None]).reshape(q.shape[0], -1)
sizes = {}
def dump(name, obj):
    s = json.dumps(obj, separators=(",", ":")); open(f"{D}/{name}.json", "w").write(s); sizes[name] = len(s)
names = ["emb"] + [f"{n}{l}" for l in range(3) for n in ("qkv", "o", "u", "dn")]
for n in names:
    q, s = unpack(n); dump(n, dict(q=q.tolist(), s=[[float(v) for v in r] for r in s]))
dump("norms", {k: [float(v) for v in Z[k]] for k in Z.files if k[0] == "g"})
print(sum(sizes.values()), "bytes JSON total")
# reference for first op chain on token 'q'
voc = "".join(chr(c) for c in Z["voc"]); t = voc.index("q")
x = dq("emb")[t].astype(np.float64)
g1 = Z["g1_0"].astype(np.float64); xn = x / np.sqrt((x * x).mean() + 1e-5) * g1
qkv = dq("qkv0").astype(np.float64) @ xn
json.dump(dict(tok=t, x=x.tolist(), xn=xn.tolist(), qkv=qkv.tolist()), open(root + "/ref_l5.json", "w"))
print(t, qkv[:4])
