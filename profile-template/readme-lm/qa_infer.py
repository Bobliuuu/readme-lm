"""numpy-only inference for the README-embedded QA transformer (KV cache, greedy)."""
import re, io, base64, numpy as np

def load(readme):
    r = open(readme).read()
    b = re.search(r"```\n(.*?)\n```", r.split("<!--LM-WEIGHTS-START-->")[1], re.S).group(1).replace("\n", "")
    Z = np.load(io.BytesIO(base64.b64decode(b)))
    def dq(n):
        p, s = Z[n + "_q"], Z[n + "_s"].astype(np.float32); o = p.shape[0]
        q = np.empty((o, p.shape[1] * 2), np.float32); q[:, 0::2] = (p & 15).astype(np.float32) - 8; q[:, 1::2] = (p >> 4).astype(np.float32) - 8
        return (q.reshape(o, -1, 32) * s[..., None]).reshape(o, -1)
    d, L, H, ff = [int(x) for x in Z["cfg"]]
    m = dict(d=d, L=L, H=H, voc="".join(chr(c) for c in Z["voc"]), emb=dq("emb"), gf=Z["gf"].astype(np.float32), blocks=[])
    for l in range(L):
        m["blocks"].append({n: dq(f"{n}{l}") for n in ("qkv", "o", "u", "dn")} | {"g1": Z[f"g1_{l}"].astype(np.float32), "g2": Z[f"g2_{l}"].astype(np.float32)})
    m["params"] = sum(Z[k].size for k in Z.files if k.endswith("_q")) * 2
    return m

def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * g
def erf(x):
    s = np.sign(x); x = np.abs(x); t = 1 / (1 + 0.3275911 * x)
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x))
def gelu(x): return 0.5 * x * (1 + erf(x / 1.41421356))

def rope(x, pos, D):  # x: H,D
    f = 1.0 / (10000 ** (np.arange(0, D, 2) / D)); a = pos * f; c, s = np.cos(a), np.sin(a)
    out = np.empty_like(x); out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s; out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c; return out

def answer(m, question, maxn=300, norm=None):
    voc, d, H = m["voc"], m["d"], m["H"]; D = d // H; EOS = len(voc)
    q = "".join(c for c in question.lower() if c in voc); q = re.sub(r"\s+", " ", q).strip()[:140]
    toks = [voc.index(c) for c in "q: " + q + "\na: "]; K = [[] for _ in range(m["L"])]; Vv = [[] for _ in range(m["L"])]; out = []
    for pos in range(len(toks) + maxn):
        t = toks[pos]; x = m["emb"][t].copy()
        for l, B in enumerate(m["blocks"]):
            qkv = (B["qkv"] @ rms(x, B["g1"])).reshape(3, H, D)
            K[l].append(rope(qkv[1], pos, D)); Vv[l].append(qkv[2]); qq = rope(qkv[0], pos, D)
            Kc, Vc = np.stack(K[l], 1), np.stack(Vv[l], 1)           # H,T,D
            sc = np.einsum("hd,htd->ht", qq, Kc) / np.sqrt(D); sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
            x = x + B["o"] @ np.einsum("ht,htd->hd", sc, Vc).reshape(d)
            x = x + B["dn"] @ gelu(B["u"] @ rms(x, B["g2"]))
        if pos + 1 < len(toks): continue
        n = int((m["emb"] @ rms(x, m["gf"])).argmax())
        if n == EOS: break
        toks.append(n); out.append(voc[n])
    return "".join(out)
