"""C4 numeric feasibility: simulate the QA transformer with CSS-font-compatible arithmetic limits vs the float numpy reference.
Constraints modelled: per-row symmetric int weights (WB bits, folding the 4-bit group scales into the integer deltas),
matmul inputs = axis values clipped to [-1,1]/static per-site scale and quantised to F2Dot14 (1/16384),
matmul outputs quantised to 1/64 unit (Chromium LayoutUnit) of a width whose full-scale is 30000 units,
exact-erf GELU (CSS has sign/abs/exp), RoPE via cos/sin, softmax via exp, argmax exact."""
import sys, re, io, base64, json, glob, random, numpy as np
sys.path.insert(0, "/Users/jhome/readme-lm")
import importlib.util
spec = importlib.util.spec_from_file_location("qa_engine", "/Users/jhome/readme-lm/mdrt/qa_engine.py")
def load(path="/Users/jhome/readme-lm/qa/weights.md"):
    b = re.search(r"```\n(.*?)\n```", open(path).read(), re.S).group(1).replace("\n", "")
    Z = np.load(io.BytesIO(base64.b64decode(b)))
    def dq(n):
        p, s = Z[n + "_q"], Z[n + "_s"].astype(np.float32); o = p.shape[0]
        q = np.empty((o, p.shape[1] * 2), np.float32); q[:, 0::2] = (p & 15).astype(np.float32) - 8; q[:, 1::2] = (p >> 4).astype(np.float32) - 8
        return (q.reshape(o, -1, 32) * s[..., None]).reshape(o, -1)
    d, L, H, ff = [int(x) for x in Z["cfg"]]
    m = dict(d=d, L=L, H=H, ff=ff, voc="".join(chr(c) for c in Z["voc"]), emb=dq("emb"), gf=Z["gf"].astype(np.float32), blocks=[])
    for l in range(L):
        m["blocks"].append({n: dq(f"{n}{l}") for n in ("qkv", "o", "u", "dn")} | {"g1": Z[f"g1_{l}"].astype(np.float32), "g2": Z[f"g2_{l}"].astype(np.float32)})
    return m
def rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * g
def erf(x):
    s = np.sign(x); x = np.abs(x); t = 1 / (1 + 0.3275911 * x)
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x))
def gelu(x): return 0.5 * x * (1 + erf(x / 1.41421356))
def rope(x, pos, D):
    f = 1.0 / (10000 ** (np.arange(0, D, 2) / D)); a = pos * f; c, s = np.cos(a), np.sin(a)
    o = np.empty_like(x); o[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s; o[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c; return o

class Q:  # CSS-font matvec
    def __init__(s, W, bits, scale=None):
        s.W = W; s.bits = bits
        if bits:
            qmax = 2 ** (bits - 1) - 1; s.rs = np.abs(W).max(1, keepdims=True) / qmax
            s.Wi = np.round(W / s.rs); 
        s.scale = scale
    def __call__(s, x, record=None):
        if not s.bits: return s.W @ x
        if record is not None: record.append(np.abs(x).max())
        a = np.clip(x / s.scale, -1, 1); a = np.round(a * 16384) / 16384     # F2Dot14 axis
        raw = s.Wi @ a                                                      # integer deltas * axis (units of rs*scale)
        full = np.abs(s.Wi).sum(1).max(); u = 30000 / full                   # K: font units per int step so |sum|<=30000
        raw = np.round(raw * u * 64) / 64 / u                               # 1/64 unit layout quantisation
        return raw * s.rs[:, 0] * s.scale

def run(m, question, WB, scales=None, rec=None, maxn=300, qk_bits=14):
    voc, d, H = m["voc"], m["d"], m["H"]; D = d // H; EOS = len(voc)
    q = "".join(c for c in question.lower() if c in voc); q = re.sub(r"\s+", " ", q).strip()[:140]
    toks = [voc.index(c) for c in "q: " + q + "\na: "]; L = m["L"]
    K = [[] for _ in range(L)]; Vv = [[] for _ in range(L)]; out = []
    if "mats" not in m or m.get("wb") != WB:
        m["wb"] = WB; m["mats"] = []
        for l, B in enumerate(m["blocks"]):
            m["mats"].append({n: Q(B[n], WB) for n in ("qkv", "o", "u", "dn")})
        m["head"] = Q(m["emb"], WB)
    if scales is not None:
        for l in range(L):
            for n in ("qkv", "o", "u", "dn"): m["mats"][l][n].scale = scales[(l, n)]
        m["head"].scale = scales["head"]
    for pos in range(len(toks) + maxn):
        x = m["emb"][toks[pos]].copy()
        for l, B in enumerate(m["blocks"]):
            M = m["mats"][l]; R = (lambda n: rec.setdefault((l, n), [])) if rec is not None else (lambda n: None)
            qkv = M["qkv"](rms(x, B["g1"]), R("qkv")).reshape(3, H, D)
            K[l].append(rope(qkv[1], pos, D)); Vv[l].append(qkv[2]); qq = rope(qkv[0], pos, D)
            Kc, Vc = np.stack(K[l], 1), np.stack(Vv[l], 1)
            sc = np.einsum("hd,htd->ht", qq, Kc) / np.sqrt(D); sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
            x = x + M["o"](np.einsum("ht,htd->hd", sc, Vc).reshape(d), R("o"))
            h = M["u"](rms(x, B["g2"]), R("u")); x = x + M["dn"](gelu(h), R("dn"))
        if pos + 1 < len(toks): continue
        lg = m["head"](rms(x, m["gf"]), rec.setdefault("head", []) if rec is not None else None)
        n = int(lg.argmax())
        if n == EOS: break
        toks.append(n); out.append(voc[n])
    return "".join(out)

if __name__ == "__main__":
    random.seed(0); m = load()
    Qs = []
    for f in sorted(glob.glob("/Users/jhome/readme-lm/qa/g*.jsonl")):
        for l in open(f):
            r = json.loads(l); Qs.append((r["qs"][0], r["a"]))
    random.shuffle(Qs); calib, test = Qs[:12], Qs[12:42]
    # float reference
    for B in m["blocks"]: pass
    ref = {q: run(m, q, 0) for q, _ in test}
    for q, a in test[:3]: print("Q:", q, "\n ref:", ref[q][:90], "\n tgt:", a[:90])
    exact = sum(ref[q].strip() == a.strip() for q, a in test); print("float model exact-match to target on 30:", exact)
    rec = {}
    for q, _ in calib: run(m, q, 9, scales={**{(l, n): 1.0 for l in range(m["L"]) for n in ("qkv","o","u","dn")}, "head": 1.0}, rec=rec, maxn=60)
    for WB in (9, 8, 6):
        sc = {k: (max(v) * 1.15 if k != "head" else max(v) * 1.15) for k, v in rec.items()}
        got = {q: run(m, q, WB, scales=sc) for q, _ in test}
        same = sum(got[q] == ref[q] for q, _ in test)
        import difflib
        sim = np.mean([difflib.SequenceMatcher(None, got[q], ref[q]).ratio() for q, _ in test])
        print(f"WB={WB}: identical-to-float {same}/30, mean similarity {sim:.4f}")
    print({str(k): round(max(v), 2) for k, v in rec.items()})
