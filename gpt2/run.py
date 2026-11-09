#!/usr/bin/env python3
"""run.py "prompt" [max_new_tokens] [seed] [temperature]
GPT-2 small (124M) in numpy. Weights AND tokenizer are read from the Markdown shards in this directory
(weights/part-*.md, tokenizer/part-*.md, manifest.md): int8 + base64 + zlib. Needs: numpy, regex."""
import sys, os, re, json, zlib, base64, glob, numpy as np, regex
D = os.path.dirname(os.path.abspath(__file__))

def read_shards(kind, d=None):
    parts = sorted(glob.glob(f"{d or D}/{kind}/part-*.md")); b = []
    for p in parts: b.append(re.search(r"```\n(.*?)\n```", open(p).read(), re.S).group(1).replace("\n", ""))
    return zlib.decompress(base64.b64decode("".join(b)))

def load_weights(d=None):
    man = json.loads(re.search(r"```json\n(.*?)\n```", open(f"{d or D}/manifest.md").read(), re.S).group(1))["tensors"]
    raw = memoryview(read_shards("weights", d)); off = 0; W = {}
    for t in man:
        shape, n = tuple(t["shape"]), 1
        for s in t["shape"]: n *= s
        if t["q"] is None: W[t["name"]] = np.frombuffer(raw[off:off + n * 4], "<f4").reshape(shape).astype(np.float32)
        else:
            q = np.frombuffer(raw[off:off + n], np.int8).reshape(shape); ns = shape[1 - t["q"]]
            s = np.frombuffer(raw[off + n:off + n + ns * 2], "<f2").astype(np.float32)
            W[t["name"]] = q.astype(np.float32) * (s[None, :] if t["q"] == 0 else s[:, None])
        off += t["nbytes"]
    return W

# ---- GPT-2 byte-level BPE ----
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))
PAT = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
class Tok:
    def __init__(s, d=None):
        t = json.loads(read_shards("tokenizer", d)); s.enc = t["vocab"]; s.dec = {v: k for k, v in s.enc.items()}
        s.rank = {tuple(m.split()): i for i, m in enumerate(t["merges"])}; s.b2u = bytes_to_unicode(); s.u2b = {v: k for k, v in s.b2u.items()}; s.cache = {}
    def bpe(s, w):
        if w in s.cache: return s.cache[w]
        parts = list(w)
        while len(parts) > 1:
            pairs = {(parts[i], parts[i + 1]) for i in range(len(parts) - 1)}
            best = min(pairs, key=lambda p: s.rank.get(p, 1e18))
            if best not in s.rank: break
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best: out.append(parts[i] + parts[i + 1]); i += 2
                else: out.append(parts[i]); i += 1
            parts = out
        s.cache[w] = parts; return parts
    def encode(s, text):
        ids = []
        for tok in PAT.findall(text):
            u = "".join(s.b2u[b] for b in tok.encode("utf-8")); ids += [s.enc[p] for p in s.bpe(u)]
        return ids
    def decode(s, ids): return bytearray(s.u2b[c] for c in "".join(s.dec[i] for i in ids)).decode("utf-8", errors="replace")

# ---- model ----
def ln(x, g, b): m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + 1e-5) * g + b
def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
H, DM = 12, 768
def forward(W, ids, cache, pos0):
    """ids: list of token ids at positions pos0..; cache: list of [K,V] per layer (T,H,64). returns last-position logits."""
    x = W["wte"][ids] + W["wpe"][pos0:pos0 + len(ids)]; T = len(ids)
    for l in range(12):
        p = f"h.{l}."; h = ln(x, W[p + "ln1.g"], W[p + "ln1.b"]); qkv = h @ W[p + "attn.w"] + W[p + "attn.b"]
        q, k, v = [a.reshape(T, H, 64) for a in np.split(qkv, 3, -1)]
        if cache[l] is None: cache[l] = [k, v]
        else: cache[l] = [np.concatenate([cache[l][0], k]), np.concatenate([cache[l][1], v])]
        K, V = cache[l]; S = K.shape[0]
        sc = np.einsum("thd,shd->hts", q, K) / 8.0
        mask = (np.arange(S)[None, :] > (pos0 + np.arange(T))[:, None]); sc = np.where(mask[None], -1e30, sc)
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        x = x + np.einsum("hts,shd->thd", sc, V).reshape(T, DM) @ W[p + "proj.w"] + W[p + "proj.b"]
        h = ln(x, W[p + "ln2.g"], W[p + "ln2.b"]); x = x + gelu(h @ W[p + "fc.w"] + W[p + "fc.b"]) @ W[p + "mp.w"] + W[p + "mp.b"]
    return ln(x[-1:], W["lnf.g"], W["lnf.b"])[0] @ W["wte"].T

def generate(W, tok, prompt, n=60, seed=0, temp=0.8, topk=40):
    rng = np.random.default_rng(seed); ids = tok.encode(prompt)[-900:] or [50256]; cache = [None] * 12; out = []
    lg = forward(W, ids, cache, 0); pos = len(ids)
    for _ in range(n):
        if temp <= 0: t = int(lg.argmax())
        else:
            k = np.argsort(-lg)[:topk]; p = np.exp((lg[k] - lg[k].max()) / temp); p /= p.sum(); t = int(rng.choice(k, p=p))
        if t == 50256: break
        out.append(t); lg = forward(W, [t], cache, pos); pos += 1
    return tok.decode(out)

if __name__ == "__main__":
    prompt = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    print(generate(load_weights(), Tok(), prompt, n, seed, temp))
