# The GitHub QA transformer as one self-contained Markdown document

A 600K-parameter character-level transformer (3 layers, d=128, 4 heads, RoPE, RMSNorm, tied embeddings, 4-bit weights) that answers questions about Jerry Zhu's public GitHub repos. This document carries the engine source, the model program and a run request; the weights are in the Markdown file [`../qa/weights.md`](../qa/weights.md).

**Bootstrap** (extracts the engine from this very document and runs it):

```sh
python3 -c "import re,sys;t=open(sys.argv[1]).read();exec(re.search(chr(96)*4+'python mdrt-engine\\n(.*?)\\n'+chr(96)*4,t,re.S).group(1))" QA.md
```

Needs `numpy`. Flags: `-o OUT.md`, `--run-only`.

GitHub only displays this document; the extracted engine does the arithmetic. It knows nothing outside Jerry's GitHub, so other questions get a refusal.

## Weights

```mdrt-weights
weights: ../qa/weights.md
```

## The model: one step

```mdrt-program
x = embed(tokens, emb)
repeat 3 as l
  h = rmsnorm(x, g1.{l})
  qkv = linear(h, qkv.{l})
  a = attention_rope(qkv, heads=4)
  a = linear(a, o.{l})
  x = add(x, a)
  h = rmsnorm(x, g2.{l})
  m = gelu(linear(h, u.{l}))
  m = linear(m, dn.{l})
  x = add(x, m)
x = rmsnorm(x, gf)
logits = head(x, emb)
```

## Run

```mdrt-run
question: what is qwentpu
max_chars: 300
```

## The engine

````python mdrt-engine
import sys, os, re, io, base64, numpy as np
FENCE = chr(96) * 3

def blocks(md, kind): return [m.group(1) for m in re.finditer(FENCE + kind + r"\n(.*?)\n" + FENCE, md, re.S)]
def kv(txt): return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in txt.split("\n") if ":" in l)}

# ---- weights live in a Markdown file: base64 of an npz of 4-bit packed tensors ----
def load_weights(path):
    b = re.search(FENCE + r"\n(.*?)\n" + FENCE, open(path).read(), re.S).group(1).replace("\n", "")
    Z = np.load(io.BytesIO(base64.b64decode(b)))
    def dq(n):
        p, s = Z[n + "_q"], Z[n + "_s"].astype(np.float32); o = p.shape[0]
        q = np.empty((o, p.shape[1] * 2), np.float32); q[:, 0::2] = (p & 15).astype(np.float32) - 8; q[:, 1::2] = (p >> 4).astype(np.float32) - 8
        return (q.reshape(o, -1, 32) * s[..., None]).reshape(o, -1)
    d, L, H, ff = [int(x) for x in Z["cfg"]]; W = {"emb": dq("emb"), "gf": Z["gf"].astype(np.float32)}
    for l in range(L):
        for n in ("qkv", "o", "u", "dn"): W[f"{n}.{l}"] = dq(f"{n}{l}")
        W[f"g1.{l}"] = Z[f"g1_{l}"].astype(np.float32); W[f"g2.{l}"] = Z[f"g2_{l}"].astype(np.float32)
    return W, "".join(chr(c) for c in Z["voc"])

# ---- ops ----
def erf(x):
    s = np.sign(x); x = np.abs(x); t = 1 / (1 + 0.3275911 * x)
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x))
def op_embed(tokens, emb): return emb[tokens]
def op_rmsnorm(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * g
def op_linear(x, w): return x @ w.T
def op_gelu(x): return 0.5 * x * (1 + erf(x / 1.41421356))
def op_add(a, b): return a + b
def op_head(x, emb): return x[-1] @ emb.T
def rope(x):  # x: T,H,D  (interleaved pairs, as trained)
    T, H, D = x.shape; f = 1.0 / (10000 ** (np.arange(0, D, 2) / D)); a = np.arange(T)[:, None] * f[None]; c, s = np.cos(a)[:, None], np.sin(a)[:, None]
    out = np.empty_like(x); out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s; out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c; return out
def op_attention_rope(qkv, heads):
    T, D3 = qkv.shape; D = D3 // 3; hd = D // heads
    q, k, v = [a.reshape(T, heads, hd) for a in np.split(qkv, 3, -1)]; q, k = rope(q), rope(k)
    sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd); sc = np.where(np.tril(np.ones((T, T), bool))[None], sc, -1e30)
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", sc, v).reshape(T, D)
OPS = dict(embed=op_embed, rmsnorm=op_rmsnorm, linear=op_linear, gelu=op_gelu, add=op_add, head=op_head, attention_rope=op_attention_rope)

# ---- interpreter for the program block ----
TOK = re.compile(r"\s*([A-Za-z0-9_.{}\-]+|[(),=])")
def parse_expr(s):
    toks = TOK.findall(s); pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def eat(): pos[0] += 1; return toks[pos[0] - 1]
    def expr():
        name = eat()
        if peek() == "(":
            eat(); args, kw = [], {}
            while peek() != ")":
                if pos[0] + 1 < len(toks) and toks[pos[0] + 1] == "=": k = eat(); eat(); kw[k] = eat()
                else: args.append(expr())
                if peek() == ",": eat()
            eat(); return ("call", name, args, kw)
        return ("ref", name)
    return expr()
def parse_program(text):
    lines = [l.rstrip() for l in text.strip().split("\n") if l.strip()]
    def block(i, indent):
        out = []
        while i < len(lines):
            ind = len(lines[i]) - len(lines[i].lstrip())
            if ind < indent: break
            l = lines[i].strip(); m = re.match(r"repeat (\d+) as (\w+)$", l)
            if m: body, i = block(i + 1, ind + 2); out.append(("repeat", int(m.group(1)), m.group(2), body)); continue
            lhs, rhs = l.split("=", 1); out.append(("assign", lhs.strip(), parse_expr(rhs.strip()))); i += 1
        return out, i
    return block(0, 0)[0]
class Machine:
    def __init__(s, W): s.W = W
    def lookup(s, name, env, sub):
        for k, v in sub.items(): name = name.replace("{" + k + "}", str(v))
        if name in env: return env[name]
        if name in s.W: return s.W[name]
        try: return int(name)
        except ValueError: return name
    def ev(s, e, env, sub):
        if e[0] == "ref": return s.lookup(e[1], env, sub)
        _, op, args, kw = e; kw = {k: int(v) if v.isdigit() else v for k, v in kw.items()}
        return OPS[op](*[s.ev(a, env, sub) for a in args], **kw)
    def exec(s, prog, env, sub={}):
        for st in prog:
            if st[0] == "assign": env[st[1]] = s.ev(st[2], env, sub)
            else:
                for i in range(st[1]): s.exec(st[3], env, {**sub, st[2]: i})
        return env

# ---- renderer: executes this document's own mdrt-run block ----
def main():
    doc = sys.argv[1]; md = open(doc).read(); base = os.path.dirname(os.path.abspath(doc))
    cfg = kv(blocks(md, "mdrt-weights")[0]); prog = parse_program(blocks(md, "mdrt-program")[0])
    W, voc = load_weights(os.path.join(base, cfg["weights"])); M = Machine(W); EOS = len(voc)
    def run_block(m):
        r = kv(m.group(1)); q = "".join(c for c in r["question"].lower() if c in voc); q = re.sub(r"\s+", " ", q).strip()[:140]
        ids = [voc.index(c) for c in "q: " + q + chr(10) + "a: "]; out = []; pts = []
        for step in range(int(r.get("max_chars", 300))):
            lg = M.exec(prog, {"tokens": ids}, {})["logits"]; p = np.exp(lg - lg.max()); p /= p.sum(); top = np.argsort(-p)[:3]; t = int(top[0])
            if t == EOS: break
            if p[t] < 0.95: pts.append((len(out), [(voc[i] if i < EOS else "END", float(p[i])) for i in top]))
            out.append(voc[t]); ids = ids + [t]
        ans = "".join(out); show = lambda c: "space" if c == " " else c.replace("|", "/")
        tbl = "\n".join(f"| {i} | `{ans[:i][-18:]}` | " + " · ".join(f"`{show(c)}` {pp*100:.0f}%" for c, pp in tp) + " |" for i, tp in pts[:40])
        head = f"**Question:** {q}\n\n**Answer ({len(ans)} characters, greedy):**\n\n> {ans}\n\n"
        if not pts: return head + "The model was at least 95% sure of every character, so there were no real alternatives to show.\n"
        return head + f"Places where the model was less than 95% sure (up to 40):\n\n| position | text before | top-3 next characters |\n|---|---|---|\n{tbl}\n"
    outs = []
    def sub(m): s = run_block(m); outs.append(s); return s
    rendered = re.sub(FENCE + r"mdrt-run\n(.*?)\n" + FENCE, sub, md, flags=re.S)
    text = "\n\n".join(outs) if "--run-only" in sys.argv else rendered
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else None
    open(out, "w").write(text) if out else print(text)
main()
````
