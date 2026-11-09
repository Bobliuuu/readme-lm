import sys, os, re, glob, json, zlib, base64, numpy as np, regex
FENCE = chr(96) * 3

def blocks(md, kind): return [m.group(1) for m in re.finditer(FENCE + kind + r"\n(.*?)\n" + FENCE, md, re.S)]
def kv(txt): return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in txt.split("\n") if ":" in l)}

# ---- weights and tokenizer live in Markdown shards (base64 of zlib of int8) ----
def read_shards(pattern):
    parts = sorted(glob.glob(pattern)); b = []
    for p in parts: b.append(re.search(FENCE + r"\n(.*?)\n" + FENCE, open(p).read(), re.S).group(1).replace("\n", ""))
    return zlib.decompress(base64.b64decode("".join(b)))

def load_weights(manifest, pattern):
    man = json.loads(re.search(FENCE + r"json\n(.*?)\n" + FENCE, open(manifest).read(), re.S).group(1))["tensors"]
    raw = memoryview(read_shards(pattern)); off = 0; W = {}
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
    def __init__(s, pattern):
        t = json.loads(read_shards(pattern)); s.enc = t["vocab"]; s.dec = {v: k for k, v in s.enc.items()}
        s.rank = {tuple(m.split()): i for i, m in enumerate(t["merges"])}; s.b2u = bytes_to_unicode(); s.u2b = {v: k for k, v in s.b2u.items()}; s.cache = {}
    def bpe(s, w):
        if w in s.cache: return s.cache[w]
        parts = list(w)
        while len(parts) > 1:
            pairs = {(parts[i], parts[i + 1]) for i in range(len(parts) - 1)}; best = min(pairs, key=lambda p: s.rank.get(p, 1e18))
            if best not in s.rank: break
            out, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best: out.append(parts[i] + parts[i + 1]); i += 2
                else: out.append(parts[i]); i += 1
            parts = out
        s.cache[w] = parts; return parts
    def encode(s, text):
        ids = []
        for tok in PAT.findall(text): ids += [s.enc[p] for p in s.bpe("".join(s.b2u[b] for b in tok.encode("utf-8")))]
        return ids
    def decode(s, ids): return bytearray(s.u2b[c] for c in "".join(s.dec[i] for i in ids)).decode("utf-8", errors="replace")

# ---- ops the declarative program may call ----
def op_embed(tokens, wte, wpe): return wte[tokens] + wpe[:len(tokens)]
def op_layernorm(x, g, b): m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + 1e-5) * g + b
def op_linear(x, w, b): return x @ w + b
def op_gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
def op_add(a, b): return a + b
def op_head(x, wte): return x[-1] @ wte.T
def op_attention(qkv, heads):
    T, D3 = qkv.shape; D = D3 // 3; hd = D // heads
    q, k, v = [a.reshape(T, heads, hd) for a in np.split(qkv, 3, -1)]
    sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd); sc = np.where(np.tril(np.ones((T, T), bool))[None], sc, -1e30)
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", sc, v).reshape(T, D)
OPS = dict(embed=op_embed, layernorm=op_layernorm, linear=op_linear, gelu=op_gelu, add=op_add, head=op_head, attention=op_attention)

# ---- interpreter for the program block:  name = op(arg, op(arg), k=v)  and  repeat N as i ----
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

# ---- the renderer: executes this document's own mdrt-run block and writes the rendered Markdown ----
def main():
    doc = sys.argv[1]; md = open(doc).read(); base = os.path.dirname(os.path.abspath(doc))
    cfg = kv(blocks(md, "mdrt-weights")[0]); prog = parse_program(blocks(md, "mdrt-program")[0])
    P = lambda k: os.path.join(base, cfg[k])
    W = load_weights(P("manifest"), P("weights")); tok = Tok(P("tokenizer")); M = Machine(W)
    def run_block(m):
        r = kv(m.group(1)); prompt = r["prompt"]; n = int(r.get("tokens", 20)); T = float(r.get("temperature", 0)); k = int(r.get("top_k", 40)); rng = np.random.default_rng(int(r.get("seed", 0)))
        ids = tok.encode(prompt); rows = []
        for step in range(n):
            logits = M.exec(prog, {"tokens": ids}, {})["logits"]; p = np.exp(logits - logits.max()); p /= p.sum(); top = np.argsort(-p)[:3]
            if T <= 0: t = int(top[0])
            else: kk = np.argsort(-logits)[:k]; q = np.exp((logits[kk] - logits[kk].max()) / T); q /= q.sum(); t = int(rng.choice(kk, p=q))
            if t == 50256: break
            rows.append((step, t, float(p[t]), [(tok.decode([int(i)]), float(p[i])) for i in top])); ids = ids + [t]
        esc = lambda s: s.replace(chr(10), "⏎").replace("|", "/").replace(chr(96), chr(39))
        tbl = "\n".join(f"| {i} | `{esc(tok.decode([t]))}` | {pt*100:.1f}% | " + " · ".join(f"`{esc(s)}` {q*100:.0f}%" for s, q in tp) + " |" for i, t, pt, tp in rows)
        out = tok.decode(ids).replace(chr(96), chr(39))
        return (f"**Prompt:** {prompt}\n\n**Generated ({len(rows)} tokens, temperature {T}):**\n\n> {out.replace(chr(10), ' ⏎ ')}\n\n| step | chosen token | p | top-3 |\n|---|---|---|---|\n{tbl}\n")
    outs = []
    def sub(m): s = run_block(m); outs.append(s); return s
    rendered = re.sub(FENCE + r"mdrt-run\n(.*?)\n" + FENCE, sub, md, flags=re.S)
    text = "\n\n".join(outs) if "--run-only" in sys.argv else rendered
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else None
    open(out, "w").write(text) if out else print(text)
main()
