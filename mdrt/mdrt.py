#!/usr/bin/env python3
"""mdrt.py DOC.md [-o OUT.md] : a Markdown renderer that executes executable blocks.
  ```mdrt-weights```  : where the model's weights/tokenizer shards (Markdown files) live
  ```mdrt-program```  : declarative forward pass (one token step), interpreted by this renderer
  ```mdrt-run```      : generation request; replaced in the output by the generated tokens
Needs numpy + regex."""
import sys, os, re, glob, json, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpt2"))
import run as G   # reuse shard reader + byte-level BPE (all data comes from Markdown shards)

# ---------------- ops ----------------
def op_embed(tokens, wte, wpe): return wte[tokens] + wpe[:len(tokens)]
def op_layernorm(x, g, b): return G.ln(x, g, b)
def op_linear(x, w, b): return x @ w + b
def op_gelu(x): return G.gelu(x)
def op_add(a, b): return a + b
def op_head(x, wte): return x[-1] @ wte.T
def op_attention(qkv, heads):
    T, D3 = qkv.shape; D = D3 // 3; hd = D // heads
    q, k, v = [a.reshape(T, heads, hd) for a in np.split(qkv, 3, -1)]
    sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd); sc = np.where(np.tril(np.ones((T, T), bool))[None], sc, -1e30)
    sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", sc, v).reshape(T, D)
OPS = dict(embed=op_embed, layernorm=op_layernorm, linear=op_linear, gelu=op_gelu, add=op_add, head=op_head, attention=op_attention)

# ---------------- tiny expression parser:  name = op(arg, op(arg), k=v) ----------------
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
            l = lines[i].strip()
            m = re.match(r"repeat (\d+) as (\w+)$", l)
            if m: body, i = block(i + 1, ind + 2); out.append(("repeat", int(m.group(1)), m.group(2), body)); continue
            lhs, rhs = l.split("=", 1); out.append(("assign", lhs.strip(), parse_expr(rhs.strip()))); i += 1
        return out, i
    return block(0, 0)[0]

class Machine:
    def __init__(s, weights): s.W = weights
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

def blocks(md, kind): return [m.group(1) for m in re.finditer(rf"```{kind}\n(.*?)\n```", md, re.S)]
def kv(txt): return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in txt.split("\n") if ":" in l)}

def render(path):
    md = open(path).read(); base = os.path.dirname(os.path.abspath(path))
    cfg = kv(blocks(md, "mdrt-weights")[0]); prog = parse_program(blocks(md, "mdrt-program")[0])
    d = os.path.normpath(os.path.join(base, os.path.dirname(cfg["manifest"])))          # shard root
    W = G.load_weights(d); tok = G.Tok(d); M = Machine(W)
    def run_block(m):
        r = kv(m.group(1)); prompt = r["prompt"]; n = int(r.get("tokens", 20)); T = float(r.get("temperature", 0)); k = int(r.get("top_k", 40)); rng = np.random.default_rng(int(r.get("seed", 0)))
        ids = tok.encode(prompt); rows = []
        for step in range(n):
            logits = M.exec(prog, {"tokens": ids}, {})["logits"]
            p = np.exp(logits - logits.max()); p /= p.sum(); top = np.argsort(-p)[:3]
            if T <= 0: t = int(top[0])
            else:
                kk = np.argsort(-logits)[:k]; q = np.exp((logits[kk] - logits[kk].max()) / T); q /= q.sum(); t = int(rng.choice(kk, p=q))
            if t == 50256: break
            rows.append((step, t, float(p[t]), [(tok.decode([int(i)]), float(p[i])) for i in top])); ids = ids + [t]
        out = tok.decode(ids)
        tbl = "\n".join(f"| {i} | `{tok.decode([t]).replace(chr(10), '⏎').replace('|', '/')}` | {pt*100:.1f}% | " + " · ".join(f"`{s.replace(chr(10), '⏎').replace('|', '/')}` {q*100:.0f}%" for s, q in tp) + " |" for i, t, pt, tp in rows)
        return (f"**Prompt:** {prompt}\n\n**Generated (by `mdrt`, {len(rows)} tokens, temperature {T}):**\n\n> {out.replace(chr(10), ' ⏎ ')}\n\n"
                f"| step | chosen token | p | top-3 |\n|---|---|---|---|\n{tbl}\n"), out
    outs = []
    def sub(m): s, o = run_block(m); outs.append(o); return s
    rendered = re.sub(r"```mdrt-run\n(.*?)\n```", sub, md, flags=re.S)
    return rendered, outs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("doc"); ap.add_argument("-o"); a = ap.parse_args()
    out, gens = render(a.doc); open(a.o, "w").write(out) if a.o else print(out)
