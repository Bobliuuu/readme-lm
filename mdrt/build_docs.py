#!/usr/bin/env python3
"""Generate the self-contained executable Markdown documents GPT2.md and QA.md from engine sources."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); F = "`" * 3; F4 = "`" * 4
BOOT = lambda doc: f'''{F}sh
python3 -c "import re,sys;t=open(sys.argv[1]).read();exec(re.search(chr(96)*4+'python mdrt-engine\\\\n(.*?)\\\\n'+chr(96)*4,t,re.S).group(1))" {doc}
{F}'''
def engine(name): return open(os.path.join(HERE, name)).read().rstrip()

gpt2 = f'''# GPT-2 as one self-contained Markdown document

This file carries everything needed to run GPT-2 small (124M parameters) except a Python interpreter:

1. **The engine** (below): Markdown shard reader, GPT-2 byte-level BPE tokenizer, the tensor ops, and an interpreter for the declarative program.
2. **The weights and tokenizer**, referenced from Markdown shards (`../gpt2/*.md`, int8 + base64 + zlib, 112 MB in 341 files).
3. **The model**: the forward pass written as a declarative program (below).
4. **A run request** (below).

**Bootstrap** (the only thing not in this file; the engine is extracted from this very document and executed):

{BOOT("GPT2.md")}

Needs `numpy` and `regex`. Flags: `-o OUT.md` writes the rendered document to a file; `--run-only` outputs only the rendered run block (prompt, generated text, per-token top-3 table).

GitHub's renderer only *displays* this document. The computation is done by the engine extracted from it, so this is Markdown as the complete artifact (weights, architecture, interpreter, request), not Markdown executing itself.

## Weights and tokenizer

{F}mdrt-weights
manifest: ../gpt2/manifest.md
weights: ../gpt2/weights/part-*.md
tokenizer: ../gpt2/tokenizer/part-*.md
{F}

## The model: one token step

`tokens` is the id sequence so far; the program returns `logits` for the last position. Ops: `embed`, `layernorm`, `linear` (x @ W + b), `attention` (multi-head causal over a fused qkv tensor), `gelu`, `add`, `head` (tied output embedding).

{F}mdrt-program
x = embed(tokens, wte, wpe)
repeat 12 as l
  h = layernorm(x, h.{{l}}.ln1.g, h.{{l}}.ln1.b)
  qkv = linear(h, h.{{l}}.attn.w, h.{{l}}.attn.b)
  a = attention(qkv, heads=12)
  a = linear(a, h.{{l}}.proj.w, h.{{l}}.proj.b)
  x = add(x, a)
  h = layernorm(x, h.{{l}}.ln2.g, h.{{l}}.ln2.b)
  m = gelu(linear(h, h.{{l}}.fc.w, h.{{l}}.fc.b))
  m = linear(m, h.{{l}}.mp.w, h.{{l}}.mp.b)
  x = add(x, m)
x = layernorm(x, lnf.g, lnf.b)
logits = head(x, wte)
{F}

## Run

{F}mdrt-run
prompt: The meaning of life is
tokens: 24
temperature: 0.8
top_k: 40
seed: 0
{F}

## The engine

{F4}python mdrt-engine
{engine("engine.py")}
{F4}
'''
qa = f'''# The GitHub QA transformer as one self-contained Markdown document

A 600K-parameter character-level transformer (3 layers, d=128, 4 heads, RoPE, RMSNorm, tied embeddings, 4-bit weights) that answers questions about Jerry Zhu's public GitHub repos. This document carries the engine source, the model program and a run request; the weights are in the Markdown file [`../qa/weights.md`](../qa/weights.md).

**Bootstrap** (extracts the engine from this very document and runs it):

{BOOT("QA.md")}

Needs `numpy`. Flags: `-o OUT.md`, `--run-only`.

GitHub only displays this document; the extracted engine does the arithmetic. It knows nothing outside Jerry's GitHub, so other questions get a refusal.

## Weights

{F}mdrt-weights
weights: ../qa/weights.md
{F}

## The model: one step

{F}mdrt-program
x = embed(tokens, emb)
repeat 3 as l
  h = rmsnorm(x, g1.{{l}})
  qkv = linear(h, qkv.{{l}})
  a = attention_rope(qkv, heads=4)
  a = linear(a, o.{{l}})
  x = add(x, a)
  h = rmsnorm(x, g2.{{l}})
  m = gelu(linear(h, u.{{l}}))
  m = linear(m, dn.{{l}})
  x = add(x, m)
x = rmsnorm(x, gf)
logits = head(x, emb)
{F}

## Run

{F}mdrt-run
question: what is qwentpu
max_chars: 300
{F}

## The engine

{F4}python mdrt-engine
{engine("qa_engine.py")}
{F4}
'''
open(os.path.join(HERE, "GPT2.md"), "w").write(gpt2); open(os.path.join(HERE, "QA.md"), "w").write(qa); print("wrote GPT2.md, QA.md")
