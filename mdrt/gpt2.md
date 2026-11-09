# GPT-2 as an executable Markdown document

This file is a program. It contains (by reference) GPT-2 small's 124M weights stored as Markdown shards, the model's forward pass written as a declarative program, and a run request. `mdrt.py` is a Markdown renderer that executes the `mdrt-run` block: it loads the weights from the shards, interprets the `mdrt-program` block once per generated token, and renders this document with the generated tokens filled in.

**What this shows:** Markdown can be the *complete model artifact*: weights, architecture and inference procedure all live in `.md` files. **What it does not show:** GitHub's own renderer does not execute any of this. A renderer we wrote does.

## Weights and tokenizer

```mdrt-weights
manifest: ../gpt2/manifest.md
weights: ../gpt2/weights/part-*.md
tokenizer: ../gpt2/tokenizer/part-*.md
```

## Forward pass (one token step)

`tokens` is the id sequence so far; the program returns `logits` for the last position. Ops: `embed` (token + position embeddings), `layernorm`, `linear` (x @ W + b, int8 weights dequantized by the loader), `attention` (multi-head causal self-attention over a fused qkv tensor), `gelu`, `add`, `head` (tied output embedding).

```mdrt-program
x = embed(tokens, wte, wpe)
repeat 12 as l
  h = layernorm(x, h.{l}.ln1.g, h.{l}.ln1.b)
  qkv = linear(h, h.{l}.attn.w, h.{l}.attn.b)
  a = attention(qkv, heads=12)
  a = linear(a, h.{l}.proj.w, h.{l}.proj.b)
  x = add(x, a)
  h = layernorm(x, h.{l}.ln2.g, h.{l}.ln2.b)
  m = gelu(linear(h, h.{l}.fc.w, h.{l}.fc.b))
  m = linear(m, h.{l}.mp.w, h.{l}.mp.b)
  x = add(x, m)
x = layernorm(x, lnf.g, lnf.b)
logits = head(x, wte)
```

## Run

```mdrt-run
prompt: The meaning of life is
tokens: 24
temperature: 0
top_k: 40
seed: 0
```
