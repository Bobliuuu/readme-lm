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

**Prompt:** The meaning of life is

**Generated (by `mdrt`, 24 tokens, temperature 0.0):**

> The meaning of life is not the same as the meaning of death. ⏎  ⏎ The meaning of life is not the same as the meaning of death

| step | chosen token | p | top-3 |
|---|---|---|---|
| 0 | ` not` | 9.3% | ` not` 9% · ` to` 7% · ` the` 6% |
| 1 | ` the` | 5.7% | ` the` 6% · ` to` 4% · ` determined` 4% |
| 2 | ` same` | 25.1% | ` same` 25% · ` end` 2% · ` only` 2% |
| 3 | ` as` | 48.7% | ` as` 49% · ` for` 7% · ` in` 6% |
| 4 | ` the` | 32.8% | ` the` 33% · ` that` 7% · ` life` 5% |
| 5 | ` meaning` | 33.5% | ` meaning` 33% · ` definition` 2% · ` life` 1% |
| 6 | ` of` | 97.9% | ` of` 98% · ` and` 0% · ` or` 0% |
| 7 | ` death` | 19.2% | ` death` 19% · ` life` 14% · ` the` 10% |
| 8 | `.` | 32.7% | `.` 33% · `,` 14% · `;` 12% |
| 9 | `⏎` | 8.8% | `⏎` 9% · ` It` 7% · ` The` 7% |
| 10 | `⏎` | 97.1% | `⏎` 97% · `The` 0% · `This` 0% |
| 11 | `The` | 11.3% | `The` 11% · `"` 5% · `In` 3% |
| 12 | ` meaning` | 7.8% | ` meaning` 8% · ` word` 2% · ` Bible` 1% |
| 13 | ` of` | 95.0% | ` of` 95% · ` is` 2% · ` and` 0% |
| 14 | ` life` | 78.8% | ` life` 79% · ` death` 8% · ` the` 3% |
| 15 | ` is` | 74.2% | ` is` 74% · ` depends` 2% · ` has` 1% |
| 16 | ` not` | 37.6% | ` not` 38% · ` the` 12% · ` that` 8% |
| 17 | ` the` | 62.3% | ` the` 62% · ` a` 3% · ` that` 2% |
| 18 | ` same` | 97.7% | ` same` 98% · ` meaning` 0% · ` opposite` 0% |
| 19 | ` as` | 98.9% | ` as` 99% · `.` 0% · ` thing` 0% |
| 20 | ` the` | 75.2% | ` the` 75% · ` death` 13% · ` life` 3% |
| 21 | ` meaning` | 93.4% | ` meaning` 93% · ` meanings` 1% · ` life` 1% |
| 22 | ` of` | 99.9% | ` of` 100% · `,` 0% · ` and` 0% |
| 23 | ` death` | 96.4% | ` death` 96% · ` life` 1% · ` the` 0% |

