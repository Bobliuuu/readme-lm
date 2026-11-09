#!/usr/bin/env python3
"""convert.py MODEL.safetensors VOCAB.json MERGES.txt OUTDIR
Quantize GPT-2 small to int8 (per-output-channel) and write it as sharded Markdown files
(base64 in code fences). Weights -> weights/part-NNNN.md, tokenizer -> tokenizer/part-NNNN.md, manifest.md."""
import sys, os, json, zlib, base64, numpy as np
from safetensors.numpy import load_file
src, vocab, merges, out = sys.argv[1:5]
SHARD = 440_000  # base64 chars per shard (< GitHub's ~500 KiB markdown render limit)
W = load_file(src); W = {k.replace("transformer.", ""): v for k, v in W.items()}
order, blobs, man = [], [], []
def add(name, arr, quant=None):
    arr = arr.astype(np.float32)
    if quant is None:
        b = arr.astype("<f4").tobytes(); man.append(dict(name=name, shape=arr.shape, q=None, nbytes=len(b))); blobs.append(b); return
    ax = quant                                    # axis reduced for the scale
    s = np.abs(arr).max(ax, keepdims=True).clip(1e-8) / 127
    q = np.clip(np.round(arr / s), -127, 127).astype(np.int8); s = s.squeeze(ax).astype("<f2")
    b = q.tobytes() + s.tobytes(); man.append(dict(name=name, shape=arr.shape, q=ax, nbytes=len(b))); blobs.append(b)
add("wte", W["wte.weight"], 1); add("wpe", W["wpe.weight"], 1)
for i in range(12):
    p = f"h.{i}."
    add(p + "ln1.g", W[p + "ln_1.weight"]); add(p + "ln1.b", W[p + "ln_1.bias"])
    add(p + "attn.w", W[p + "attn.c_attn.weight"], 0); add(p + "attn.b", W[p + "attn.c_attn.bias"])
    add(p + "proj.w", W[p + "attn.c_proj.weight"], 0); add(p + "proj.b", W[p + "attn.c_proj.bias"])
    add(p + "ln2.g", W[p + "ln_2.weight"]); add(p + "ln2.b", W[p + "ln_2.bias"])
    add(p + "fc.w", W[p + "mlp.c_fc.weight"], 0); add(p + "fc.b", W[p + "mlp.c_fc.bias"])
    add(p + "mp.w", W[p + "mlp.c_proj.weight"], 0); add(p + "mp.b", W[p + "mlp.c_proj.bias"])
add("lnf.g", W["ln_f.weight"]); add("lnf.b", W["ln_f.bias"])

def write(kind, raw, title):
    z = zlib.compress(raw, 9); b64 = base64.b64encode(z).decode(); d = os.path.join(out, kind); os.makedirs(d, exist_ok=True)
    n = (len(b64) + SHARD - 1) // SHARD
    for i in range(n):
        chunk = b64[i * SHARD:(i + 1) * SHARD]; lines = "\n".join(chunk[j:j + 120] for j in range(0, len(chunk), 120))
        open(f"{d}/part-{i:04d}.md", "w").write(f"# {title} shard {i + 1}/{n}\n\nBase64 of a zlib stream; shards concatenate in order.\n\n```\n{lines}\n```\n")
    return n, len(raw), len(z)

wn, wraw, wz = write("weights", b"".join(blobs), "GPT-2 small (124M) int8 weights")
tok = json.dumps({"vocab": json.load(open(vocab)), "merges": open(merges).read().split("\n")[1:-1]}).encode()
tn, traw, tz = write("tokenizer", tok, "GPT-2 BPE tokenizer")
open(f"{out}/manifest.md", "w").write("# GPT-2 manifest\n\n```json\n" + json.dumps(dict(weights_shards=wn, tokenizer_shards=tn, tensors=man)) + "\n```\n")
print(f"weights: {wn} shards, {wraw/1e6:.1f} MB raw -> {wz/1e6:.1f} MB zlib; tokenizer: {tn} shards {traw/1e6:.2f} MB -> {tz/1e6:.2f} MB")
