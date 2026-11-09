#!/usr/bin/env python3
"""export_qa.py MODEL.pt README.md [d L ff] : pack 4-bit group-quantized weights into README between markers."""
import sys, io, re, base64, numpy as np, torch
from qa_model import *
pt, readme = sys.argv[1], sys.argv[2]
d, L, ff = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (128, 3, 512)
sd = torch.load(pt, map_location="cpu"); out = {"cfg": np.array([d, L, 4, ff]), "voc": np.array([ord(c) for c in VOC], dtype=np.uint8)}

def q4(w):
    w = w.float().numpy(); o, i = w.shape; g = w.reshape(o, i // G, G)
    s = np.abs(g).max(-1, keepdims=True).clip(1e-8) / 7; s = s.astype(np.float16).astype(np.float32)
    q = (np.clip(np.round(g / s), -7, 7) + 8).astype(np.uint8).reshape(o, i)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8), s[..., 0].astype(np.float16)

out["emb_q"], out["emb_s"] = q4(sd["emb"])
for l in range(L):
    for n in ("qkv", "o", "u", "dn"): out[f"{n}{l}_q"], out[f"{n}{l}_s"] = q4(sd[f"blocks.{l}.{n}.w"])
    out[f"g1_{l}"] = sd[f"blocks.{l}.g1"].numpy().astype(np.float16); out[f"g2_{l}"] = sd[f"blocks.{l}.g2"].numpy().astype(np.float16)
out["gf"] = sd["gf"].numpy().astype(np.float16)
buf = io.BytesIO(); np.savez_compressed(buf, **out); raw = buf.getvalue()
b64 = base64.b64encode(raw).decode(); lines = "\n".join(b64[i:i + 120] for i in range(0, len(b64), 120))
block = ("<!--LM-WEIGHTS-START-->\n<details><summary>The model: %d-parameter transformer, 4-bit weights, %d KB (this README <b>is</b> the model)</summary>\n\n```\n%s\n```\n\n</details>\n<!--LM-WEIGHTS-END-->"
         % (sum(v.numel() for v in sd.values()), len(b64) // 1024, lines))
r = open(readme).read()
r = (re.sub(r"<!--LM-WEIGHTS-START-->.*<!--LM-WEIGHTS-END-->", lambda m: block, r, flags=re.S) if "<!--LM-WEIGHTS-START-->" in r else r.rstrip() + "\n\n" + block + "\n")
open(readme, "w").write(r); print(len(raw), "raw;", len(b64), "b64;", len(r.encode()), "README bytes")
