import json, numpy as np, torch
from q8_common import *
def export(m, path):
    import copy; m = copy.deepcopy(m).cpu(); d = {}
    def q4(w):
        w = w.detach().float(); o, i = w.shape; g = w.view(o, i // GRP, GRP)
        s = (g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7).half().float()
        c = torch.clamp(torch.floor(g / s + 0.5), -7, 7).to(torch.int8).view(o, i); return c.numpy(), s[..., 0].half().numpy()
    d["emb_c"], d["emb_s"] = q4(m.emb)
    for l in range(N_LAYER):
        for n in ("qkv", "o", "u", "dn"): d[f"{n}{l}_c"], d[f"{n}{l}_s"] = q4(m.blk[l][n].weight)
        d[f"g1_{l}"] = m.g1[l].detach().numpy().astype(np.float16); d[f"g2_{l}"] = m.g2[l].detach().numpy().astype(np.float16)
    d["gf"] = m.gf.detach().numpy().astype(np.float16); d["spec"] = np.array(json.dumps(m.spec))
    np.savez(path, **d)
