"""Runs every op test (bit-exact vs integer numpy reference) and records PASS/FAIL, primitive count, SVG bytes, render seconds -> results_all.json"""
import json, sys, time, importlib
res = []
def add(r):
    if isinstance(r, list): [add(x) for x in r]; return
    r = {k: v for k, v in r.items() if k not in ("got", "ref", "detail")}; print(r, flush=True); res.append(r); json.dump(res, open("results_all.json", "w"), indent=1, default=str)
import t_o1, t_o2, t_o3, t_o56, t_o7, t_o8, t_gather
from ops_lib import *
import core
for name, no, ni in [("o0", 128, 128), ("qkv0", 384, 128), ("u0", 512, 128), ("dn0", 128, 512)]:
    add(t_o1.run_linear(name, no, ni, tag="a_o1_" + name))
r = t_o1.run_linear("o1", 128, 128, tag="a_o1_img", img=True); r["op"] += " via <img>"; add(r)
tabs = {"gelu": t_o2.Q.lut_gelu(t_o2.sp["L0.u.s"], t_o2.sp["L0.u.z"], t_o2.sp["L0.g.s"], t_o2.sp["L0.g.z"]), "exp": t_o2.Q.lut_exp(t_o2.sp["L0.f.s"] if "L0.f.s" in t_o2.sp else t_o2.sp["L0.sc.s"]),
        "reciprocal": t_o2.Q.lut_rec(t_o2.sp["L0.D"]), "rsqrt": t_o2.Q.lut_rsq(t_o2.sp["L0.n1.C"], t_o2.sp["L0.n1.mm"], t_o2.sp["L0.n1.sq.s"], t_o2.sp["L0.n1.sr"], t_o2.sp["L0.n1.zr"]),
        "square": t_o2.Q.lut_sq(t_o2.sp["emb.s"], 128, t_o2.sp["L0.n1.sq.s"])}
for k, t in tabs.items(): add(t_o2.run_lut(k, t, "a_o2_" + k))
P = t_o3.norm_params("L0.n1", "emb", t_o3.Zp["g1_0"].astype(float)); r, _, _ = t_o3.run_norm(P, tag="a_o3"); add(r)
add(t_o3.run_resadd("a_o4"))
for f in (t_o56.run_embed, t_o56.run_argmax, t_o56.run_rope): add(f())
add(t_o8.run_glyph())
for T in (8, 16, 32, 64): add(t_o7.run_att(T, tag=f"a_o7_{T}"))
for T in (8, 16, 32, 64): add(dict(op=f"row gathers (stretch + compact) via feDisplacementMap T={T}", **{k: v for k, v in t_gather.run(T).items() if k in ("T","m","P","prims")}, PASS=bool(t_gather.run(T)["stretch_ok"] and t_gather.run(T)["compact_roundtrip_ok"])))
print("ALL DONE", sum(1 for r in res if r.get("PASS")), "/", len(res))
