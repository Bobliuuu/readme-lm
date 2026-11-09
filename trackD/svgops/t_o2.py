import json
from ops_lib import *
import sys; sys.path.insert(0,"/Users/jhome/readme-lm/trackD/qat8")
import q8_ref as Q
sp=json.loads(str(__import__("numpy").load("/Users/jhome/readme-lm/trackD/qat8/q8_params.npz",allow_pickle=False)["spec"]))
def run_lut(name, table, tag):
    rows=4; W=256
    rng=np.random.default_rng(0)
    U=np.stack([np.arange(256)]+[rng.permutation(256) for _ in range(rows-1)])
    c=Ctx(W,rows); c.image(U.astype(np.uint8),res="src"); out=c.lut("src",table)
    svg=c.g.svg(); a,dt=render(svg,W,rows,tag); got=a[:,:,0].astype(int); ref=lut_ref(U,table)
    return dict(op=f"LUT {name}",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2))
if __name__=="__main__":
    res=[]
    tabs={"gelu":Q.lut_gelu(sp["L0.u.s"],sp["L0.u.z"],sp["L0.g.s"],sp["L0.g.z"]),
          "exp":Q.lut_exp(sp["L0.sc.s"]),"reciprocal":Q.lut_rec(sp["L0.D"]),
          "rsqrt":Q.lut_rsq(sp["L0.n1.C"],sp["L0.n1.mm"],sp["L0.n1.sq.s"],sp["L0.n1.sr"],sp["L0.n1.zr"]),
          "square":Q.lut_sq(sp["emb.s"],128,sp["L0.n1.sq.s"])}
    for k,t in tabs.items():
        r=run_lut(k,t,"o2_"+k); print(r,flush=True); res.append(r)
    json.dump(res,open("results_o2.json","w"),indent=1)
