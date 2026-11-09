import json
from ops_lib import *
import sys; sys.path.insert(0,"/Users/jhome/readme-lm/trackD/qat8")
import q8_ref as Q
sp=json.loads(str(__import__("numpy").load("/Users/jhome/readme-lm/trackD/qat8/q8_params.npz",allow_pickle=False)["spec"]))
Zp=np.load("/Users/jhome/readme-lm/trackD/qat8/q8_params.npz",allow_pickle=False)
def norm_params(n, inn, gain, S1=None, xmax=40):
    sx,zx=sp[inn+".s"],sp[inn+".z"]; ssq=sp[n+".sq.s"]; sy,zy=sp[n+".s"],sp[n+".z"]; sr,zr=sp[n+".sr"],sp[n+".zr"]; C=int(sp[n+".C"]); mm=sp[n+".mm"]
    e=int(round(mm*1000)); d=1000; mm2=e/d
    P=dict(sq=Q.lut_sq(sx,zx,ssq),rsq=Q.lut_rsq(C,mm2,ssq,sr,zr),C=C,e=e,d=d,zx=int(zx),zr=int(zr),zy=int(zy),S2=255)
    S1=S1 or 600; P["S1"]=S1
    P["gq"]=np.clip(np.round(P["S2"]*S1*gain*sx*sr/sy),0,255).astype(np.int64)
    return P
def run_norm(P,H=8,xsd=30,tag="o3"):
    rng=np.random.default_rng(5); n=128
    X=np.clip(rng.normal(P["zx"],xsd,(H,n)).round(),0,255).astype(np.int64)
    xc0=0; W=n+2
    c=Ctx(W,H); arr=np.zeros((H,W),np.uint8); arr[:,:n]=X; c.image(arr,res="src"); out=c.merge(["src"])
    res=rmsnorm(c,out,xc0,n,P)
    svg=c.g.svg(); a,dt=render(svg,W,H,tag); got=a[:,:n,0].astype(np.int64); ref=rmsnorm_ref(X,P)
    return dict(op="RMSNorm 128",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2),sat=float(((ref==0)|(ref==255)).mean())),got,ref
def run_resadd(tag="o4"):
    rng=np.random.default_rng(6); H,n=8,128
    a=rng.integers(0,256,(H,n)); b=rng.integers(0,256,(H,n))
    out=[]
    for (pa,pb,za,zb,zo) in [(256,256,128,128,128),(37,230,128,124,128),(190,200,128,128,120)]:
        W=n+2; c=Ctx(W,H); A=np.zeros((H,W),np.uint8); A[:,:n]=a; B=A.copy(); B[:,:n]=b
        c.image(A,res="A"); c.image(B,res="B"); r=resadd(c,"A","B",pa,pb,za,zb,zo)
        svg=c.g.svg(); g,dt=render(svg,W,H,tag); got=g[:,:n,0].astype(np.int64); ref=resadd_ref(a,b,pa,pb,za,zb,zo)
        out.append(dict(op=f"residual add pa={pa} pb={pb} za={za} zb={zb} zo={zo}",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2)))
    return out
if __name__=="__main__":
    res=[]
    P=norm_params("L0.n1","emb",Zp["g1_0"].astype(np.float64))
    print("gq",P["gq"][:8],"e",P["e"])
    r,got,ref=run_norm(P); print(r,flush=True); res.append(r)
    for r in run_resadd(): print(r,flush=True); res.append(r)
    json.dump(res,open("results_o3o4.json","w"),indent=1)
