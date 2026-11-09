import json, sys, time
from ops_lib import *
sys.path.insert(0,"/Users/jhome/readme-lm/trackD/qat8")
import q8_ref as Q
import pipeline
_PL={}
def att_params(T,l=0):
    if T not in _PL: _PL[T]=pipeline.Plan(T)
    return _PL[T].att_params(l)
def run_att(T, l=0, tag="o7", seed=7):
    P=att_params(T,l); rng=np.random.default_rng(seed)
    Qm=np.clip(rng.normal(P["zq"],25,(T,128)).round(),0,255).astype(np.int64); Km=np.clip(rng.normal(P["zk"],25,(T,128)).round(),0,255).astype(np.int64)
    Vm=np.clip(rng.normal(P["zv"],30,(T,128)).round(),0,255).astype(np.int64)
    W=WA+64; H=attn_H(T)
    c=Ctx(W,H)
    def put(a,name):
        arr=np.zeros((T,W),np.uint8); arr[:,ACOLS]=a; c.image(arr,res=name+"0"); return c.merge([name+"0"],h=T)
    q=put(Qm,"q"); k=put(Km,"k"); v=put(Vm,"v")
    out=attention(c,q,k,v,P,T)
    svg=c.g.svg()
    svg=svg.replace(f'width="{W}" height="{H}" viewBox="0 0 {W} {H}"',f'width="{WA}" height="{T}" viewBox="0 0 {WA} {T}"',1)
    t0=time.time(); a,dt=render(svg,WA,T,tag,timeout=250); got=a[:,ACOLS,0].astype(np.int64)
    ref=attention_ref(Qm,Km,Vm,P,T)
    return dict(op=f"causal attention T={T} (4 heads x 32)",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2),maxerr=int(np.abs(got-ref).max()),sat=float(((ref==0)|(ref==255)).mean()),ref_std=float(ref.std()))
if __name__=="__main__":
    T=int(sys.argv[1]) if len(sys.argv)>1 else 8
    r=run_att(T); print(r,flush=True)
