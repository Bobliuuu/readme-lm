import json, sys
from ops_lib import *
sys.path.insert(0,"/Users/jhome/readme-lm/trackD/qat8")
import q8_ref as Q
def run_embed(tag="o5e"):
    rng=np.random.default_rng(1)
    m=Q.Q8Ref("/Users/jhome/readme-lm/trackD/qat8/q8_params.npz",maxT=64)
    emb=m.emb_tab; H=16; ids=rng.integers(0,Q.V,H)
    W=140; c=Ctx(W,H); out=embed_lookup(c,ids,emb,0)
    svg=c.g.svg(); a,dt=render(svg,W,H,tag); got=a[:,:128,0].astype(np.int64); ref=embed_ref(ids,emb)
    return dict(op="token embedding lookup (86x128, per-channel LUT on id pixel)",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2))
def run_argmax(tag="o5a"):
    rng=np.random.default_rng(2); H=8; V=86
    L=rng.integers(0,256,(H,V)); L[1]=7; L[2,40]=250; L[2,10]=250; L[3,85]=255; L[3,0]=255; L[4]=0; L[4,60]=1
    W=300; c=Ctx(W,H); arr=np.zeros((H,W),np.uint8); arr[:,:V]=L; c.image(arr,res="src"); lg=c.merge(["src"])
    X=argmax_row(c,lg,V)
    svg=c.g.svg(); a,dt=render(svg,W,H,tag); got=a[:,127,0].astype(np.int64); ref=argmax_ref(L)
    return dict(op="argmax tournament over 86 logits (feMorphology dilate/erode, id in R)",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2),got=got.tolist(),ref=ref.tolist())
def run_rope(tag="o6"):
    rng=np.random.default_rng(3); H=16; n=128; R=100
    f=1.0/(10000**(np.arange(0,32,2)/32)); ang=np.arange(H)[:,None]*f[None]          # (H,16)
    cosc=np.round(R*np.cos(ang)).astype(np.int64); sinc=np.round(R*np.sin(ang)).astype(np.int64)
    cosc=np.tile(cosc,(1,4)); sinc=np.tile(sinc,(1,4))          # 4 heads
    X=np.clip(rng.normal(128,30,(H,n)).round(),0,255).astype(np.int64)
    W=n+4; c=Ctx(W,H); arr=np.zeros((H,W),np.uint8); arr[:,:n]=X; c.image(arr,res="src"); x=c.merge(["src"])
    out=rope_apply(c,x,list(range(n)),cosc,sinc,128,128,R)
    svg=c.g.svg(); a,dt=render(svg,W,H,tag); got=a[:,:n,0].astype(np.int64); ref=rope_ref(X,cosc,sinc,128,128,R)
    # float sanity
    xf=(X-128).astype(float); xe,xo=xf[:,0::2],xf[:,1::2]; ce,se=cosc/R,sinc/R
    fl=np.empty_like(xf); fl[:,0::2]=xe*ce-xo*se; fl[:,1::2]=xe*se+xo*ce
    return dict(op="RoPE (per-position cos/sin const rasters, 128 ch = 4 heads x 16 pairs)",PASS=bool((got==ref).all()),mismatches=int((got!=ref).sum()),of=int(ref.size),prims=c.g.count(),svg_bytes=len(svg),render_s=round(dt,2),max_err_vs_float=float(np.abs(ref-128-fl).max()))
if __name__=="__main__":
    res=[]
    for f in (run_embed,run_argmax,run_rope):
        r=f(); print(r,flush=True); res.append(r)
    json.dump(res,open("results_o56.json","w"),indent=1)
