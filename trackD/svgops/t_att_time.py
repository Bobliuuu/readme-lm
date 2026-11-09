import sys,time
from t_o7 import *
import core
T=int(sys.argv[1])
P=att_params(T); rng=np.random.default_rng(7)
Qm=np.clip(rng.normal(P["zq"],25,(T,128)).round(),0,255).astype(np.int64)
for upto in sys.argv[2].split(","):
    W=200;H=attn_H(T);c=Ctx(W,H)
    def put(a,name):
        arr=np.zeros((T,W),np.uint8); arr[:,:128]=a; c.image(arr,res=name+"0"); return c.merge([name+"0"],h=T)
    q=put(Qm,"q");k=put(Qm,"k");v=put(Qm,"v")
    out=attention(c,q,k,v,P,T,upto=None if upto=="all" else upto)
    c.prim(f'<feComponentTransfer in="{out}"/>')
    svg=c.g.svg().replace(f'width="{W}" height="{H}" viewBox="0 0 {W} {H}"','width="128" height="16" viewBox="0 0 128 16"',1)
    a,dt=core.render_pw(svg,128,16,"tt"); print("T",T,"upto",upto,"prims",c.g.count(),"render_s",round(dt,2),flush=True)
