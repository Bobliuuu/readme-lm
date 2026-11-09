import time
from ops_lib import *
import core
for n in (8,16,32,64,128):
    rng=np.random.default_rng(2); H=8; V=min(n,86)
    L=rng.integers(0,256,(H,V))
    W=200; c=Ctx(W,H); arr=np.zeros((H,W),np.uint8); arr[:,:V]=L; c.image(arr,res="src"); lg=c.merge(["src"])
    X=argmax_row(c,lg,V,W_pow2=n)
    svg=c.g.svg(); a,dt=core.render_pw(svg,W,H,"diam")
    got=a[:,0,1].astype(int); ref=L.argmax(1)
    print("rounds",int(np.log2(n)),"prims",c.g.count(),"render_s",round(dt,2),"ok",(got==ref).all(),flush=True)
