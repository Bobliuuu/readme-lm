import time
from ops_lib import *
import core
rng=np.random.default_rng(3)
H=16; n=128; Wd=n+260
Wr=rng.normal(0,1,(n,n)); Wr=Wr*(0.1/Wr.std()); E,d=quantize_weights(Wr,127)
for depth in (1,2,3,4):
    c=Ctx(Wd,H); arr=np.zeros((H,Wd),np.uint8); arr[:,:n]=np.clip(rng.normal(128,25,(H,n)).round(),0,255)
    c.image(arr,res="s0"); x=c.merge(["s0"])
    for k in range(depth):
        x=linear(c,x,[(0,n)],E,d,128,128,np.zeros(n,np.int64),list(range(n)))
    c.prim(f'<feComponentTransfer in="{x}"/>')
    svg=c.g.svg(); a,dt=core.render_pw(svg,Wd,H,"depth",timeout=900)
    print("linear depth",depth,"prims",c.g.count(),"render_s",round(dt,2),flush=True)
