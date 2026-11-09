from svg_eval import *
import core
T=16; p=make_prompt(T,"what is x"); ids=mk(p); pl=build(T,[ids]); tr=forward_ref(pl,ids)
svg,view,c=build_svg(pl,ids,stop="L0.q"); a,dt=core.render_pw(svg,128,16,"dq",keep=True)
got=a[:,:,0].astype(int); ref=tr["L0.q"]
print("dt",dt,"mism",(got!=ref).sum()); print(got[0,:12],ref[0,:12]); print((got-ref)[0,:12]); print("alpha",a[:,:,3].min())
n1=tr["L0.n1"]; print("n1 row0",n1[0,:10], "E0",pl.lins["L0.q"]["E"][0,:10],"d",pl.lins["L0.q"]["d"])
