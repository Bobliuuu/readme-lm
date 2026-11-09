from ops_lib import *
import core, time
rng=np.random.default_rng(4)
Hh=300; Wd=40
A=rng.integers(0,256,(Hh,Wd))
for ry in (3,16,33,64,100,127):
    c=Ctx(Wd,Hh); c.image(A.astype(np.uint8),res="a")
    c.prim(f'<feMorphology in="a" operator="dilate" radius="0 {ry}" RESULT/>')
    a,dt=core.render_pw(c.g.svg(),Wd,Hh,"p12",timeout=300); got=a[:,:,0].astype(int)
    ref=np.zeros_like(A)
    for y in range(Hh):
        lo=max(0,y-ry); hi=min(Hh,y+ry+1); ref[y]=A[lo:hi].max(0)
    ok=(got==ref); print("ry",ry,"match all",bool(ok.all()),"frac",round(float(ok.mean()),4),"rows bad",np.where(~ok.all(1))[0][:5].tolist(),"t",round(dt,2),"alpha min",a[:,:,3].min(),flush=True)
