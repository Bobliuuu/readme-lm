from h import *
import time,sys
rng=np.random.default_rng(7)
def layer(nin,nout,T,scale):
    W=rng.integers(-7,8,(nout,nin)); X=128+rng.integers(-8,9,(T,nin))
    f='<filter id="f" filterUnits="userSpaceOnUse" x="0" y="0" width="%d" height="%d" color-interpolation-filters="sRGB">'%(nout,T)
    for j in range(nout):
        ks=" ".join(str(int(v)) if scale==1 else f"{v*scale:.6g}" for v in W[j][::-1]); b=(128-128*W[j].sum()*scale)/255
        f+=f'<feConvolveMatrix in="SourceGraphic" order="{nin} 1" kernelMatrix="{ks}" divisor="1" bias="{b:.6f}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true" x="0" y="0" width="1" height="{T}" result="c{j}"/><feOffset in="c{j}" dx="{j}" x="{j}" y="0" width="1" height="{T}" result="o{j}"/>'
    f+="<feMerge>"+"".join(f'<feMergeNode in="o{j}"/>' for j in range(nout))+"</feMerge></filter>"
    rects="".join(f'<rect x="{j}" y="{t}" width="1" height="1" fill="rgb({X[t][j]},{X[t][j]},{X[t][j]})"/>' for t in range(T) for j in range(nin))
    w=max(nin,nout)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{T}" shape-rendering="crispEdges"><defs>{f}</defs><rect width="{w}" height="{T}" fill="#000"/><g filter="url(#f)">{rects}</g></svg>'
    ex=np.clip(np.round(128+(X-128)@W.T*scale),0,255)
    t0=time.time(); a=render(svg,w,T,"s5s"); dt=time.time()-t0
    got=a[:T,:nout,0].astype(int)
    return len(svg),dt,np.abs(got-ex).max(),(got==ex).mean()
for nin,nout,T,sc in [(128,384,16,1/64),(128,384,64,1/64),(128,512,160,1/64)]:
    print(nin,nout,"T",T,"svg bytes,time,maxerr,frac exact",layer(nin,nout,T,sc))
