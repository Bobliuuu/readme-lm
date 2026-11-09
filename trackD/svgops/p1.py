from core import *
rng=np.random.default_rng(1)
N=256
q=rng.integers(-12,13,(N,128)); src=(q+128).astype(np.uint8)
g=Graph(128,N); image_prim(g,src,res="src")
tests=[];outs=[]
cfg=[(1,0),(2,0),(3,0),(4,0),(7,0),(16,0),(16,0.1),(10,-0.05),(32,0),(5,0.25),(9,0),(64,0)]
for j,(d,bias) in enumerate(cfg):
    K=rng.integers(-3,4,128); K[-1]-=K.sum()
    k=" ".join(map(str,K[::-1]))
    g.add(f'<feConvolveMatrix in="src" order="128 1" kernelMatrix="{k}" divisor="{d}" bias="{bias+128/255:.9f}" targetX="{j}" targetY="0" edgeMode="none" preserveAlpha="true" x="{j}" y="0" width="1" height="{N}" RESULT',f"c{j}")
    tests.append((K,d,bias)); outs.append(f"c{j}")
g.add("<feMerge RESULT>"+"".join(f'<feMergeNode in="{o}"/>' for o in outs)+"</feMerge>")
a,dt=render(g.svg(),128,N,"p1")
for j,(K,d,bias) in enumerate(tests):
    s=(q*K[None]).sum(1); got=a[:,j,0].astype(int)
    v=128+s/d+bias*255
    res={m:np.mean(np.clip(f(v),0,255)==got) for m,f in(("round_half_up",lambda x:np.floor(x+0.5)),("floor",np.floor),("ceil",np.ceil),("trunc",np.trunc))}
    print(d,bias,"range",s.min(),s.max(),{k:round(float(v),3) for k,v in res.items()}, "maxdiff vs v",np.abs(got-v).max())
j=11;K,d,bias=tests[j];s=(q*K[None]).sum(1)
print(a[:6,j,:], (128+s/d)[:6]); print(a[:3,0:14,0])
for j in (2,4,5,6,8,10):
    K,d,bias=tests[j];s=(q*K[None]).sum(1);got=a[:,j,0].astype(int);v=128+s/d+bias*255
    m=(got>0)&(got<255)
    print(d,bias,"mean err",(got-v)[m].mean(),"min/max err",(got-v)[m].min(),(got-v)[m].max(),"n",m.sum())
