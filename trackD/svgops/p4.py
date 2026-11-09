from core import *
rng=np.random.default_rng(1)
N=256
q=rng.integers(-12,13,(N,128)); src=(q+128).astype(np.uint8)
K=rng.integers(-3,4,128); K[-1]-=K.sum(); S=(q*K[None]).sum(1)
print("S range",S.min(),S.max(),"sumK",K.sum(),"K[-1]",K[-1])
for d in (1,2,3,4,7,16,64):
  for tx in (0,5):
    g=Graph(128,N); image_prim(g,src,res="src")
    g.add(f'<feConvolveMatrix in="src" order="128 1" kernelMatrix="{" ".join(map(str,K[::-1]))}" divisor="{d}" bias="{128/255:.9f}" targetX="{tx}" targetY="0" edgeMode="none" preserveAlpha="true" x="{tx}" y="0" width="1" height="{N}" RESULT')
    a,dt=render(g.svg(),128,N,"p4"); got=a[:,tx,0].astype(int)
    v=128+S/d; m=(got>0)&(got<255)
    print(d,tx,"mean err",(got-v)[m].mean(),(got-v)[m].min(),(got-v)[m].max(),m.sum(), "exact rnd",np.mean(np.floor(v+.5)[m]==got[m]),np.mean(np.round(v)[m]==got[m]))
