from core import *
rng=np.random.default_rng(1)
N=64
q=rng.integers(-12,13,(N,128)); src=(q+128).astype(np.uint8)
Ks=[];g=Graph(128,N); image_prim(g,src,res="src"); outs=[]
for j in range(12):
    K=rng.integers(-3,4,128); K[-1]-=K.sum(); Ks.append(K)
    g.add(f'<feConvolveMatrix in="src" order="128 1" kernelMatrix="{" ".join(map(str,K[::-1]))}" divisor="8" bias="{128/255:.9f}" targetX="{j}" targetY="0" edgeMode="none" preserveAlpha="true" x="{j}" y="0" width="1" height="{N}" RESULT',f"c{j}"); outs.append(f"c{j}")
g.add("<feMerge RESULT>"+"".join(f'<feMergeNode in="{o}"/>' for o in outs)+"</feMerge>")
a,dt=render(g.svg(),128,N,"p5")
for j in range(12):
    v=np.floor(128+(q*Ks[j][None]).sum(1)/8+.5).clip(0,255); print(j,np.mean(v==a[:,j,0]))
