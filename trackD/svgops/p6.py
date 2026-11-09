from core import *
rng=np.random.default_rng(5)
N=128
q=rng.integers(-12,13,(N,128)); src=(q+128).astype(np.uint8)
def t(K,d=8,n=128):
    g=Graph(128,N); image_prim(g,src,res="src")
    g.add(f'<feConvolveMatrix in="src" order="{n} 1" kernelMatrix="{" ".join(map(str,K[::-1]))}" divisor="{d}" bias="{(128-128*K.sum()/d)/255:.9f}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true" x="0" y="0" width="1" height="{N}" RESULT')
    a,dt=render(g.svg(),128,N,"q1"); got=a[:,0,0].astype(int); S=(q[:,:n]*K[None]).sum(1); v=128+S/d
    m=(got>0)&(got<255)
    return np.round((got-v)[m].mean(),2) if m.sum() else None, round(float(np.mean(np.floor(v+.5)[m]==got[m])),3), int(m.sum())
for lo,hi in [(-1,1),(-3,3),(-7,7),(-7,8),(-8,7),(-15,15),(-30,30),(-31,31),(-127,127),(-60,60)]:
    for n in (128,):
        K=rng.integers(lo,hi+1,n); K[0]=lo;K[1]=hi
        print((lo,hi),n,"sumK",K.sum(),t(K,d=max(8,(hi*12*8)//80)))
