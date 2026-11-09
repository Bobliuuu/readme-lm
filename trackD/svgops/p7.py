from core import *
rng=np.random.default_rng(5)
N=128
q=rng.integers(-12,13,(N,128)); src=np.concatenate([(q+128).astype(np.uint8),np.zeros((N,2),np.uint8)],1)
def t(K,d,lo=-127,hi=128):
    # semantic kernel K (applied as sum K_i*pix_i); append guards at idx 128,129 with values lo,hi
    Kf=np.concatenate([K,[lo,hi]]); n=len(Kf)
    g=Graph(130,N); image_prim(g,src,res="src")
    g.add(f'<feConvolveMatrix in="src" order="{n} 1" kernelMatrix="{" ".join(map(str,Kf[::-1]))}" divisor="{d}" bias="{(128-128*K.sum()/d)/255:.9f}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true" x="0" y="0" width="1" height="{N}" RESULT')
    a,dt=render(g.svg(),130,N,"q1"); got=a[:,0,0].astype(int); S=(q*K[None]).sum(1); v=128+S/d
    m=(got>0)&(got<255)
    return round(float(np.mean(np.floor(v+.5)[m]==got[m])),3), int(m.sum())
for lo,hi in [(-1,1),(-3,3),(-7,7),(-15,15),(-31,31),(-60,60),(-127,127)]:
    K=rng.integers(lo,hi+1,128); print((lo,hi),"sumK",K.sum(),t(K,max(8,hi*12*8//80)))
