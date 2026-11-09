from core import *
# constant-per-row input value c (row y => c=y), 16-wide kernel; see out for (K=[1]*? , d, bias)
N=256
src=np.tile(np.arange(N,dtype=np.uint8)[:,None],(1,16))
def run(K,d,bias):
    g=Graph(16,N); image_prim(g,src,res="src")
    g.add(f'<feConvolveMatrix in="src" order="16 1" kernelMatrix="{" ".join(map(str,K))}" divisor="{d}" bias="{bias}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true" x="0" y="0" width="1" height="{N}" RESULT')
    a,dt=render(g.svg(),16,N,"p2"); return a[:,0,0].astype(int)
for K,d,b in [([1]+[0]*15,1,0),([1]+[0]*15,3,0),([1]+[0]*15,2,0),([1]+[0]*15,7,0),([1]+[0]*15,3,0.2),([3]+[0]*15,1,0),([1,1]+[0]*14,3,0),([1,-1]+[0]*14,3,0.5),([1,-1]+[0]*14,1,0.5),([1,-1]+[0]*14,7,0.5)]:
    o=run(K,d,b); y=np.arange(N); S=y*sum(K)
    # find offset
    m=(o>0)&(o<255)
    print(K[:3],d,b,"expected S/d+255b",(S/d+255*b)[[0,1,2,3,10,100]] ,"got",o[[0,1,2,3,10,100]], "mean(got-S/d)",(o-S/d)[m].mean() if m.any() else None)
