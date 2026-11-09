from core import *
N=4;src=np.tile((np.arange(128)+20).astype(np.uint8),(N,1))
for n,m,t,x in [(8,0,0,0),(8,1,0,0),(8,0,0,10),(8,1,0,10),(8,0,3,10),(8,1,3,10),(8,7,3,10),(8,0,7,10),(128,0,5,5),(128,0,100,100),(128,1,100,100)]:
    g=Graph(128,N); image_prim(g,src,res="src")
    k=" ".join("1" if i==m else "0" for i in range(n))
    g.add(f'<feConvolveMatrix in="src" order="{n} 1" kernelMatrix="{k}" divisor="1" targetX="{t}" targetY="0" edgeMode="none" preserveAlpha="true" x="{x}" y="0" width="1" height="{N}" RESULT')
    a,dt=render(g.svg(),128,N,"p3"); print("order",n,"impulse",m,"targetX",t,"x",x,"-> src idx",int(a[0,x,0])-20)
