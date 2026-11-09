from core import *
rng=np.random.default_rng(2)
W,N=128,128
A=rng.integers(0,256,(N,W)).astype(np.uint8); B=rng.integers(0,256,(N,W)).astype(np.uint8)
def run(k1,k2,k3,k4,name="p8"):
    g=Graph(W,N); image_prim(g,A,res="A"); image_prim(g,B,res="B")
    g.add(f'<feComposite in="A" in2="B" operator="arithmetic" k1="{k1:.12g}" k2="{k2:.12g}" k3="{k3:.12g}" k4="{k4:.12g}" RESULT')
    a,dt=render(g.svg(),W,N,name); return a
def ref(k1,k2,k3,k4):
    a=A/255.;b=B/255.; r=k1*a*b+k2*a+k3*b+k4; return np.clip(r,0,1)*255
for ks in [(1,0,0,0),(0,1,1,-128/255),(0,0.5,0.5,0),(0,1,-1,128/255),(0,1,-1,0.5),(0,0.25,0.25,0.25),(255/16,-128/16,-128/16,128/255+128*128/(255*16)),(0,2,0,-1)]:
    a=run(*ks); r=ref(*ks)
    got=a[:,:,0].astype(float)
    res={m:float(np.mean(f(r)==got)) for m,f in [("rnd_half_up",lambda x:np.floor(x+.5)),("floor",np.floor),("trunc_even",np.rint)]}
    print([round(k,4) for k in ks],res,"alpha min",a[:,:,3].min(),"maxerr",np.abs(np.floor(r+.5)-got).max())
print("---- float32 emulation")
f32=np.float32
def emu(k1,k2,k3,k4,variant):
    a=A.astype(f32)/f32(255); b=B.astype(f32)/f32(255); k1,k2,k3,k4=map(f32,(k1,k2,k3,k4))
    if variant==0: r=k1*a*b+k2*a+k3*b+k4
    elif variant==1: r=k4+k3*b+k2*a+k1*a*b
    elif variant==2: r=(k1*a*b+k2*a)+(k3*b+k4)
    elif variant==3: r=k1*(a*b)+k2*a+k3*b+k4
    r=np.clip(r,0,1).astype(f32)*f32(255)
    return np.floor(r+f32(0.5))
for ks in [(0,0.5,0.5,0),(15.9375,-8,-8,4.5176),(0,1,1,-128/255)]:
    a=run(*ks); got=a[:,:,0].astype(float)
    print(ks,[float(np.mean(emu(*ks,v)==got)) for v in range(4)])
