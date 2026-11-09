from h import *
import math, json
rng=np.random.default_rng(4)
relu=[max(m,128) for m in range(256)]
def lutxml(t,inp,res):
    v=" ".join(f"{(x+0.5)/255:.8f}" for x in t); f=f'type="discrete" tableValues="{v}"'
    return f'<feComponentTransfer in="{inp}" result="{res}"><feFuncR {f}/><feFuncG {f}/><feFuncB {f}/></feComponentTransfer>'
def dense(inp,n_in,W,b,out,rl):
    """W: (n_out,n_in) ints, b ints (value units). emits convolve per neuron, assembled into a row."""
    x=""
    for j in range(W.shape[0]):
        k=" ".join(str(v) for v in W[j][::-1]); bias=(128-128*W[j].sum()+b[j])/255
        x+=f'<feConvolveMatrix in="{inp}" order="{n_in} 1" kernelMatrix="{k}" divisor="1" bias="{bias:.8f}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true" x="0" y="0" width="1" height="1" result="{out}c{j}"/>'
        x+=f'<feOffset in="{out}c{j}" dx="{j}" x="{j}" y="0" width="1" height="1" result="{out}o{j}"/>'
    x+="<feMerge"+(' result="%sm"'%out)+">"+"".join(f'<feMergeNode in="{out}o{j}"/>' for j in range(W.shape[0]))+"</feMerge>"
    if rl: x+=lutxml(relu,out+'m','H')
    return x
def graph(xv,W1,b1,W2,b2,tag):
    n=8
    f=f'<filter id="F{tag}" filterUnits="userSpaceOnUse" x="0" y="0" width="8" height="1" color-interpolation-filters="sRGB">'+dense("SourceGraphic",4,W1,b1,"a",True).replace('result="am"','result="am"')
    # layer 2 reads from last result (H after relu)
    f+=dense("H",8,W2,b2,"b",False)+'</filter>'
    return f
def np_net(x,W1,b1,W2,b2):
    h=np.clip(W1@x+b1,-128,127); h=np.maximum(h,0); y=np.clip(W2@h+b2,-128,127); return y
tests=[]
for t in range(24):
    W1=rng.integers(-2,3,(8,4));b1=rng.integers(-10,10,8);W2=rng.integers(-1,2,(4,8));b2=rng.integers(-10,10,4);x=rng.integers(-7,8,4)
    tests.append((x,W1,b1,W2,b2))
defs="";body=""
for t,(x,W1,b1,W2,b2) in enumerate(tests):
    defs+=graph(x,W1,b1,W2,b2,t)
    rects="".join(f'<rect x="{j}" y="{t}" width="1" height="1" fill="rgb({128+x[j]},{128+x[j]},{128+x[j]})"/>' for j in range(4))
    # filter region is y=0 fixed -> translate group so row is at y=0 within its own coordinate system
    body+=f'<g transform="translate(0,{t})"><g filter="url(#F{t})">'+"".join(f'<rect x="{j}" y="0" width="1" height="1" fill="rgb({128+x[j]},{128+x[j]},{128+x[j]})"/>' for j in range(4))+'</g></g>'
H=len(tests)
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="{H}"><defs>{defs}</defs><rect width="8" height="{H}" fill="#000"/>{body}</svg>'
open("mlp_4_8_4.svg","w").write(svg)
print("svg bytes",len(svg))
for img in (False,True):
    a=render(svg,8,H,"s4",img)
    ok=0;mx=0;am=0
    for t,(x,W1,b1,W2,b2) in enumerate(tests):
        y=np_net(x,W1,b1,W2,b2); got=np.array([int(a[t,j,0])-128 for j in range(4)])
        ok+=(got==y).all(); mx=max(mx,abs(got-y).max()); am+=(got.argmax()==y.argmax())
        if t<3 and not img: print(" x",x,"np",y,"svg",got)
    print("img" if img else "direct","exact",ok,"/",H,"maxerr",mx,"argmax agree",am)
