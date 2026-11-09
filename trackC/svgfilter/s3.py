from h import *
import math, json
rng=np.random.default_rng(3)
ins=list(range(256))
relu=[max(m,128) for m in range(256)]            # centered ReLU on offset-128 encoding
gelu=[int(np.clip(round(128+0.5*(m-128)*(1+math.erf((m-128)/6/math.sqrt(2)))*1),0,255)) for m in range(256)]
perm=list(rng.permutation(256))
def lut(t,kind="discrete"):
    v=" ".join(f"{(x+0.5)/255:.8f}" for x in t)
    f=f'type="{kind}" tableValues="{v}"'
    return f'<feComponentTransfer><feFuncR {f}/><feFuncG {f}/><feFuncB {f}/></feComponentTransfer>'
def fl(m): return f'<feFlood flood-color="rgb({m},{m},{m})"/>'
res={}
for name,t in (("relu",relu),("gelu_lut",gelu),("random_perm",perm)):
  for kind in ("discrete","table"):
    for img in (False,True):
        r=grid([fl(m)+lut(t,kind) for m in ins],cols=256,name="s3",img=img)[:,0].astype(int)
        ex=(r==np.array(t)).mean(); me=abs(r-np.array(t)).max()
        res[f"{name}/{kind}/{'img' if img else 'direct'}"]=(float(ex),int(me)); print(name,kind,"img" if img else "direct","exact frac",round(ex,4),"maxerr",me)
# clamp-based ReLU with colorMatrix (no LUT): relu on 0-based encoding is just clamp: y=clamp(a*x+b)
r=grid([fl(m)+'<feColorMatrix type="matrix" values="1 0 0 0 -0.5 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0"/>' for m in ins],cols=256,name="s3c")[:,0].astype(int)
print("clamp relu (x-127.5): maxerr",abs(r-np.clip(np.array(ins)-127.5,0,255).round()).max())
json.dump(res,open("s3_results.json","w"))
