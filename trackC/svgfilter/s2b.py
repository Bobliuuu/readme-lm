from h import *
rng=np.random.default_rng(2)
def build(tests, bias_mode):
    defs="";body=""
    for t,(W,x) in enumerate(tests):
        for i in range(4):
            y=4*t+i; fid=f"f{t}_{i}"
            ks=" ".join(f"{v}" for v in W[i][::-1]); b=(128-128*W[i].sum())/255
            defs+=f'<filter id="{fid}" filterUnits="userSpaceOnUse" x="0" y="{y}" width="4" height="1" color-interpolation-filters="sRGB"><feConvolveMatrix order="4 1" kernelMatrix="{ks}" divisor="1" bias="{b:.8f}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true"/></filter>'
            rects="".join(f'<rect x="{j}" y="{y}" width="1" height="1" fill="rgb({128+x[j]},0,0)"/>' for j in range(4))
            body+=f'<g filter="url(#{fid})">{rects}</g>'
    H=4*len(tests)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="{H}"><defs>{defs}</defs><rect width="4" height="{H}" fill="#000"/>{body}</svg>'
tests=[(rng.integers(-3,4,(4,4)),rng.integers(-7,8,4)) for _ in range(30)]
for img in (False,True):
    a=render(build(tests,0),4,120,"s2b",img)
    ok=0;errs=[]
    for t,(W,x) in enumerate(tests):
        y=W@x; got=np.array([int(a[4*t+i,0,0])-128 for i in range(4)])
        ok+=(got==y).all(); errs.append(abs(got-y).max())
        if t<3: print(" y",y,"got",got)
    print("img" if img else "direct","convolve exact",ok,"/",len(tests),"maxerr",max(errs))
