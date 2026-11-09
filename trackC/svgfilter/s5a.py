from h import *
import time
rng=np.random.default_rng(5)
def test(n,scale,T=16):
    # T tests (rows); each: input row of n pixels, kernel random 4-bit ints * scale; out at x=0
    W=rng.integers(-7,8,n)*scale
    ks=" ".join(f"{v:.8g}" for v in W[::-1])
    X=128+rng.integers(-20,21,(T,n))
    defs=f'<filter id="f" filterUnits="userSpaceOnUse" x="0" y="0" width="{n}" height="{T}" color-interpolation-filters="sRGB"><feConvolveMatrix order="{n} 1" kernelMatrix="{ks}" divisor="1" bias="{(128-128*W.sum())/255:.8f}" targetX="0" targetY="0" edgeMode="none" preserveAlpha="true"/></filter>'
    from PIL import Image
    import io,base64
    im=Image.fromarray(np.stack([X.astype(np.uint8)]*3+[np.full_like(X,255,dtype=np.uint8)],-1),"RGBA"); b=io.BytesIO(); im.save(b,"PNG")
    uri="data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{max(n,8)}" height="{T}"><defs>{defs}</defs><rect width="{n}" height="{T}" fill="#000"/><g filter="url(#f)"><image href="{uri}" width="{n}" height="{T}" style="image-rendering:pixelated"/></g></svg>'
    t0=time.time(); a=render(svg,max(n,8),T,"s5a"); dt=time.time()-t0
    got=a[:,0,0].astype(int); ref=np.clip(X@W[::-1][::-1],0,255)  # kernel flipped twice: ref = sum W[j]*X[j]
    exact=np.clip(np.round(128+(X-128)@W),0,255)
    return got,exact,X@W,dt
for n,sc in [(4,1/4),(32,1/16),(128,1/16),(384,1/32),(512,1/32),(1024,1/32)]:
    got,ex,raw,dt=test(n,sc)
    print(n,sc,"rendered in %.1fs"%dt,"maxerr vs round(clip)",abs(got-ex).max(),"frac exact",(got==ex).mean(),"unclipped sums",raw[:4].round(1),"got",got[:4])
