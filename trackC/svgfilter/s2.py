from h import *
rng=np.random.default_rng(1)
def enc(x): return (np.asarray(x)+128)/255
def run(W,x,alpha_data=True,engine="chrome",img=False):
    # 4x4 matvec, x in RGBA of one flood, W in feColorMatrix. Output: 4 pixels? -> use readout stages moving each channel to gray
    p=enc(x); off=(128-128*W.sum(1))/255
    M=" ".join(" ".join(f"{v:g}" for v in list(W[i])+[off[i]]) for i in range(4))
    # readout: channel c -> gray with alpha 1 (reads unpremult channel c)
    def ro(c):
        r=[0]*5; r[c]=1
        return " ".join(map(str,r*3+[0,0,0,0,1]))
    outs=""
    for c in range(4):
        outs+=f'<feColorMatrix in="Y" type="matrix" values="{ro(c)}" result="o{c}"/><feOffset in="o{c}" dx="{c*2}" result="p{c}"/>'
    # pixel layout: 4 pixels at x=0,2,4,6 each 2px wide
    clip=lambda c:f'<feFlood flood-color="#000" result="bk"/>'
    merge="<feMerge>"+"".join(f'<feMergeNode in="p{c}"/>' for c in range(4))+"</feMerge>"
    # restrict each readout to own 2px column using feOffset after crop
    s=f'''<svg xmlns="http://www.w3.org/2000/svg" width="8" height="2"><defs>
<filter id="f" x="0" y="0" width="8" height="2" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
<feFlood flood-color="rgb({int(p[0]*255)},{int(p[1]*255)},{int(p[2]*255)})" flood-opacity="{p[3]:.6f}" x="0" y="0" width="2" height="2"/>
<feColorMatrix type="matrix" values="{M}" result="Y"/>
{outs.replace('dx="2"','dx="0"')}</filter></defs><rect width="8" height="2" filter="url(#f)"/></svg>'''
    return s
# simpler: render 4 separate svgs, one per output channel (readout via matrix), flood fills whole canvas
def svg(W,x,c):
    p=enc(x); off=(128-128*W.sum(1))/255
    M=" ".join(" ".join(f"{v:.6f}" for v in list(W[i])+[off[i]]) for i in range(4))
    r=[0]*5; r[c]=1; R=" ".join(map(str,r*3+[0,0,0,0,1]))
    col=",".join(str(int(round(p[i]*255))) for i in range(3))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><defs><filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">
<feFlood flood-color="rgb({col})" flood-opacity="{round(p[3]*255)/255:.8f}"/><feColorMatrix type="matrix" values="{M}"/><feColorMatrix type="matrix" values="{R}"/></filter></defs>
<rect width="4" height="4" filter="url(#f)"/></svg>'''
import sys
ok=0;n=0;maxerr=0
for t in range(12):
    W=rng.integers(-3,4,(4,4)); x=rng.integers(-7,8,4)
    y=W@x
    got=[]
    for c in range(4):
        px=render(svg(W,x,c),4,4,"s2")[2,2,0]; got.append(int(px)-128)
    err=np.abs(np.array(got)-y); maxerr=max(maxerr,err.max()); n+=1; ok+= (err.max()==0)
    if t<4: print("x",x,"y",y,"got",got)
print("exact",ok,"/",n,"maxerr",maxerr)
