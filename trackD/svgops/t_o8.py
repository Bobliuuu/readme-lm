import json, sys
from ops_lib import *
sys.path.insert(0,"/Users/jhome/readme-lm/trackD/qat8")
import q8_ref as Q
def run_glyph(tag="o8", gw=16, gh=24):
    voc=list(Q.VOC)+["EOS"]; atlas=make_atlas(voc,gw,gh)
    H=4; ids=[0,23,85,40]
    ox=int(gw*127.5-gw*85); W=max(ox+gw+4, atlas.shape[1]+4); oy=4 ; Hh=max(H,oy+gh+gw)
    c=Ctx(W,Hh); arr=np.zeros((Hh,W),np.uint8)
    res=[]
    # one run per id: X image with G = id at (0,row)
    for row,i in [(2,40)] if False else []: pass
    out=[]
    for i in ids:
        c=Ctx(W,Hh); a3=np.zeros((Hh,W,3),np.uint8); a3[2,127,0]=i
        c.image(a3,res="src"); X=c.merge(["src"]); g,ox=glyph_stage(c,X,2,atlas,gw,gh,oy)
        svg=c.g.svg(); 
        # crop view to the glyph cell via viewBox (no scaling)
        svg2=svg.replace(f'width="{W}" height="{Hh}" viewBox="0 0 {W} {Hh}"',f'width="{gw}" height="{gh}" viewBox="{ox} {oy} {gw} {gh}"',1)
        a,dt=render(svg2,gw,gh,tag,timeout=100); got=a[:,:,0]
        ref=atlas[:,i*gw:(i+1)*gw]
        out.append((i,bool((got==ref).all()),int((got!=ref).sum()),round(dt,2)))
    ok=all(o[1] for o in out)
    return dict(op="glyph atlas via feDisplacementMap (id pixel -> 16x24 glyph, viewBox-cropped)",PASS=ok,detail=out,prims=c.g.count(),svg_bytes=len(svg2))
if __name__=="__main__":
    r=run_glyph(); print(r); json.dump([r],open("results_o8.json","w"),indent=1)
