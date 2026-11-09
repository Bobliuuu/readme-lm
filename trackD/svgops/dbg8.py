from t_o8 import *
gw,gh=16,24
voc=list(Q.VOC)+["EOS"]; atlas=make_atlas(voc,gw,gh)
ox=int(gw*127.5-gw*85); W=max(ox+gw+4, atlas.shape[1]+4); oy=4; Hh=oy+gh+4
i=23
c=Ctx(W,Hh); a3=np.zeros((Hh,W,3),np.uint8); a3[2,0,1]=i
c.image(a3,res="src"); X=c.merge(["src"]); g,ox=glyph_stage(c,X,2,atlas,gw,gh,oy)
svg=c.g.svg(); a,dt=render(svg,W,Hh,"dbg8",timeout=100)
got=a[oy:oy+gh,ox:ox+gw,0]; ref=atlas[:,i*gw:(i+1)*gw]
print("ox",ox,"W",W,"Hh",Hh,"dt",dt)
for r in range(0,gh,2): print("".join("#" if v>128 else "." for v in got[r]),"  ","".join("#" if v>128 else "." for v in ref[r]))
print("alpha",a[oy:oy+gh,ox:ox+gw,3].min(), "uniq got",np.unique(got)[:10])
# search shift
for dx in range(-20,21):
  for dy in range(-14,15):
    sh=np.roll(np.roll(ref,dx,1),dy,0)
    if (sh==got).all(): print("match shift",dx,dy)
