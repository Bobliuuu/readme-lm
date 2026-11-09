from core import *
import core
# exactness of feDisplacementMap over all 256 codes: y channel (G) and x channel (R)
Hh=70000; Wd=300
def run(scale, axis):
    y0 = 35000
    A=np.zeros((Hh,600,3),np.uint8); ys=np.arange(Hh); A[:,:,0]=(ys%256)[:,None]; A[:,:,1]=((ys//256)%256)[:,None]; A[:,:,2]=(ys//65536)[:,None]
    M=np.zeros((1,256,3),np.uint8); M[0,:,0]=128; M[0,:,1]=128; M[0,:,axis]=np.arange(256)
    g=Graph(600,Hh)
    image_prim(g,A,x=0,y=0,res="atlas")
    g.add(f'<feImage href="{png_uri(M)}" x="0" y="{y0}" width="256" height="1" preserveAspectRatio="none" RESULT',"map")
    g.add(f'<feDisplacementMap in="atlas" in2="map" scale="{scale}" xChannelSelector="R" yChannelSelector="G" x="0" y="{y0}" width="256" height="1" RESULT')
    svg=g.svg().replace(f'width="600" height="{Hh}" viewBox="0 0 600 {Hh}"',f'width="256" height="1" viewBox="0 {y0} 256 1"',1)
    a,dt=core.render_pw(svg,256,1,"p10")
    return a[0]
# y axis: displacement in y by G, read row id
res={}
for K in (1,2,4,12,66):
    scale=255*K
    out=run(scale,1)
    rows=out[:,0].astype(int)+256*out[:,1].astype(int)+65536*out[:,2].astype(int)
    dy=rows-35000
    G=np.arange(256)
    exact=K*G-127.5*K
    print("K",K,"dy[0..3]",dy[:4].tolist(),"exact",exact[:4].tolist(),"dy-floor(exact)==0:",bool((dy==np.floor(exact+0.5)).all()),"dy-floor(exact)==0 (floor):",bool((dy==np.floor(exact)).all()),"max|dy-exact|",float(np.abs(dy-exact).max()))
print("----")
for K in (1,2,12,66):
    out=run(255*K,1); rows=out[:,0].astype(int)+256*out[:,1].astype(int)+65536*out[:,2].astype(int); dy=rows-35000
    G=np.arange(256); ex=K*G-127.5*K
    bad=np.where(np.abs(dy-ex)>1.01)[0]
    print("K",K,"n bad (>1)",len(bad),"first bad G",bad[:6].tolist(),"dy there",dy[bad[:6]].tolist(),"exact",ex[bad[:6]].tolist())
    good=np.where(np.abs(dy-ex)<=1.01)[0]
    diff=(dy-ex)[good]; print("   among good: diff counts",{float(v):int((diff==v).sum()) for v in np.unique(diff)})
