from core import *
import core
def run(scale,y0,axis=1,Hh=600,tile=False):
    A=np.zeros((Hh,600,3),np.uint8); ys=np.arange(Hh); A[:,:,0]=(ys%256)[:,None]; A[:,:,1]=((ys//256)%256)[:,None]
    M=np.zeros((Hh,1,3),np.uint8); M[:,0,0]=128; M[:,0,1]=128
    M[y0,0,axis]=0
    Gs=np.arange(256)
    g=Graph(256,Hh); image_prim(g,A,x=0,y=0,res="atlas")
    # map: column x=G -> value G at row y0  (1 row, 256 cols)
    M=np.zeros((1,256,3),np.uint8); M[0,:,0]=128; M[0,:,1]=128; M[0,:,axis]=Gs
    g.add(f'<feImage href="{png_uri(M)}" x="0" y="{y0}" width="256" height="1" preserveAspectRatio="none" RESULT',"map")
    g.add(f'<feDisplacementMap in="atlas" in2="map" scale="{scale}" xChannelSelector="R" yChannelSelector="G" x="0" y="{y0}" width="256" height="1" RESULT')
    svg=g.svg().replace(f'width="256" height="{Hh}" viewBox="0 0 256 {Hh}"',f'width="256" height="1" viewBox="0 {y0} 256 1"',1)
    a,dt=core.render_pw(svg,256,1,"p11"); out=a[0]; rows=out[:,0].astype(int)+256*out[:,1].astype(int)
    return rows-y0
for y0 in (0,3,100,300):
    dy=run(255,y0); print("y0",y0,"dy for G=126..131:",dy[126:132].tolist(),"G=0:",dy[0],"G=255:",dy[255])
