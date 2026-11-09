from core import *
Wd,Hd=1300,40
A=np.zeros((Hd,Wd,3),np.uint8); xs=np.arange(Wd); A[:,:,0]=xs[None]%256; A[:,:,2]=xs[None]//256; A[:,:,1]=np.arange(Hd)[:,None]
cases=[(2040,127,127),(2040,128,128),(2040,0,0),(2040,255,255),(2040,100,127),(2040,101,127),(2040,102,127),(255,128,128),(255,127,127),(510,128,128)]
g=Graph(Wd,Hd); image_prim(g,A,x=0,y=0,res="atlas")
for i,(scale,rc,gc) in enumerate(cases):
    M=np.zeros((1,1,3),np.uint8); M[0,0]=(rc,gc,0)
    g.add(f'<feImage href="{png_uri(M)}" x="{600+i*2}" y="10" width="1" height="1" preserveAspectRatio="none" RESULT',f"m{i}")
    g.add(f'<feDisplacementMap in="atlas" in2="m{i}" scale="{scale}" xChannelSelector="R" yChannelSelector="G" x="{600+i*2}" y="10" width="1" height="1" RESULT',f"d{i}")
g.add("<feMerge RESULT>"+"".join(f'<feMergeNode in="d{i}"/>' for i in range(len(cases)))+"</feMerge>")
a,dt=render(g.svg(),Wd,Hd,"p9",timeout=100)
for i,(scale,rc,gc) in enumerate(cases):
    px=a[10,600+i*2]; ax=int(px[0])+256*int(px[2]); ay=int(px[1])
    print("scale",scale,"R",rc,"G",gc,"-> src x",ax,"y",ay,"alpha",px[3],"; dx=",ax-(600+i*2),"dy=",ay-10)
