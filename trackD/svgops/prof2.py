import sys, time
import pipeline
from pipeline import *
from svg_eval import *
import core
T=16; p=make_prompt(T,"what is x"); ids=mk(p); pl=build(T,[ids])
def front(c):
    zero=lambda n: np.zeros(n,np.int64)
    x=embed_lookup(c,ids,pl.emb,0); n1=rmsnorm(c,x,0,128,pl.norms["L0.n1"])
    L=pl.lins["L0.q"]; q=linear(c,n1,[(0,128)],L["E"],L["d"],L["zin"],L["zout"],zero(128),ACOLS)
    R_=pl.rope["L0.q"]; qr=rope_apply(c,q,ACOLS,R_["cosc"],R_["sinc"],R_["z"],R_["zo"],R_["R"],wspan=WA)
    return qr
def mk_ctx():
    gw,gh=GW,GH; ox=int(gw*127.5-gw*85); W=max(660,ox+gw+8,gw*87+8); H=attn_H(T); c=Ctx(W,H); c.rows=T; return c
def run(name,fn):
    c=mk_ctx(); out=fn(c, front(c)); c.prim(f'<feComponentTransfer in="{out}"/>')
    svg=c.g.svg().replace(f'width="{c.W}" height="{c.H}" viewBox="0 0 {c.W} {c.H}"',f'width="{WA}" height="{T}" viewBox="0 0 {WA} {T}"',1)
    a,dt=core.render_pw(svg,WA,T,"prof2",timeout=900); print(name,"prims",c.g.count(),"render_s",round(dt,2),flush=True)
m,P=pick_pitch(T); TT=T*P
run("front only", lambda c,qr: qr)
run("front + tile(K)", lambda c,qr: c.tile(c.merge([qr],h=P),0,0,WA,TT))
run("front + stretch", lambda c,qr: gather_stretch(c,qr,T,P,m,TT,WA))
run("front + 2 offsets", lambda c,qr: c.merge([c.offset(qr,3,0,x=0,y=0,w=c.W,h=TT+2), c.offset(qr,3,1,x=0,y=0,w=c.W,h=TT+2)],h=TT))
