import sys,time
from pipeline import *
voc=Q.VOC
def mk(txt): return [voc.index(ch) for ch in txt]
def make_prompt(T,q="what is x"):
    s="q: "+q+"\na: "; 
    if len(s)<T: s="q: "+q+" "*(T-len(s))+"\na: "
    assert len(s)==T,(len(s),T); return s
if __name__=="__main__":
    T=int(sys.argv[1]) if len(sys.argv)>1 else 16
    qs=["what is x","hi there","what is qwentpu","how are you","tell me a joke","who are you","capital of france","what is two"]
    ps=[make_prompt(T,q[:T-7]) for q in qs]
    pl=build(T,[mk(p) for p in ps[:4]])
    for k,v in pl.norms.items(): print(k,"S1",v["S1"],end="; ")
    print()
    agree=0
    for p in ps:
        tr=forward_ref(pl,mk(p)); ref=pl.m.forward_token(mk(p)); lg=tr["logits"][-1]
        agree+= int(lg.argmax()==ref.argmax()); print(repr(p),"svgref",lg.argmax(),"Q8",ref.argmax(),"logit mean|d|",round(float(np.abs(lg-ref.astype(int)).mean()),2),"max",int(np.abs(lg-ref.astype(int)).max()))
    print("argmax agree",agree,"/",len(ps))
