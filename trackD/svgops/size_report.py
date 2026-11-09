import json, time
import pipeline
from pipeline import *
from svg_eval import *
import core
out=[]
for T in (16,64,192):
    p=make_prompt(T,"what is x") if T<=64 else None
    if T==192:
        # prompt text irrelevant for size: pad question
        q="what is x"; s="q: "+q+" "*(T-len(q)-7)+"\na: "; ids=mk(s)
    else: ids=mk(p)
    try:
        pl=build(T,[ids])
        svg,view,c=build_svg(pl,ids,stop=None)
        r=dict(T=T,prims=c.g.count(),svg_MB=round(len(svg)/1e6,3),region=(c.W,c.H))
    except Exception as e:
        r=dict(T=T,error=str(e)[:200])
    print(r,flush=True); out.append(r)
json.dump(out,open("results_size.json","w"),indent=1,default=str)
