import sys, time
import pipeline
from svg_eval import *
import core
T=16; p=make_prompt(T,"what is x"); ids=mk(p); pl=build(T,[ids])
for upto in sys.argv[1].split(","):
    pipeline.ATT_UPTO = None if upto=="all" else upto
    svg,view,c=pipeline.build_svg(pl,ids,stop="L0.ao")
    t=time.time(); a,dt=core.render_pw(svg,view[2],view[3],"prof",timeout=3000); print("upto",upto,"prims",c.g.count(),"render_s",round(dt,1),flush=True)
