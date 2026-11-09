import sys, json
from t_o7 import *
import core, t_o7
t_o7.render = core.render_pw
res=[]
for T in [int(x) for x in sys.argv[1].split(",")]:
    r=run_att(T,tag=f"o7c_{T}"); print({k:v for k,v in r.items() if k!='info'},flush=True); res.append(r)
    json.dump(res,open("results_o7.json","w"),indent=1,default=str)
