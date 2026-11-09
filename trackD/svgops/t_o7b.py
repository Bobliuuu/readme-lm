import json,sys
from t_o7 import *
res=[]
for T in (16,32,64):
    r=run_att(T,tag=f"o7_{T}"); print(r,flush=True); res.append(r)
    json.dump(res,open("results_o7.json","w"),indent=1,default=str)
