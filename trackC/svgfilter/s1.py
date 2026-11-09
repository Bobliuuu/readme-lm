from h import *
import itertools, json
res={}
A=list(range(0,256,5))+[255]
pairs=[(a,b) for a in A for b in A]
def fl(a,b): return f'<feFlood flood-color="rgb({a},{a},{a})" result="A"/><feFlood flood-color="rgb({b},{b},{b})" result="B"/>'
for ci in ("sRGB","linearRGB"):
  for img in (False,True):
    prod=grid([fl(a,b)+'<feComposite in="A" in2="B" operator="arithmetic" k1="1"/>' for a,b in pairs],cols=52*2,name="s1p",img=img,ci=ci)[:,0].astype(int)
    summ=grid([fl(a,b)+'<feComposite in="A" in2="B" operator="arithmetic" k2="0.5" k3="0.5" k4="0" result="s"/><feColorMatrix type="matrix" values="1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1"/>' for a,b in pairs],cols=104,name="s1s",img=img,ci=ci)[:,0].astype(int)
    a=np.array([p[0] for p in pairs]);b=np.array([p[1] for p in pairs])
    ep=prod-np.round(a*b/255); es=summ-np.round((a+b)/2)
    k=f"{ci}{'/img' if img else '/direct'}"
    res[k]=dict(prod_maxerr=float(abs(ep).max()),prod_exact=float((ep==0).mean()),sum_maxerr=float(abs(es).max()),sum_exact=float((es==0).mean()))
    print(k,res[k])
json.dump(res,open("s1_results.json","w"),indent=1)
