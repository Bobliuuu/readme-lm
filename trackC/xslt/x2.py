import numpy as np, subprocess, sys
sys.path.insert(0,"/Users/jhome/readme-lm")
rng=np.random.default_rng(2)
f=lambda a:[float(f"{v:.9g}") for v in a]
x=np.array(f(np.concatenate([rng.normal(size=10)*3,[-25,-12,0,0.5,8,30]])))
x2=np.array(f(rng.normal(size=128)*2)); x2g=np.array(f(rng.normal(size=128)*0.3+1))
sq=np.array([1e-6,3.7e-4,0.01,0.2,0.25,1,2,3.99,4,17.5,1e3,2.5e5,1e8])
V=lambda n,a:f"<{n}>"+"".join(f"<v>{v:.9g}</v>" for v in a)+f"</{n}>"
open("x2.xml","w").write("<t>"+V("x",x)+V("g",np.ones(len(x)))+V("x2",x2)+V("x2g",x2g)+V("sq",sq)+"</t>")
out=subprocess.run(["xsltproc","x2.xsl","x2.xml"],capture_output=True,text=True)
print(out.stderr[:500]); L={l.split(" ",1)[0]:l.split(" ",1)[1].split() for l in out.stdout.strip().split("\n")}
A=lambda k:np.array([float(t) for t in L[k]])
def erf(x):
    s=np.sign(x);x=np.abs(x);t=1/(1+0.3275911*x)
    return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*np.exp(-x*x))
from math import erf as merf
res=[]
def chk(name,got,ref,tol):
    e=np.max(np.abs(got-ref)/(np.abs(ref)+1e-12)) if name in("exp","sqrt") else np.max(np.abs(got-ref))
    ok=e<tol; res.append((name,e,ok)); print(f"{name:8s} maxerr={e:.3e} tol={tol:g} {'PASS' if ok else 'FAIL'}")
chk("relu",A("relu"),np.maximum(x,0),1e-12)
chk("gelu",A("gelu"),0.5*x*(1+erf(x/1.41421356)),1e-10)
chk("gelu(exact erf, info)",A("gelu"),np.array([0.5*v*(1+merf(v/2**.5)) for v in x]),1e-6)
e=np.exp(x-x.max()); chk("softmax",A("softmax"),e/e.sum(),1e-12)
chk("exp",A("exp"),np.exp(x),1e-9)
chk("rms",A("rms"),x2/np.sqrt((x2*x2).mean()+1e-5)*x2g,1e-10)
chk("sqrt",A("sqrt"),np.sqrt(sq),1e-10)
print("libxslt math:exp (info):",L["mathexp"][:5])
with open("x2_result.txt","w") as fh:
    fh.write("X2 inputs x=%s\nrms in(128 random)\nsqrt in=%s\n\n"%(x.tolist(),sq.tolist()))
    for k,v in L.items(): fh.write(k+": "+" ".join(v)+"\n")
    for n,e,ok in res: fh.write(f"{n} maxerr={e:.3e} {'PASS' if ok else 'FAIL'}\n")
