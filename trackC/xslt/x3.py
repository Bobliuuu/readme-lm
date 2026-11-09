import numpy as np, subprocess
rng=np.random.default_rng(3); d=128; Vn=16; N=24
E=rng.normal(size=(Vn,d))*0.5; W1=rng.normal(size=(d,d))/np.sqrt(d)*1.5; W2=rng.normal(size=(d,d))/np.sqrt(d)*1.5; H=rng.normal(size=(Vn,d)); g=rng.normal(size=d)*.2+1
r9=lambda a:np.array([[float(f"{v:.9g}") for v in row] for row in np.atleast_2d(a)])
E,W1,W2,g,H=r9(E),r9(W1),r9(W2),r9(g)[0],r9(H)
def ref(t0):
    t=t0; out=[t]; s=np.zeros(d)
    for _ in range(N):
        h=E[t]+s
        for W in (W1,W2):
            n=h/np.sqrt((h*h).mean()+1e-5)*g; h=h+np.maximum(W@n,0)
        s=0.5*s+0.5*h*0.1   # recurrent context state
        lg=H@(h/np.sqrt((h*h).mean()+1e-5)); t=int(lg.argmax()); out.append(t)
    return out
R=ref(3)
V=lambda a:"".join(f"<v>{v:.9g}</v>" for v in a)
M=lambda A:"".join("<r>"+"".join("<c>"+V(r[i:i+32])+"</c>" for i in range(0,128,32))+"</r>" for r in A)
open("x3.xml","w").write(f"<m start='3' steps='{N}'><E>{M(E)}</E><W1>{M(W1)}</W1><W2>{M(W2)}</W2><H>{M(H)}</H><g>{V(g)}</g><s0>{V(np.zeros(d))}</s0></m>")
o=subprocess.run(["xsltproc","x3.xsl","x3.xml"],capture_output=True,text=True)
print(o.stderr[:400]); got=[int(t) for t in o.stdout.split()]
print("xslt ",got);print("numpy",R);ok=got==R
print("X3","PASS" if ok else "FAIL")
open("x3_result.txt","w").write(f"toy 2-layer net d=128 vocab=16, start token 3, {N} greedy steps in one xsltproc run\nxslt ={got}\nnumpy={R}\n{'PASS' if ok else 'FAIL'}\n")
