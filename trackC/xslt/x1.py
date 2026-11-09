import numpy as np, subprocess, json
rng=np.random.default_rng(1)
W=rng.normal(size=(4,4)); x=rng.normal(size=4)
xml="<d><m>"+"".join("<r>"+"".join(f"<v>{v:.9g}</v>" for v in r)+"</r>" for r in W)+"</m><x>"+"".join(f"<v>{v:.9g}</v>" for v in x)+"</x></d>"
open("x1.xml","w").write(xml)
r=subprocess.run(["xsltproc","x1.xsl","x1.xml"],capture_output=True,text=True).stdout.split()
y=np.array([float(t) for t in r]); ref=W@x
# reference uses the same 9-digit rounded inputs
W2=np.array([[float(f"{v:.9g}") for v in row] for row in W]); x2=np.array([float(f"{v:.9g}") for v in x])
ref=W2@x2
print("xslt",y);print("numpy",ref);err=np.abs(y-ref).max();print("maxerr",err)
print("X1",["FAIL","PASS"][err<1e-9])
open("x1_result.txt","w").write(f"inputs W=\n{W2}\nx={x2}\nxslt={y}\nnumpy={ref}\nmaxerr={err}\n{'PASS' if err<1e-9 else 'FAIL'}\n")
