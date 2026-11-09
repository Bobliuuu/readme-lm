import numpy as np, subprocess, time, os, json
import ref_np as R
from xml.sax.saxutils import escape, quoteattr
cases=[("what is qwentpu",""),("what is qwentpu","qwentpu is a TPU compiled"),("who are you",""),
       ("What is a TPU?","a T"),("does it need training","no TPU hardware and no tra"),
       ("what is qwentpu","qwentpu is a TPU compiled into the weights of a stock Qwen3.5 (0.8B, 2B or 4B), with no TPU hardware")]
os.makedirs("x4",exist_ok=True)
procs=[]
for i,(q,f) in enumerate(cases):
    open(f"x4/ask{i}.xml","w").write(f'<ask max="0" dump="1" forced={quoteattr(f)}><q>{escape(q)}</q></ask>')
    t=time.time();p=subprocess.Popen(["xsltproc","--maxdepth","5000","qa.xsl",f"x4/ask{i}.xml"],stdout=subprocess.PIPE,text=True);procs.append((p,t))
rows=[];allok=True;log=[]
for i,((q,f),(p,t)) in enumerate(zip(cases,procs)):
    out=p.communicate()[0];dt=time.time()-t
    lines=out.split("\n");lg=np.array([float(x) for x in lines[0].split()[1:]]);am=int(lines[1].split()[1])
    ref,_,rn=R.run(R.seq(q,f))
    err=np.abs(lg-ref).max();ok=(am==rn) and err<1e-4;allok&=ok
    s=f"case{i} q={q!r} forced={f[:30]!r}({len(f)}ch) ntok={len(R.seq(q,f))} xslt_argmax={am}({R.voc[am] if am<85 else 'EOS'}) numpy_argmax={rn}({R.voc[rn] if rn<85 else 'EOS'}) max|dlogit|={err:.2e} wall={dt:.0f}s(parallel) {'PASS' if ok else 'FAIL'}"
    print(s);log.append(s)
open("x4_result.txt","w").write("\n".join(log)+f"\nX4 {'PASS' if allok else 'FAIL'}\n")
print("X4","PASS" if allok else "FAIL")
