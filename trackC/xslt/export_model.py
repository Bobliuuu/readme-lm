# dequantize 4-bit weights.md -> model.xml (floats as text, one <v> per weight)
import re, io, base64, numpy as np, math
from xml.sax.saxutils import escape
FENCE="`"*3
md=open("/Users/jhome/readme-lm/qa/weights.md").read()
b=re.search(FENCE+r"\n(.*?)\n"+FENCE,md,re.S).group(1).replace("\n","")
Z=np.load(io.BytesIO(base64.b64decode(b)))
def dq(n):
    p,s=Z[n+"_q"],Z[n+"_s"].astype(np.float32);o=p.shape[0]
    q=np.empty((o,p.shape[1]*2),np.float32);q[:,0::2]=(p&15).astype(np.float32)-8;q[:,1::2]=(p>>4).astype(np.float32)-8
    return (q.reshape(o,-1,32)*s[...,None]).reshape(o,-1)
d,L,H,ff=[int(x) for x in Z["cfg"]]
voc="".join(chr(c) for c in Z["voc"])
W={"emb":dq("emb"),"gf":Z["gf"].astype(np.float32)}
for l in range(L):
    for n in("qkv","o","u","dn"):W[f"{n}{l}"]=dq(f"{n}{l}")
    W[f"g1_{l}"]=Z[f"g1_{l}"].astype(np.float32);W[f"g2_{l}"]=Z[f"g2_{l}"].astype(np.float32)
np.savez("model_f32.npz",voc=voc,**W)   # reference weights (exactly what XML holds, up to %.9g = float32-exact)
f=lambda a:"".join(f"<v>{float(v):.9g}</v>" for v in a)
import os
CH=int(os.environ.get("CH","32"))
M=lambda A:"".join("<r>"+"".join("<c>"+f(r[i:i+CH])+"</c>" for i in range(0,len(r),CH))+"</r>" for r in A)
o=['<?xml version="1.0"?>\n<model d="%d" L="%d" H="%d" ff="%d">'%(d,L,H,ff),"<voc>"+escape(voc)+"</voc>","<emb>"+M(W["emb"])+"</emb>","<gf>"+f(W["gf"])+"</gf>"]
for l in range(L):
    o.append(f'<layer>'+"".join(f"<{n}>{M(W[n+str(l)])}</{n}>" for n in("qkv","o","u","dn"))+f'<g1>{f(W[f"g1_{l}"])}</g1><g2>{f(W[f"g2_{l}"])}</g2></layer>')
D=d//H;fr=1.0/(10000**(np.arange(0,D,2)/D))
o.append("<rope>")
for pos in range(256):
    a=pos*fr;o.append("<p>"+f(np.cos(a))+f(np.sin(a))+"</p>")
o.append("</rope></model>")
open("model.xml","w").write("".join(o))
print(d,L,H,ff,len(voc),W["emb"].shape)
