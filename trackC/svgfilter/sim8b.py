from sim8 import *
import glob,random
m=load();random.seed(0);qs=[]
for f in sorted(glob.glob("/Users/jhome/readme-lm/qa/g*.jsonl")):
    for ln in open(f): j=json.loads(ln);qs.append((j["qs"][0],j["a"]))
random.shuffle(qs);cal=qs[:8];test=qs[8:8+16]
c0=Ctx()
for qn,a in cal: answer(m,qn,c0,maxn=120)
ref=[answer(m,qn,Ctx()) for qn,a in test]
print({k:round(v,1) for k,v in c0.mx.items() if k.endswith("_0") or "_" not in k})
print("p range",[round(c0.mx[f'e_{l}'],1) for l in range(3)],"r2",[round(c0.mx[f'r2_{l}'],1) for l in range(3)])
class OnlyCtx(Ctx):
    def __init__(s,bits,scales,only): super().__init__(bits,scales); s.only=only
    def q(s,name,x,unsigned=False):
        base=name.split("_")[0]
        if base not in s.only: return x
        return Ctx.q(s,name,x,unsigned)
groups=["emb","n1","qkv","k","q","sc","e","p","ao","o","r1","n2","u","g","dn","r2","nf","logits"]
import sys
bits=int(sys.argv[1]) if len(sys.argv)>1 else 8
for g in groups:
    ctx=OnlyCtx(bits,dict(c0.mx),{g});o=[answer(m,qn,ctx) for qn,_ in test]
    print(g,"match",round(np.mean([a==b for a,b in zip(o,ref)]),2),"char",round(np.mean([np.mean([x==y for x,y in zip(a,b)]) if b else 1 for a,b in zip(o,ref)]),3))
