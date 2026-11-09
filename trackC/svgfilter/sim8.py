"""Simulate the QA transformer with every filter-primitive boundary quantized to 8 bits (static per-tensor scales)."""
import sys, re, io, base64, json, numpy as np
sys.path.insert(0,"/Users/jhome/readme-lm")
import qa_infer as Q
def load():
    r=open("/Users/jhome/readme-lm/qa/weights.md").read()
    b=re.search(r"```\n(.*?)\n```",r,re.S).group(1).replace("\n","")
    Z=np.load(io.BytesIO(base64.b64decode(b)))
    def dq(n):
        p,s=Z[n+"_q"],Z[n+"_s"].astype(np.float32);o=p.shape[0]
        q=np.empty((o,p.shape[1]*2),np.float32);q[:,0::2]=(p&15).astype(np.float32)-8;q[:,1::2]=(p>>4).astype(np.float32)-8
        return (q.reshape(o,-1,32)*s[...,None]).reshape(o,-1)
    d,L,H,ff=[int(x) for x in Z["cfg"]]
    m=dict(d=d,L=L,H=H,voc="".join(chr(c) for c in Z["voc"]),emb=dq("emb"),gf=Z["gf"].astype(np.float32),blocks=[])
    for l in range(L): m["blocks"].append({n:dq(f"{n}{l}") for n in ("qkv","o","u","dn")}|{"g1":Z[f"g1_{l}"].astype(np.float32),"g2":Z[f"g2_{l}"].astype(np.float32)})
    return m
class Ctx:
    def __init__(s,bits=None,scales=None,noise=0): s.bits=bits;s.scales=scales or {};s.mx={};s.noise=noise;s.rng=np.random.default_rng(0)
    def q(s,name,x,unsigned=False):
        if s.bits is None:
            s.mx[name]=max(s.mx.get(name,0),float(np.abs(x).max())); return x
        a=s.scales[name]; lv=2**s.bits-1
        if unsigned: return np.clip(np.round(x/a*lv),0,lv)*a/lv
        h=(lv-1)//2; y=x/a*h
        if getattr(s,'noise',0) and name.split('_')[0] in ('qkv','o','u','dn','logits'): y=y+s.rng.uniform(-s.noise,s.noise,np.shape(y))
        return np.clip(np.round(y),-h,h)*a/h
def lut_gelu(x): return Q.gelu(x)
def answer(m,question,ctx,maxn=300,sm_lut=True):
    voc,d,H=m["voc"],m["d"],m["H"];D=d//H;EOS=len(voc)
    qq="".join(c for c in question.lower() if c in voc);qq=re.sub(r"\s+"," ",qq).strip()[:140]
    toks=[voc.index(c) for c in "q: "+qq+"\na: "];K=[[] for _ in range(m["L"])];Vv=[[] for _ in range(m["L"])];out=[]
    q=ctx.q
    for pos in range(len(toks)+maxn):
        t=toks[pos];x=q("emb",m["emb"][t].copy())
        for l,B in enumerate(m["blocks"]):
            n1=q(f"n1_{l}",Q.rms(x,B["g1"]))
            qkv=q(f"qkv_{l}",(B["qkv"]@n1)).reshape(3,H,D)
            kk=q(f"k_{l}",Q.rope(qkv[1],pos,D));qr=q(f"q_{l}",Q.rope(qkv[0],pos,D))
            K[l].append(kk);Vv[l].append(qkv[2])
            Kc,Vc=np.stack(K[l],1),np.stack(Vv[l],1)
            sc=q(f"sc_{l}",np.einsum("hd,htd->ht",qq if False else qr,Kc)/np.sqrt(D))
            if ctx.bits is None:
                q(f"e_{l}",np.ones(1),unsigned=True); e=np.exp(sc-sc.max(1,keepdims=True)); p=e/e.sum(1,keepdims=True); q(f"p_{l}",p,unsigned=True)
            else:
                e=q(f"e_{l}",np.exp(sc-sc.max(1,keepdims=True)),unsigned=True); p=q(f"p_{l}",e/np.maximum(e.sum(1,keepdims=True),1e-9),unsigned=True)
            ao=q(f"ao_{l}",np.einsum("ht,htd->hd",p,Vc).reshape(d))
            x=q(f"r1_{l}",x+q(f"o_{l}",B["o"]@ao))
            n2=q(f"n2_{l}",Q.rms(x,B["g2"]))
            u=q(f"u_{l}",B["u"]@n2); g=q(f"g_{l}",Q.gelu(u))
            x=q(f"r2_{l}",x+q(f"dn_{l}",B["dn"]@g))
        if pos+1<len(toks): continue
        lg=q("logits",m["emb"]@q("nf",Q.rms(x,m["gf"])))
        n=int(lg.argmax())
        if n==EOS:break
        toks.append(n);out.append(voc[n])
    return "".join(out)
if __name__=="__main__":
    import glob,random
    m=load();random.seed(0);qs=[]
    for f in sorted(glob.glob("/Users/jhome/readme-lm/qa/g*.jsonl")):
        for ln in open(f):
            j=json.loads(ln);qs.append((j["qs"][0],j["a"]))
    random.shuffle(qs);cal=qs[:8];test=qs[8:8+40]
    c0=Ctx()
    for qn,a in cal: answer(m,qn,c0,maxn=120)
    ref=[answer(m,qn,Ctx()) for qn,a in test]
    gt=[a.lower() for _,a in test]
    def sim(bits,tag,noise=0):
        sc={k:v*1.0 for k,v in c0.mx.items()}
        ctx=Ctx(bits,sc,noise);o=[answer(m,qn,ctx) for qn,_ in test]
        # char agreement with fp ref (prefix match) and exact match
        ex=np.mean([a==b for a,b in zip(o,ref)]);pre=np.mean([np.mean([x==y for x,y in zip(a,b)]) if b else 1 for a,b in zip(o,ref)])
        print(tag,"noise",noise,"bits",bits,"exact-string match vs fp4w ref",round(ex,3),"mean first-len char match",round(pre,3));return o
    print("fp ref vs training answers char-prefix match:",np.mean([np.mean([x==y for x,y in zip(r,g)]) for r,g in zip(ref,gt)]))
    res={}
    for b in (8,10,12,16): sim(b,"sim")
    sim(8,"sim",1.5)
    for qn,r,g in list(zip([t[0] for t in test],ref,gt))[:2]: print(qn,"->",r[:70])
