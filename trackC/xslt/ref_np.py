import numpy as np, re
Z=np.load("model_f32.npz"); voc=str(Z["voc"]); EOS=len(voc); d=128;H=4;D=32;Lr=3
W={k:Z[k].astype(np.float64) for k in Z.files if k!="voc"}
def rms(x,g): return x/np.sqrt((x*x).mean()+1e-5)*g
def erf(x):
    s=np.sign(x);x=np.abs(x);t=1/(1+0.3275911*x)
    return s*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*np.exp(-x*x))
gelu=lambda x:0.5*x*(1+erf(x/1.41421356))
def rope(x,pos):
    f=1.0/(10000**(np.arange(0,D,2)/D));a=pos*f;c,s=np.cos(a),np.sin(a);o=np.empty_like(x)
    o[:,0::2]=x[:,0::2]*c-x[:,1::2]*s;o[:,1::2]=x[:,0::2]*s+x[:,1::2]*c;return o
def clean(q):
    q="".join(c for c in q.lower() if c in voc);return re.sub(r"\s+"," ",q).strip()[:140]
def seq(question,forced=""): return "q: "+clean(question)+"\na: "+forced
def run(text,maxn=0):
    toks=[voc.index(c) for c in text];K=[[] for _ in range(Lr)];Vv=[[] for _ in range(Lr)];out=[];lg=None
    pos=0
    while True:
        t=toks[pos];x=W["emb"][t].copy()
        for l in range(Lr):
            qkv=(W[f"qkv{l}"]@rms(x,W[f"g1_{l}"])).reshape(3,H,D)
            K[l].append(rope(qkv[1],pos));Vv[l].append(qkv[2]);qq=rope(qkv[0],pos)
            Kc,Vc=np.stack(K[l],1),np.stack(Vv[l],1);sc=np.einsum("hd,htd->ht",qq,Kc)/np.sqrt(D);sc=np.exp(sc-sc.max(1,keepdims=True));sc/=sc.sum(1,keepdims=True)
            x=x+W[f"o{l}"]@np.einsum("ht,htd->hd",sc,Vc).reshape(d)
            x=x+W[f"dn{l}"]@gelu(W[f"u{l}"]@rms(x,W[f"g2_{l}"]))
        if pos+1<len(toks): pos+=1; continue
        lg=W["emb"]@rms(x,W["gf"]);n=int(lg.argmax())
        if maxn==0 or n==EOS or len(out)>=maxn: return lg,out and "".join(out) or "",n
        toks.append(n);out.append(voc[n]);pos+=1
if __name__=="__main__":
    lg,ans,n=run(seq("what is qwentpu"),maxn=300);print(repr(ans))
