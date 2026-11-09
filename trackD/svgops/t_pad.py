from t_o1 import *
Wr = real_weights("qkv0"); Wr = Wr * (0.1 / Wr.std()); E,d=quantize_weights(Wr,127)
rng = np.random.default_rng(3)
H=8;n_in=128;n_out=384
for pad in (130,):
    c=Ctx(130+384+1+pad,H); out0=130; out_cols=[out0+j for j in range(n_out)]
    U=np.clip(rng.normal(128,25,(H,n_in)).round(),0,255).astype(np.int64)
    arr=np.zeros((H,c.W),np.uint8); arr[:,:128]=U; c.image(arr,res="src")
    bacc=np.zeros(n_out,int)
    linear(c,"src",[(0,128)],E,d,128,128,bacc,out_cols)
    a,dt=render(c.g.svg(),c.W,H,"dbg2"); got=a[:,out0:out0+n_out,0].astype(int); ref=linear_ref(U,[128],E,d,128,128,bacc)
    print(pad,"bad neurons",(got!=ref).any(0).sum(),"t",dt,flush=True)
