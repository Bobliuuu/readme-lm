import io, sys, zlib
from svg_eval import *
from PIL import Image
import base64
def sizes(wmax):
    T=16; p=make_prompt(T,"what is x"); ids=mk(p)
    pl=build(T,[ids],wmax=wmax)
    kern=0; png=0; n=0
    for k,L in pl.lins.items():
        if not isinstance(L,dict): continue
        E=L["E"]; n+=E.size
        # kernel text: per-neuron "e e e ..." (+guard) as emitted
        for row in E: kern+=len(" ".join(map(str,row.tolist()))) + 6 + 4
        # as PNG raster (E+128 codes, 8-bit gray)
        b=io.BytesIO(); Image.fromarray((E+128).astype(np.uint8)).save(b,"PNG",optimize=True); png+=len(base64.b64encode(b.getvalue()))+30
    # agreement
    qs=["what is x","hi there","what is qwentpu","how are you","tell me a joke","who are you","capital of france","what is two"]
    ag=0; md=[]
    for q in qs:
        pr=make_prompt(T,q[:T-7]); i2=mk(pr); tr=forward_ref(pl,i2); ref=pl.m.forward_token(i2); lg=tr["logits"][-1]
        ag+=int(lg.argmax()==ref.argmax()); md.append(np.abs(lg-ref.astype(int)).mean())
    return dict(wmax=wmax,weights=n,kernel_text_bytes=kern,png_b64_bytes=png,bytes_per_weight_text=round(kern/n,2),bytes_per_weight_png=round(png/n,2),argmax_agree=f"{ag}/{len(qs)}",mean_logit_err=round(float(np.mean(md)),2))
for w in (127,63,31,15):
    print(sizes(w),flush=True)
