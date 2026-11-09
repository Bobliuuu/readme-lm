import subprocess, os, numpy as np
from PIL import Image
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
D=os.path.dirname(os.path.abspath(__file__))
def render(svg, W, H, name="t", img=False, engine="chrome"):
    p=f"{D}/work/{name}.svg"; os.makedirs(f"{D}/work",exist_ok=True); open(p,"w").write(svg)
    out=f"{D}/work/{name}.{engine}{'.img' if img else ''}.png"
    if engine=="rsvg":
        subprocess.run(["rsvg-convert","-w",str(W),"-h",str(H),p,"-o",out],check=True); return np.array(Image.open(out).convert("RGBA"))
    if img:
        h=f"{D}/work/{name}.html"; open(h,"w").write(f'<html><body style="margin:0;background:#000"><img src="{name}.svg" width={W} height={H} style="display:block"></body></html>'); u="file://"+h
    else: u="file://"+p
    if os.path.exists(out): os.remove(out)
    subprocess.run([CH,"--headless=new","--disable-gpu","--hide-scrollbars","--force-device-scale-factor=1",f"--screenshot={out}",f"--window-size={W},{H+ (0)}",u],capture_output=True,timeout=60)
    return np.array(Image.open(out).convert("RGBA"))[:H,:W]

def grid(filters, W=None, img=False, name="g", engine="chrome", cols=64, ci="sRGB"):
    """filters: list of inner-filter-XML strings; each rendered into its own 1px cell (flood region). returns array (n,4) RGBA"""
    n=len(filters); W=cols; Hh=(n+cols-1)//cols
    defs="".join(f'<filter id="f{i}" x="0" y="0" width="1" height="1" color-interpolation-filters="{ci}">{f}</filter>' for i,f in enumerate(filters))
    rects="".join(f'<rect x="{i%cols}" y="{i//cols}" width="1" height="1" filter="url(#f{i})"/>' for i in range(n))
    s=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}"><defs>{defs}</defs>{rects}</svg>'
    a=render(s,W,Hh,name,img,engine)
    return np.array([a[i//cols,i%cols] for i in range(n)])
