"""Core: SVG filter graph builder + Chrome renderer + PNG data-URI helpers. No JS in generated SVG."""
import subprocess, os, io, base64, time, numpy as np
from PIL import Image
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
D=os.path.dirname(os.path.abspath(__file__)); WORK=f"{D}/work"; os.makedirs(WORK,exist_ok=True)

def png_uri(a):
    """a: HxW (gray replicated to RGB, alpha 255) or HxWx3/4 uint8 -> data URI (palette-free RGB(A), no gamma chunk)."""
    a=np.asarray(a,np.uint8)
    if a.ndim==2: a=np.stack([a,a,a],-1)
    if a.shape[2]==3: a=np.concatenate([a,np.full(a.shape[:2]+(1,),255,np.uint8)],-1)
    b=io.BytesIO(); Image.fromarray(a,"RGBA").save(b,"PNG",optimize=True)
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

class Graph:
    def __init__(s,W,H): s.W,s.H=W,H; s.p=[]; s.n=0
    def add(s,xml,res=None):
        s.n+=1; res=res or f"r{s.n}"; x=xml.replace("RESULT",f'result="{res}"')
        if x.endswith("RESULT") or not x.rstrip().endswith((">",)): x=x.rstrip()+"/>"
        s.p.append(x); return res
    def count(s): return len(s.p)
    def svg(s,out=None,extra_defs=""):
        out=out or s.p and None
        body="".join(s.p)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.W}" height="{s.H}" viewBox="0 0 {s.W} {s.H}"><defs>'
                f'<filter id="F" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" x="0" y="0" width="{s.W}" height="{s.H}" color-interpolation-filters="sRGB">{body}</filter>{extra_defs}</defs>'
                f'<rect width="{s.W}" height="{s.H}" fill="#000" filter="url(#F)"/></svg>')

def image_prim(g,arr,x=0,y=0,res=None):
    h,w=np.asarray(arr).shape[:2]
    return g.add(f'<feImage href="{png_uri(arr)}" x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" RESULT/>',res)

_dpr=1
def render(svg,W,H,name="t",img=False,dpr=1,timeout=100,keep=False):
    p=f"{WORK}/{name}.svg"; open(p,"w").write(svg)
    out=f"{WORK}/{name}{'.img' if img else ''}.png"
    if img:
        h=f"{WORK}/{name}.html"
        open(h,"w").write(f'<html><body style="margin:0;background:#000"><img src="{name}.svg" width={W} height={H} style="display:block"></body></html>'); u="file://"+h
    else: u="file://"+p
    if os.path.exists(out): os.remove(out)
    import shutil, signal, uuid; UD=f"{WORK}/ud_{uuid.uuid4().hex[:8]}"
    t=time.time()
    for attempt in range(3):
        pr=subprocess.Popen([CH,"--headless=new","--disable-gpu","--hide-scrollbars",f"--force-device-scale-factor={dpr}",f"--screenshot={out}",f"--window-size={W},{H}","--user-data-dir="+UD,u],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        try: pr.wait(timeout=timeout)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(pr.pid,signal.SIGKILL)
        except Exception: pass
        if os.path.exists(out): break
    dt=time.time()-t
    a=np.array(Image.open(out).convert("RGBA")); shutil.rmtree(UD,ignore_errors=True); os.remove(out)
    if not keep: os.remove(p)
    return a[:H*dpr,:W*dpr],dt

# ---- persistent Playwright-driven Chrome (installed Google Chrome, headless) : much faster/more reliable than --screenshot ----
_pw = None; _br = None
def _browser():
    global _pw, _br
    if _br is None:
        from playwright.sync_api import sync_playwright
        if _pw is None: _pw = sync_playwright().start()
        _br = _pw.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu", "--hide-scrollbars"])
    return _br
def render_pw(svg, W, H, name="t", img=False, keep=False, timeout=900, dpr=1):
    timeout = timeout * 1000
    p = f"{WORK}/{name}.svg"; open(p, "w").write(svg)
    if img:
        h = f"{WORK}/{name}.html"
        open(h, "w").write(f'<html><body style="margin:0;background:#000"><img src="{name}.svg" width={W} height={H} style="display:block"></body></html>'); u = "file://" + h
    else: u = "file://" + p
    ctx = _browser().new_context(viewport={"width": max(W, 1), "height": max(H, 1)}, device_scale_factor=dpr)
    pg = ctx.new_page(); t = time.time()
    try:
        pg.goto(u, timeout=timeout, wait_until="load")
        png = pg.screenshot(timeout=max(1000, timeout - int((time.time() - t) * 1000)), full_page=False, type="png"); dt = time.time() - t
    except Exception as e:
        global _br, _pw
        try: ctx.close()
        except Exception: pass
        try: _br.close()
        except Exception: pass
        _br = None
        raise TimeoutError(f"render failed/timeout after {time.time()-t:.0f}s: {str(e)[:80]}")
    ctx.close()
    a = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    if not keep:
        os.remove(p)
        if img: os.remove(h)
    return a[:H * dpr, :W * dpr], dt

render_cli = render          # original `chrome --screenshot` path (works, but intermittently stalls under load)
render = render_pw           # default: persistent Playwright-driven installed Chrome
