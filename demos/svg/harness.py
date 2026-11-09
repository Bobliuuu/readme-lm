"""Headless Chrome (Playwright) harness: render an SVG at DPR 1 and return RGBA pixels + wall time (load + screenshot)."""
import io, os, time, tempfile, numpy as np
from PIL import Image
_pw = None; _br = None
def _browser():
    global _pw, _br
    if _br is None:
        from playwright.sync_api import sync_playwright
        _pw = sync_playwright().start(); _br = _pw.chromium.launch(channel="chrome", headless=True, args=["--disable-gpu", "--hide-scrollbars"])
    return _br
def render(path, W, H, img=False, timeout=120):
    """path: svg file.  img=True embeds it via <img> in an html page (as a README would)."""
    global _br
    if img:
        d = os.path.dirname(os.path.abspath(path)); h = os.path.join(d, "_h.html")
        open(h, "w").write(f'<html><body style="margin:0;background:#000"><img src="{os.path.basename(path)}" width={W} height={H} style="display:block"></body></html>'); u = "file://" + h
    else: u = "file://" + os.path.abspath(path)
    ctx = _browser().new_context(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg = ctx.new_page(); t = time.time()
    try:
        pg.goto(u, timeout=timeout * 1000, wait_until="load")
        png = pg.screenshot(timeout=timeout * 1000, type="png"); dt = time.time() - t
    except Exception as e:
        try: ctx.close()
        except Exception: pass
        try: _br.close()
        except Exception: pass
        _br = None
        try: _pw.stop()
        except Exception: pass
        _pw = None; raise TimeoutError(f"render failed after {time.time()-t:.0f}s: {str(e)[:100]}")
    ctx.close()
    if img: os.remove(h)
    return np.array(Image.open(io.BytesIO(png)).convert("RGBA")), dt
