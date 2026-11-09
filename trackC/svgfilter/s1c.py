from h import *
import h, subprocess
fl=[f'<feFlood flood-color="rgb({a},{a},{a})"/><feComposite operator="arithmetic" k2="0.1" k4="0" in2="SourceGraphic" result="s"/><feComposite in="s" in2="s" operator="arithmetic" k2="10"/>' for a in range(0,256,5)]
# make fully opaque: use k2 with alpha preserved via k4? flood opacity1 -> alpha 0.1 ... use colormatrix to keep alpha
fl=[f'<feFlood flood-color="rgb({a},{a},{a})"/><feColorMatrix type="matrix" values="0.1 0 0 0 0 0 0.1 0 0 0 0 0 0.1 0 0 0 0 0 1 0" result="s"/><feColorMatrix in="s" type="matrix" values="10 0 0 0 0 0 10 0 0 0 0 0 10 0 0 0 0 0 1 0"/>' for a in range(0,256,5)]
exp=np.clip(np.round(np.round(np.arange(0,256,5)*.1)*10),0,255)
for eng in ("chrome","rsvg"):
    r=grid(fl,cols=52,name="s1c",engine=eng)[:,0].astype(int)
    print(eng,"exact-8bit-chain" if (abs(r-exp)<=2).all() else "differs", r[:12], "maxdev from 8bit-model",abs(r-exp).max(),"from ideal",abs(r-np.arange(0,256,5)).max())
# chrome with GPU enabled
CH=h.CH
p=h.D+"/work/s1c.svg"
out=h.D+"/work/s1c.gpu.png"
subprocess.run([CH,"--headless=new","--hide-scrollbars","--force-device-scale-factor=1",f"--screenshot={out}","--window-size=52,1","file://"+p],capture_output=True,timeout=60)
a=np.array(Image.open(out).convert("RGBA"))[0,:52,0].astype(int); print("chrome-gpu",a[:12],"maxdev ideal",abs(a-np.arange(0,256,5)).max())
