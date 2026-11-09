from h import *
# S1a: a+b, a*b via feComposite with flood constants
def svg(a,b,lin):
    ci="linearRGB" if lin else "sRGB"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><defs>
<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="{ci}">
<feFlood flood-color="rgb({a},{a},{a})" result="A"/><feFlood flood-color="rgb({b},{b},{b})" result="B"/>
<feComposite in="A" in2="B" operator="arithmetic" k1="0" k2="0.5" k3="0.5" k4="0" result="S"/>
<feComposite in="A" in2="B" operator="arithmetic" k1="1" k2="0" k3="0" k4="0"/>
</filter></defs><rect width="8" height="8" filter="url(#f)"/></svg>'''
for lin in (False,True):
  for a,b in [(255,255),(200,100),(128,128),(100,50),(10,10),(64,200)]:
    px=render(svg(a,b,lin),8,8,"s1a")[4,4]
    print("lin" if lin else "srgb",a,b,"prod px",px[:3],"expect",round(a*b/255,2))
