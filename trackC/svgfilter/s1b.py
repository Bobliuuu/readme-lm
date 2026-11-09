from h import *
# precision: x -> x*0.1 -> *10 in two primitives; and in ONE primitive with k1 form. Also a+b exact in one step.
def svg(a, body, ci="sRGB"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><defs>
<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="{ci}">
<feFlood flood-color="rgb({a},{a},{a})" result="A"/>{body}</filter></defs><rect width="8" height="8" filter="url(#f)"/></svg>'''
two='<feComposite in="A" in2="A" operator="arithmetic" k2="0.1" result="s"/><feComposite in="s" in2="s" operator="arithmetic" k2="10"/>'
one='<feComposite in="A" in2="A" operator="arithmetic" k2="1"/>'
for ci in ("sRGB","linearRGB"):
  res=[render(svg(a,two,ci),8,8,"s1b")[4,4,0] for a in range(0,256,17)]
  print(ci,"x*.1*10 :",res)
# does alpha-premult matter: flood-opacity
# float intermediate test: feComponentTransfer... compare with single step x*1.0 sums
print("single step k2=.1,k4=.0:",[render(svg(a,'<feComposite in="A" in2="A" operator="arithmetic" k2="0.1"/>'),8,8,"s1b")[4,4,0] for a in range(0,256,51)])
