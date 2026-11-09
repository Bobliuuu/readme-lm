import sys, subprocess, re
font, var = sys.argv[1], sys.argv[2]
txt = sys.argv[3]
html = f"""<!doctype html><meta charset=utf-8><style>@font-face{{font-family:tok;src:url({font})}}
.T{{display:inline-block;font:1000px tok;white-space:normal;font-variation-settings:{var}}}</style>
<span class=T id=a>{txt}</span><span class=T id=b style="font-feature-settings:'calt' 1">{txt}</span><span class=T id=c>what is</span><pre id=o></pre>
<script>document.fonts.load('1000px tok').then(()=>{{document.getElementById('o').textContent=['a','b','c'].map(i=>document.getElementById(i).getBoundingClientRect().width).join(' ')}})</script>"""
open("probe.html","w").write(html)
r = subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome","--headless=new","--disable-gpu","--virtual-time-budget=3000","--dump-dom","file:///Users/jhome/readme-lm/trackC/cssfont/input/probe.html"],capture_output=True,text=True).stdout
print(re.search(r'<pre id="o">(.*?)</pre>',r,re.S).group(1))
