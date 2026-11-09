"""Render every static diagram as an animated GIF + MP4 (docs/gifs/).
Usage: python3 docs/tools/make_gifs.py [name ...]
SVGs are animated generically: elements reveal in order, arrows draw, packets
flow along arrows, bars grow. PNG screenshots get a scanning wipe reveal."""
import sys, subprocess, shutil, pathlib
from playwright.sync_api import sync_playwright
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/gifs'; TMP = pathlib.Path('/tmp/rlm_frames')
FPS, DUR = 12, 6.0
SVGS = ['arch-three-versions','arch-fast-version','arch-qa-transformer','arch-where-arithmetic',
        'arch-mediawiki-net','time-per-answer','precision','gpt2-graph-coverage','qa-graph-compression']
PNGS = {'mediawiki-render':'top','svg-net-reference':'left'}
JS = r"""
(() => {
const svg = document.querySelector('svg');
const NS='http://www.w3.org/2000/svg';
const kids=[...svg.children].filter(e=>!['style','defs','title'].includes(e.tagName));
const bg=kids.shift();
let texts=0; const items=[]; const arrows=[];
for(const e of kids){
  const c=e.getAttribute('class')||'';
  if(e.tagName==='text'&&texts<2){texts++;continue;}
  if(c.includes('grid')) continue;
  if(e.tagName==='text'&&(c.includes('t2'))&&e.getAttribute('font-size')==='11') continue; // axis labels
  const isArrow=/\b(arr|dash)\b/.test(c);
  const o={e,isArrow,c};
  if(isArrow){try{o.len=e.getTotalLength();}catch(x){o.len=60;} arrows.push(o);}
  items.push(o);
}
const N=items.length, REV=3.4;
const gp=document.createElementNS(NS,'g'); svg.appendChild(gp);
const packets=arrows.map(a=>{const p=document.createElementNS(NS,'circle');p.setAttribute('r',4.5);p.setAttribute('fill','#eb6834');p.setAttribute('opacity',0);gp.appendChild(p);return p;});
const ease=x=>x<0?0:x>1?1:1-Math.pow(1-x,3);
window.setT=(t)=>{
 items.forEach((o,i)=>{
  const s=(i/N)*REV, p=ease((t-s)/0.45), e=o.e, c=o.c, tag=e.tagName;
  e.style.opacity = p;
  if(o.isArrow){ e.style.strokeDasharray=o.len; e.style.strokeDashoffset=o.len*(1-p); e.style.opacity=p>0?1:0; e.removeAttribute('marker-end'); if(!o.mk){o.mk=e.getAttribute('marker-end')||'';} }
  else if(tag==='rect'&&/\bc[12]\b/.test(c)&&!c.includes('node')){ const x=+e.getAttribute('x'),y=+e.getAttribute('y'),h=+e.getAttribute('height');
     e.style.transformBox='view-box'; e.style.transformOrigin=x+'px '+(y+h/2)+'px'; e.style.transform='scaleX('+p+')'; }
  else if(tag==='line'&&/\bs[12]\b/.test(c)){ const y1=+e.getAttribute('y1'); e.style.transformBox='view-box'; e.style.transformOrigin='0 '+y1+'px'; e.style.transform='scaleY('+p+')'; }
  else if(tag==='circle'||tag==='rect'){ // node pop + flash
   const bb=e.getBBox(); e.style.transformBox='fill-box'; e.style.transformOrigin='center'; e.style.transform='scale('+(0.92+0.08*p)+')';
   if(tag==='rect'&&c.includes('node')){const f=Math.max(0,1-Math.abs(t-s-0.3)/0.5); e.style.strokeWidth=1.5+2.5*f; e.style.stroke=f>0.05?'#eb6834':'';}
  }
 });
 arrows.forEach((a,j)=>{ const p=packets[j]; const ph=(((t-3.3-j*0.18)%1.5)+1.5)%1.5/1.1;
   if(t<3.3||ph>1){p.setAttribute('opacity',0);return;}
   const pt=a.e.getPointAtLength? a.e.getPointAtLength(a.len*ph):{x:0,y:0};
   p.setAttribute('cx',pt.x);p.setAttribute('cy',pt.y);p.setAttribute('opacity',1);});
 // arrowheads: show once arrow drawn
 arrows.forEach(a=>{const m=a.e.getAttribute('data-mk'); });
};
// restore arrowheads by marker: re-add after reveal
arrows.forEach(a=>{ a.mk=a.e.getAttribute('marker-end'); a.e.setAttribute('data-mk',a.mk||''); });
const _s=window.setT; window.setT=(t)=>{_s(t); arrows.forEach(a=>{ if(a.mk){ if(t>3.3) a.e.setAttribute('marker-end',a.mk); else a.e.removeAttribute('marker-end'); } });};
window.setT(0);
})();
"""
def frames(page, name):
    d=TMP/name; shutil.rmtree(d,ignore_errors=True); d.mkdir(parents=True)
    n=int(FPS*DUR)
    for i in range(n):
        page.evaluate(f'window.setT({i/FPS})'); page.screenshot(path=str(d/f'f{i:03d}.png'))
    return d
def encode(d,name,w):
    OUT.mkdir(exist_ok=True)
    pal=d/'pal.png'
    subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(FPS),'-i',str(d/'f%03d.png'),'-vf',f'scale={w}:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=96',str(pal)],check=True)
    subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(FPS),'-i',str(d/'f%03d.png'),'-i',str(pal),'-lavfi',f'scale={w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle','-loop','0',str(OUT/f'{name}.gif')],check=True)
    subprocess.run(['ffmpeg','-y','-v','error','-framerate',str(FPS),'-stream_loop','1','-i',str(d/'f%03d.png'),'-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p','-c:v','libx264','-crf','27','-movflags','+faststart','-an',str(OUT/f'{name}.mp4')],check=True)
def main():
    only=sys.argv[1:]
    with sync_playwright() as p:
        b=p.chromium.launch(channel='chrome')
        for n in SVGS:
            if only and n not in only: continue
            svg=(ROOT/f'docs/img/{n}.svg').read_text()
            import re; m=re.search(r'viewBox="0 0 (\d+) (\d+)"',svg); w,h=int(m[1]),int(m[2])
            pg=b.new_page(viewport={'width':w,'height':h},device_scale_factor=1,color_scheme='light')
            pg.set_content(f'<html><body style="margin:0;background:#fcfcfb">{svg}</body></html>'); pg.evaluate(JS)
            encode(frames(pg,n),n,min(w,860)); pg.close(); print('done',n)
        for n,mode in PNGS.items():
            if only and n not in only: continue
            src=ROOT/f'docs/img/{n}.png'
            from PIL import Image; w,h=Image.open(src).size
            pg=b.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
            pg.set_content(f'<html><body style="margin:0;background:#fff;position:relative;overflow:hidden"><img id=i src="data:image/png;base64,{__import__('base64').b64encode(src.read_bytes()).decode()}" style="display:block"><div id=bar style="position:absolute;background:#eb6834;opacity:.85"></div></body></html>')
            pg.wait_for_function('document.getElementById("i").complete')
            top = mode=='top'
            pg.evaluate("""([w,h,top])=>{window.setT=(t)=>{const f=Math.min(1,Math.max(0,(t-0.4)/4.2)),e=1-Math.pow(1-f,2);
              const i=document.getElementById('i'),b=document.getElementById('bar');
              if(top){i.style.clipPath=`inset(0 0 ${h*(1-e)}px 0)`;b.style.cssText+=`;left:0;width:${w}px;height:3px;top:${h*e}px;opacity:${f<1?.85:0}`}
              else{i.style.clipPath=`inset(0 ${w*(1-e)}px 0 0)`;b.style.cssText+=`;top:0;height:${h}px;width:3px;left:${w*e}px;opacity:${f<1?.85:0}`}};setT(0)}""",[w,h,top])
            encode(frames(pg,n),n,min(w,860)); pg.close(); print('done',n)
        b.close()
main()
