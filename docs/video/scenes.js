// Deterministic timeline driver: window.init(cfg) then window.setT(t) for any t (seconds).
const clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
const ease=x=>1-Math.pow(1-x,3);
let SC=[], CFG=null, GIF=[], LIVE=[];
// real captured frames of the issue flow: [file, sentence index, offset (s), elapsed label, caption]
const LIVE_META=[[0,0,0.1,'before Create','Scroll to "Try it now"'],[1,0,0.9,'before Create','Click the qa: link: form opens prefilled'],[2,0,1.7,'before Create','Title edited'],[3,0,2.5,'+1 s','Create clicked'],
 [4,0,3.3,'+19 s','Workflow run: in progress'],[5,1,0.0,'+35 s','Bot answered, issue closed'],[6,1,1.0,'+41 s','The answer, in the issue'],[7,1,2.4,'+105 s','Profile: image cache still old'],[8,1,3.6,'+391 s','Profile: new answer typing'],[9,1,4.8,'+399 s','Profile: full new answer']];
const WIKI=`{{#switch:{{#expr:5*{{#ifeq:{{#expr:<b>6*7+2*4+2*7+7*7+2*2+3*4</b>><b>2*1+3*2+7*2+4*3+7</b>}}|1|{{#expr:6*7+2*4+2*7+7*7+2*2+3*4-(2*1+3*2+7*2+4*3+7)}}|0}}+5*{{#ifeq:{{#expr:7*5+6*3+3*7+1*7+3*2>2*1+1*2+3*2+5*4+7*7+1*3+3*4+7}}|1|{{#expr:7*5+6*3+3*7+1*7+3*2-(2*1+1*2+3*2+5*4+7*7+1*3+3*4+7)}}|0}}+5*{{#ifeq:{{#expr:4*1+2*5+6*2+3*7+7*3+7*2>5*3+5*2+7*7+3*4+7*7+9}}|1|{{#expr:4*1+2*5+6*2+3*7+7*3+7*2-(5*3+5*2+7*7+3*4+7*7+9)}}|0}}+7*{{#ifeq:{{#expr:5*1+2*2+3*7+7*3+7*2>6*5+7*3+1*2+7*7+6*4+7*7+7*4+3}}…`;
// predictions of the real 2-char net (computed with demos/mediawiki/gen.py ref_forward) for the text "hello there"
const ROWS=[["he","␣"],["el","␣"],["ll","t"],["lo","t"],["o␣","t"],["␣t","h"],["th","e"],["he","␣"],["er","y"],["re","␣"]];
function build(){
  document.getElementById('wikicode').innerHTML=WIKI.replace(/<b>/g,'<span style="color:#3fb950">').replace(/<\/b>/g,'</span>').replace(/#expr/g,'<span style="color:#58a6ff">#expr</span>');
  const rows=document.getElementById('rows');
  ROWS.forEach((r,i)=>{const d=document.createElement('div');d.className='mono';d.setAttribute('data-s','1');d.setAttribute('data-d',(0.3+i*0.4).toFixed(2));d.setAttribute('data-a','up');
    d.style.cssText='display:flex;justify-content:space-between;font-size:20px;padding:5px 4px;border-bottom:1px solid #30363d';
    d.innerHTML=`<span>${r[0]}</span><span class="gr">${r[1]}</span><span class="gr">✓</span>`;rows.appendChild(d);});
  rows.querySelectorAll('div').forEach(d=>{});
  const toks=document.getElementById('toks');
  for(let i=0;i<24;i++){const t=document.createElement('div');t.className='tok';t.style.left=(64+i*48)+'px';t.style.top='540px';t.setAttribute('data-tok',i);toks.appendChild(t);}
}
async function init(cfg){
  CFG=cfg; build();
  // gif frames
  const box=document.getElementById('gifbox');
  for(const u of cfg.gifFrames){const im=new Image();im.src=u;await im.decode();im.style.cssText='position:absolute;left:0;top:0;display:none';box.appendChild(im);GIF.push(im);}
  const lbox=document.getElementById('livebox');
  for(const u of (cfg.liveFrames||[])){const im=new Image();im.src=u;await im.decode();im.style.cssText='position:absolute;left:0;top:0;width:800px;height:413px;display:none';lbox.appendChild(im);LIVE.push(im);}
  document.querySelectorAll('img').forEach(i=>i.decode&&i.decode());
  cfg.scenes.forEach(s=>{
    const el=document.getElementById('sc-'+s.id);
    const items=[...el.querySelectorAll('[data-s]')].map(e=>({e,s:+e.dataset.s,d:+(e.dataset.d||0),end:e.dataset.e!==undefined?+e.dataset.e:null,ed:+(e.dataset.ed||0),a:e.dataset.a||'up'}));
    const lits=[...el.querySelectorAll('[data-lit]')].map(e=>{const [a,b]=e.dataset.lit.split(':');return{e,s:+a,d:+(b||0)}});
    const types=[...el.querySelectorAll('[data-type]')].map(e=>({e,s:+e.dataset.type,d:+(e.dataset.d||0),txt:e.dataset.text}));
    el.querySelectorAll('[data-strike]').forEach(e=>{e.dataset.sd=e.dataset.d;});
    const strikes=[...el.querySelectorAll('.strike')].map(e=>({e,s:+e.closest('[data-s]').dataset.s,d:+e.dataset.d}));
    SC.push({...s,el,items,lits,types,strikes});
  });
}
function apply(it,lt,st){
  const t0=st[it.s]+it.d; let p=ease(clamp((lt-t0)/0.55)); let o=p;
  if(it.end!==null){const q=clamp((lt-st[it.end]-it.ed)/0.4); o=p*(1-q);}
  const e=it.e; e.style.opacity=o; e.style.visibility=o>0.001?'visible':'hidden';
  const k=1-p; let tr='';
  switch(it.a){case 'up':tr=`translateY(${28*k}px)`;break;case 'left':tr=`translateX(${-50*k}px)`;break;case 'right':tr=`translateX(${50*k}px)`;break;
   case 'scale':tr=`scale(${0.9+0.1*p})`;break;case 'grow':tr=`scaleX(${p})`;e.style.opacity=lt>=t0?(it.end!==null?o:1):0;break;
   case 'growy':tr=`scaleY(${p})`;e.style.opacity=lt>=t0?1:0;break;default:tr='';}
  e.style.transform=tr;
}
function setT(t){
  const total=CFG.scenes[CFG.scenes.length-1].end;
  let sc=SC.find(s=>t>=s.start&&t<s.end)||SC[SC.length-1];
  const lt=t-sc.start, dur=sc.end-sc.start;
  SC.forEach(s=>{if(s!==sc){s.el.style.opacity=0;s.el.style.visibility='hidden'}});
  const fi=clamp(lt/0.4), fo=clamp((dur-lt)/0.4);
  const last=sc===SC[SC.length-1];
  sc.el.style.visibility='visible'; sc.el.style.opacity=Math.min(fi,last?1:fo);
  const st=sc.st;
  sc.items.forEach(it=>apply(it,lt,st));
  sc.lits.forEach(l=>l.e.classList.toggle('lit',lt>=st[l.s]+l.d&&lt<st[l.s]+l.d+2.4));
  sc.types.forEach(y=>{const n=Math.floor(clamp((lt-st[y.s]-y.d)*30,0,1e9));const sh=y.txt.slice(0,n);y.e.textContent=sh;y.e.classList.toggle('caret',lt>=st[y.s]+y.d&&n<=y.txt.length);const vis=lt>=st[y.s]+y.d?1:0;});
  sc.strikes.forEach(s=>{s.e.style.setProperty('--k',ease(clamp((lt-st[s.s]-s.d)/0.4)))});
  if(sc.id==='title'){document.querySelectorAll('[data-tok]').forEach(e=>{const i=+e.dataset.tok;const ph=(lt*6-i*0.9);e.classList.toggle('lit',ph>0&&ph%24<3)});}
  if(sc.id==='a'&&GIF.length){const g=st[4]+1.0;if(lt>=g){const i=Math.floor((lt-g)*2.38*2.0)%GIF.length;GIF.forEach((im,j)=>im.style.display=j===i?'block':'none');}}
  if(sc.id==='live'&&LIVE.length){let cur=-1;LIVE_META.forEach((m,k)=>{if(lt>=st[m[1]]+m[2])cur=k;});
    LIVE.forEach((im,j)=>im.style.display=(cur>=0&&j===cur)?'block':'none');
    if(cur>=0){document.getElementById('livet').textContent=LIVE_META[cur][3];document.getElementById('livecap').textContent=LIVE_META[cur][4];}
    else{document.getElementById('livet').textContent='before Create';document.getElementById('livecap').textContent='';}}
  // captions
  const cap=document.getElementById('cap');let ci=-1;
  for(let i=0;i<st.length;i++){if(lt>=st[i]-0.05&&lt<sc.ends[i]+0.2)ci=i;}
  if(ci>=0){cap.textContent=sc.caps[ci];const a=clamp((lt-st[ci])/0.25);cap.style.opacity=Math.min(a,clamp((sc.ends[ci]+0.2-lt)/0.2));}else cap.style.opacity=0;
  document.getElementById('prog').style.width=(1280*t/total)+'px';
}
window.init=init;window.setT=setT;
