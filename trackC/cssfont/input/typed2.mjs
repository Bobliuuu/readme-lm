import { spawn } from "node:child_process";
const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", ["--headless=new","--disable-gpu","--remote-debugging-port=9333","--user-data-dir=/private/tmp/claude-501/-Users-jhome/5779df58-76d8-49bb-9978-91f6c411f22c/scratchpad/prof","about:blank"], {stdio:"ignore"});
await new Promise(r=>setTimeout(r,2500));
const tabs = await (await fetch("http://127.0.0.1:9333/json")).json();
const ws = new WebSocket(tabs.find(t=>t.type==="page").webSocketDebuggerUrl);
await new Promise(r=>ws.onopen=r);
let id=0; const pend={}; ws.onmessage=e=>{const m=JSON.parse(e.data); if(m.id&&pend[m.id]) pend[m.id](m.result)};
const send=(method,params={})=>new Promise(r=>{const i=++id; pend[i]=r; ws.send(JSON.stringify({id:i,method,params}))});
await send("Page.enable"); await send("Page.navigate",{url:"file:///Users/jhome/readme-lm/trackC/cssfont/input/typed.html"});
await new Promise(r=>setTimeout(r,1500));
const ev=async x=>(await send("Runtime.evaluate",{expression:x,returnByValue:true})).result.value;
const w=()=>ev("[document.getElementById('p').getBoundingClientRect().width, document.getElementById('q').getBoundingClientRect().width]");
console.log(await ev("[...document.fonts].map(f=>f.family+f.status).join()+' '+getComputedStyle(document.getElementById('p')).fontVariationSettings+' '+document.activeElement.id+' ce='+document.getElementById('p').isContentEditable"));for(const t of [0.025,0.05,0.075,0.1]){await ev(`document.getElementById('p').style.fontVariationSettings='"POSN" ${t}'`);console.log(t,await w())}
console.log("initial 'hello' widths [id(char3='l')+1 , len]:", await w());
await ev("document.getElementById('p').focus(); const s=getSelection(); s.selectAllChildren(document.getElementById('p')); s.collapseToEnd(); 1");
// type with real key events
await send("Input.insertText",{text:" qwen"});
await new Promise(r=>setTimeout(r,300));
console.log("active",await ev("document.activeElement.tagName+document.activeElement.id+JSON.stringify(getSelection().toString())"));console.log("after typing ' qwen' into style[contenteditable]:", await ev("document.getElementById('p').textContent"), await w());
await ev("document.getElementById('q').focus(); const s=getSelection(); s.selectAllChildren(document.getElementById('q')); s.collapseToEnd(); 1");
await send("Input.insertText",{text:" qwen"});
console.log("len font after typing in its own editable:", await w());
chrome.kill(); process.exit(0);
