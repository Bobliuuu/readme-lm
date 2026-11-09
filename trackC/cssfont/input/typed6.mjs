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
console.log("initial 'hello' widths [id(char3='l')+1 , len]:", await w());
await send("Input.dispatchMouseEvent",{type:"mousePressed",x:5,y:5,button:"left",clickCount:1});await send("Input.dispatchMouseEvent",{type:"mouseReleased",x:5,y:5,button:"left",clickCount:1});await ev("getSelection().selectAllChildren(document.getElementById('p'));getSelection().collapseToEnd();1");console.log("active:",await ev("document.activeElement.tagName+document.activeElement.id"));
// type with real key events
await send("Input.insertText",{text:" qwen"});
await new Promise(r=>setTimeout(r,300));
console.log("after typing ' qwen' into style[contenteditable]:", await ev("document.getElementById('p').textContent"), await w());

await send("Input.insertText",{text:" qwen"});
const VOC=" abcdefghijklmnopqrstuvwxyz";
const got=[];for(let k=17;k<=22;k++){await ev(`document.getElementById('p').style.fontVariationSettings='"POSN" ${k/40}'`);got.push((await w())[0]-1)}
const exp=[...(" qwen"+"!")].map(c=>VOC.indexOf(c));
console.log("decoded ids of typed chars 17..22:",got," expected (' qwen' then past-end -1):",exp.slice(0,5).join(),"and -1");
chrome.kill(); process.exit(0);