"use strict";
const $=id=>document.getElementById(id), canvas=$("game"), ctx=canvas.getContext("2d");
const colors=["#0c140f","#4b5d43","#ffb47b","#c5f888"];
let live=null, shown=null, busy=false, running=false, held="stay", activeTick=null, replay=null, replayIndex=0, hasRecording=false;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const packets=[];
function notice(text=""){ $("notice").textContent=text; $("notice").classList.toggle("visible",Boolean(text)); }
function draw(pixels){
 ctx.fillStyle="#080e0a";ctx.fillRect(0,0,640,640);
 pixels.forEach((value,index)=>{const x=index%8*80,y=Math.floor(index/8)*80;ctx.fillStyle=colors[value]||colors[0];ctx.fillRect(x+3,y+3,74,74);if(value===2){ctx.fillStyle="#ffe8ce";ctx.fillRect(x+14,y+14,16,16);}if(!value){ctx.fillStyle="#263a2a";ctx.fillRect(x+39,y+39,2,2);}});
}
function mode(meta,device,isReplay=false){
 const merged=meta.kind==="merged",standalone=meta.kind==="standalone";
 document.title=merged?"PixelCore — WeightPong inside Qwen":standalone?"PixelCore — standalone game circuit":"PixelCore — reference preview";
 $("credits").textContent="PIXELCORE / WEIGHTPONG · COMPILER: QWENPUTER"+(merged?" · BASE: QWEN3.5-2B":standalone?" · CIRCUIT-ONLY MODEL":" · REFERENCE INTERPRETER");
 $("source-badge").textContent=isReplay?(merged?"QWEN RECORDING · 4×":"CIRCUIT RECORDING · 4×"):merged?`QWEN 2B · LIVE ${device.toUpperCase()}`:standalone?`CIRCUIT ONLY · ${device.toUpperCase()}`:"CPU REFERENCE · NO MODEL";
 $("mode-name").textContent=isReplay?"Verified recording":merged?"Qwen3.5-2B + PixelCore":standalone?"Standalone game circuit":"Reference interpreter";
 $("mode-detail").textContent=isReplay?"Playback of an exported run at 4× speed. No inference is happening during replay.":merged?"Full merged Qwen inference. Game state and pixel tokens come from its weights; the host checks and displays them.":standalone?"The same game circuit as a 54 MB transformer. No pretrained Qwen language backbone is loaded.":"A development preview using Python reference rules. This mode is not model inference.";
 $("layer-count").textContent=merged?"8 CORE + 24 QWEN":standalone?"8 COMPILED GAME LAYERS":"REFERENCE RULES";
 $("architecture-detail").textContent=merged?"Full Qwen mode retains 24 pretrained language layers. Eight added layers implement the game rules and pixel generator.":standalone?"Circuit-only mode runs emitted bf16 transformer weights, without the pretrained language model. Its timings must be reported separately.":"The interpreter checks the circuit specification. Choose a checkpoint at server startup for actual model execution.";
}
function controls(){
 $("play").textContent=running?"Ⅱ Pause":replay?"▶ Replay":"▶ Play";
 $("play").disabled=Boolean(shown?.state?.over||shown?.state?.won)&&!replay;
 $("step").disabled=busy||running||Boolean(shown?.state?.over||shown?.state?.won)||(replay&&replayIndex>=replay.records.length);
 $("new").disabled=false;$("export").disabled=busy;$("replay").disabled=busy||running||!hasRecording;
 $("assist").disabled=Boolean(replay);
}
function show(data,isReplay=false){
 shown=data;if(!isReplay)live=data;
 mode(data.meta,data.device,isReplay);draw(data.pixels);
 $("score").firstChild.textContent=`${data.state.score} `;
 $("state-bits").textContent=data.state.packed.toString(2).padStart(14,"0");
 $("ball").textContent=`${data.state.x}, ${data.state.y}`;$("paddle").textContent=data.state.paddle;
 $("frame-label").textContent=`TICK ${String(data.ticks).padStart(3,"0")}${isReplay?" / RECORDED":""}`;
 $("game-state").textContent=data.state.won?"YOU WIN":data.state.over?"GAME OVER":data.last?"IN PLAY":"READY";
 const overlay=$("screen-overlay");overlay.classList.toggle("hidden",Boolean(data.last)&&!data.state.over&&!data.state.won);
 overlay.querySelector("strong").textContent=data.state.won?"YOU WIN!":data.state.over?"GAME OVER":"READY?";
 overlay.querySelector("span").textContent=data.state.won?"Three saves. All computed by the circuit.":data.state.over?"The ball got past you. Try again.":"Keep the ball alive. Three saves to win.";
 if(data.last){$("time").textContent=`${data.last.seconds.toFixed(2)} s`;$("token-count").textContent=data.last.tokens;$("check-icon").textContent="✓";$("check-title").textContent=isReplay?"Recording matches the reference":data.meta.kind==="reference"?"Reference tick complete":"Tokens + recurrent state verified";$("check-detail").textContent=isReplay?"Backend provenance is a claim in the file; this is the recorded run’s compute time.":data.meta.kind==="reference"?"Reference execution only. No model or recurrent model head was loaded.":"Every emitted token matched. The model’s recurrent state was also checked.";}
 else{$("token-count").textContent="0";$("time").textContent="—";$("check-icon").textContent="○";$("check-title").textContent="Lock checker ready";}
 if(data.last&&!busy){packets.forEach(p=>p.classList.add("done"));$("live-state").textContent=isReplay?"REPLAY / 4×":"EXACT MATCH";$("event").textContent="Last complete frame verified; waiting for the next button.";}
 if(data.state.over||data.state.won)running=false;
 controls();
}
function unpack(value){return{x:value&7,y:value>>3&7,vx:value>>6&1,vy:value>>7&1,paddle:value>>8&7,score:value>>11&3,over:value>>13&1,won:(value>>11&3)===3,packed:value};}
async function api(path,body){const response=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});if(!response.ok){const e=await response.json();throw new Error(e.error||`HTTP ${response.status}`);}return response;}
function event(item){
 if(item.kind==="error")throw new Error(item.data.message);
 if(item.kind==="start"){$("token-count").textContent="0";packets.forEach(p=>p.classList.remove("done"));$("live-state").textContent="DECODING";$("event").textContent=`Button: ${item.data.action||"initial state"} · awaiting model output`;}
 if(item.kind==="token"){const d=item.data;$("token-count").textContent=d.index;if(d.chunk!==null&&d.chunk>=0&&d.chunk<16)packets[d.chunk].classList.add("done");$("event").textContent=d.token<4096?`PIXEL packet ${d.token>>8}: 0x${(d.token&255).toString(16).padStart(2,"0")}`:d.token<32768?`STATE: 0x${(d.token-16384).toString(16).padStart(4,"0")}`:"FRAME END";}
 if(item.kind==="frame"){show(item.data);$("live-state").textContent="EXACT MATCH";}
}
async function playbackTick(){
 if(replayIndex>=replay.records.length){running=false;return;}
 const record=replay.records[replayIndex++];packets.forEach(p=>p.classList.add("done"));
 const data={meta:replay.meta,device:replay.device,state:unpack(record.after),pixels:record.pixels,ticks:Math.max(0,replayIndex-1),last:{seconds:record.seconds,tokens:record.tokens,verified:record.verified}};
 show(data,true);$("live-state").textContent="REPLAY / 4×";$("event").textContent=`Recorded button: ${record.action||"initial state"} · ${record.tokens} exact tokens`;
 await sleep(Math.max(80,record.seconds*1000/4));if(replayIndex>=replay.records.length)running=false;
}
function tick(override){
 if(busy)return Promise.reject(new Error("Wait for this tick to finish"));
 busy=true;notice();controls();
 activeTick=(async()=>{
  try{
   if(replay)return await playbackTick();
   let action=override!==undefined?override:held;
   if(!live?.last)action=null;
   else if(override===undefined&&$("assist").checked)action=live.state.paddle>live.state.x?"left":live.state.paddle<live.state.x?"right":"stay";
   const response=await api("/api/tick",{action}),reader=response.body.getReader(),decoder=new TextDecoder();
   let pending="",complete=false;
   while(true){const{done,value}=await reader.read();pending+=decoder.decode(value,{stream:!done});let end;while((end=pending.indexOf("\n"))>=0){const line=pending.slice(0,end);pending=pending.slice(end+1);if(line){const item=JSON.parse(line);event(item);if(item.kind==="frame")complete=true;}}if(done)break;}
   if(!complete)throw new Error("The tick ended without a complete frame. Restart the session.");
  }finally{busy=false;controls();}
 })();return activeTick;
}
async function play(){if(running){running=false;controls();return;}running=true;controls();try{while(running){await tick();if(running)await sleep(80);}}catch(e){running=false;notice(e.message);}finally{running=false;controls();}}
$("play").onclick=play;
$("step").onclick=()=>tick().catch(e=>notice(e.message));
$("new").onclick=async()=>{running=false;controls();try{if(activeTick)await activeTick.catch(()=>{});if(replay){replay=null;replayIndex=0;show(await(await api("/api/status")).json());}if(live.failed){live=await(await api("/api/restart",{})).json();}await tick(live.last?"reset":null);}catch(e){notice(e.message);}};
for(const [id,action]of[["left","left"],["right","right"]]){const el=$(id);el.onpointerdown=e=>{e.preventDefault();held=action;el.classList.add("held");el.setPointerCapture(e.pointerId);};el.onpointerup=el.onpointercancel=()=>{held="stay";el.classList.remove("held");};}
document.addEventListener("keydown",e=>{if(["INPUT","BUTTON"].includes(document.activeElement?.tagName)&&e.code==="Space")return;if(e.code==="ArrowLeft"||e.code==="ArrowRight"){e.preventDefault();held=e.code==="ArrowLeft"?"left":"right";$(held).classList.add("held");}if(e.code==="Space"){e.preventDefault();if(!e.repeat)play();}});
document.addEventListener("keyup",e=>{if(e.code==="ArrowLeft"||e.code==="ArrowRight"){held="stay";$("left").classList.remove("held");$("right").classList.remove("held");}});
window.addEventListener("blur",()=>{held="stay";$("left").classList.remove("held");$("right").classList.remove("held");});
$("replay").onclick=async()=>{try{replay=await(await api("/api/recording")).json();replayIndex=0;running=false;notice("Playing the verified recording at 4×. The original per-frame compute time remains visible.");await play();}catch(e){notice(e.message);}};
$("export").onclick=async()=>{try{const trace=replay||await(await api("/api/trace")).json();const url=URL.createObjectURL(new Blob([JSON.stringify(trace,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=`pixelcore-${trace.meta.kind}-trace.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){notice(e.message);}};
$("focus").onclick=()=>{document.body.classList.toggle("focus");$("focus").textContent=document.body.classList.contains("focus")?"Full view ↙":"Focus ↗";};
for(let i=0;i<16;i++){const p=document.createElement("span");p.className="packet";p.textContent=String(i).padStart(2,"0");packets.push(p);$("packets").append(p);}
draw(Array(64).fill(0));
api("/api/status").then(r=>r.json()).then(data=>{hasRecording=data.has_recording;show(data);if(data.meta.kind==="merged"){$("other-mode").href="http://127.0.0.1:8767";$("other-mode").textContent="Standalone circuit mode ↗";$("other-mode").hidden=false;}else if(data.meta.kind==="standalone"){$("other-mode").href="http://127.0.0.1:8766";$("other-mode").textContent="Full Qwen mode ↗";$("other-mode").hidden=false;}}).catch(e=>{notice(e.message);$("source-badge").textContent="DISCONNECTED";["play","step","new"].forEach(id=>$(id).disabled=true);});
