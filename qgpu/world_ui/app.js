"use strict";
const $ = id => document.getElementById(id);
const palette = ["#19221a", "#ccf58b", "#ffaf74", "#89c6ef", "#d3a6f2"];
let state = null, busy = false, auto = false, tokens = 0, uploads = 0;
let replayTrace = null, replayIndex = 0, pixels = Array(64).fill(0);
let cells = [], flashes = new Set(), flashing = false;
const canvas = $("arena"), ctx = canvas.getContext("2d");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function notice(message = "") {
  $("notice").textContent = message;
  $("notice").classList.toggle("visible", Boolean(message));
}
function render() {
  ctx.fillStyle = "#0e130f"; ctx.fillRect(0, 0, 640, 640);
  for (let i = 0; i < 64; i++) {
    const x = i % 8 * 80, y = Math.floor(i / 8) * 80;
    ctx.fillStyle = pixels[i] ? palette[pixels[i]] || palette[1] : "#151e16";
    ctx.fillRect(x + 3, y + 3, 74, 74);
    if (pixels[i]) {
      ctx.fillStyle = "#26341e"; ctx.font = "15px ui-monospace, monospace";
      ctx.fillText(String(pixels[i]), x + 13, y + 24);
      ctx.fillStyle = "#ffffff55"; ctx.fillRect(x + 59, y + 58, 6, 6);
    } else {
      ctx.fillStyle = "#364332"; ctx.fillRect(x + 39, y + 39, 2, 2);
    }
  }
}
function paintMemory(snapshot) {
  const words = [...snapshot.positions, ...snapshot.velocities, ...snapshot.pixels];
  cells.forEach((cell, i) => {
    cell.title = `word ${i}: ${words[i]}`;
    if (i >= 16) cell.style.background = palette[words[i]] || palette[0];
  });
}
function flashMemory(written) {
  Object.keys(written).forEach(key => {
    const cell = cells[Number(key)];
    if (cell) { cell.classList.add("flash"); flashes.add(cell); }
  });
  if (!flashing) {
    flashing = true;
    setTimeout(() => { flashes.forEach(cell => cell.classList.remove("flash")); flashes.clear(); flashing = false; }, 120);
  }
}
function provenance(backend, recorded = false) {
  const model = backend === "model";
  $("top-source").textContent = `${model ? "MODEL" : "REFERENCE"}${recorded ? " REPLAY" : model ? ` / ${(state?.device || "LOADING").toUpperCase()}` : " / CPU"}`;
  $("backend").textContent = `${model ? "MODEL" : "REFERENCE"}${recorded ? " RECORDING" : " MODE"}`;
  document.querySelector(".mode-card").classList.toggle("model", model);
  $("provenance").textContent = recorded
    ? "Trace replay. Execution consistency is checked; the recorded backend claim is not authenticated."
    : model ? "Merged Qwen executes the kernels. The host compiles, verifies and displays."
    : "CPU interpreter. Circuit residency tests pass; full merged-model inference is a separate check.";
  $("connection").textContent = recorded ? "REPLAY / NO INFERENCE" : model ? `LOCAL MODEL / ${state?.device || "LOADING"}` : "LOCAL REFERENCE / NO MODEL";
}
function particles(snapshot) {
  const parent = $("particles"); parent.replaceChildren();
  for (let i = 0; i < 4; i++) {
    const row = document.createElement("div"); row.className = "particle-row";
    const name = document.createElement("strong"), dot = document.createElement("i");
    dot.style.background = palette[i + 1]; name.append(dot, `PARTICLE ${i + 1}`);
    const values = document.createElement("span");
    values.textContent = `${snapshot.positions.slice(i * 2, i * 2 + 2).join(",")} / ${snapshot.velocities.slice(i * 2, i * 2 + 2).join(",")}`;
    row.append(name, values); parent.append(row);
  }
}
function showStatus(next, recorded = false) {
  state = next; pixels = next.snapshot.pixels;
  render(); paintMemory(next.snapshot); particles(next.snapshot);
  $("frame-label").textContent = `FRAME ${String(next.frame).padStart(3, "0")}`;
  $("seed").value = next.seed;
  provenance(next.backend, recorded);
  $("arena-caption").textContent = next.frame ? `Frame ${next.frame} · complete` : "Initialized · step to render";
  if (next.last) {
    $("phase-state").textContent = "STEP COMPLETE";
    document.querySelectorAll(".phase").forEach(el=>{el.classList.remove("active");el.classList.add("done");});
    $("uploads").textContent = next.last.host_memory_words;
    $("resident-tag").textContent = next.last.host_memory_words === 0 ? "STATE PRESERVED" : "INITIAL UPLOAD";
    $("tokens").textContent = next.last.generated_tokens;
    $("verification").querySelector(".check").textContent = "✓";
    $("verification").querySelector("strong").textContent = recorded ? "Replay matches reference" : next.backend === "model" ? "Model tokens + memory verified" : "Reference step complete";
    $("verification").querySelector(":scope > div > span").textContent = `${next.last.generated_tokens} execution tokens · ${next.last.seconds.toFixed(3)} s ${recorded ? "recorded runtime" : "compute time"}`;
  } else {
    $("tokens").textContent="0"; $("uploads").textContent="—"; $("resident-tag").textContent="WAITING";
    $("phase-state").textContent="AWAITING STEP"; $("warp").textContent="—"; $("block").textContent="—";
    $("verification").querySelector(".check").textContent="○";
    $("verification").querySelector("strong").textContent="Ready for verification";
    $("verification").querySelector(":scope > div > span").textContent="Every launch is checked against the reference.";
    document.querySelectorAll(".phase").forEach(el=>el.classList.remove("active","done"));
  }
  updateControls();
}
function updateControls() {
  const remaining = replayTrace ? Math.max(0,(replayTrace.records.length-replayIndex)/3) : null;
  ["step", "reset", "export", "replay", "seed"].forEach(id => $(id).disabled = busy || auto);
  $("step").disabled = busy || auto || remaining === 0;
  $("seed").disabled = busy || auto || Boolean(replayTrace);
  $("run").disabled = (busy && !auto) || (remaining === 0 && !auto);
  $("run").textContent = auto ? "Stop after step" : replayTrace ? `Replay ${Math.min(4,remaining)} frames` : "Run 4 steps";
  $("step").firstChild.textContent = busy ? "Executing… " : remaining === 0 ? "Replay complete " : replayTrace ? "Replay next frame " : "Step world ";
  $("reset").textContent = replayTrace ? "↺ Exit replay" : "↺ Reset";
}
function consume(event) {
  const d = event.data;
  if (event.kind === "error") throw new Error(d.message);
  if (event.kind === "phase") {
    if (d.name === "clear") {
      tokens = 0; uploads = 0;
      document.querySelectorAll(".phase").forEach(el => el.classList.remove("done", "active"));
    }
    uploads += d.host_memory_words;
    document.querySelectorAll(".phase").forEach(el => el.classList.toggle("active", el.dataset.phase === d.name));
    $("phase-state").textContent = `${d.name.toUpperCase()} / RUNNING`;
    $("uploads").textContent = uploads;
    $("resident-tag").textContent = uploads ? "INITIAL UPLOAD" : "STATE PRESERVED";
    $("arena-caption").textContent = "Step executing · last complete frame shown";
    $("event").textContent = `${d.host_tokens} host tokens · ${d.host_memory_words} world words uploaded`;
  } else if (event.kind === "token") {
    tokens++;
    $("tokens").textContent = tokens;
    $("warp").textContent = d.warp ?? "—";
    $("block").textContent = d.block ?? "—";
    $("event").textContent = d.last;
    flashMemory(d.written || {});
  } else if (event.kind === "phase_done") {
    const el = document.querySelector(`[data-phase="${d.name}"]`);
    el.classList.remove("active"); el.classList.add("done");
    $("phase-state").textContent = `${d.name.toUpperCase()} / VERIFIED`;
  } else if (event.kind === "frame") {
    showStatus(d, Boolean(replayTrace));
    $("phase-state").textContent = "STEP COMPLETE";
  }
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  if (!response.ok) { const error = await response.json(); throw new Error(error.error || `HTTP ${response.status}`); }
  return response;
}

// The replay decoder mirrors the public qwenGPU record format. Physics never runs here.
function decoder(program) {
  let queue = [], block = 0, warp = 0, addr = 0, lo = 0;
  const regs = new Map(), written = {};
  let last = "Replay started";
  function exec(b, w) {
    block = b; warp = w;
    queue = program.live[b].flatMap(([, reg]) => Array.from({length:4}, (_, lane) => ["reg",reg,lane,0]).flatMap(a => [a,[...a.slice(0,3),1]]));
    if (program.writeback[b]) for (let lane=0; lane<4; lane++) queue.push(["addr",lane],["lo",lane],["hi",lane]);
    last = `warp ${w} runs block ${b}: ${program.stages[b].filter(op=>op!=="nop").join(" ")}`;
  }
  exec(0,0);
  return token => {
    if (token >= 65536 && token < 67584) {
      const v = token - 65536, b = v & 31, w = v >> 5 & 31;
      if (w >= program.warps) { block = null; warp = null; last = "launch done"; }
      else exec(b,w);
    } else if (queue.length) {
      const [kind, reg, lane, high] = queue.shift();
      if (kind === "reg") {
        const key = `${warp},${reg},${lane}`, old = regs.get(key) || 0;
        const value = high ? ((old & 65535) | (token << 16)) >>> 0 : ((old & 0xffff0000) | token) >>> 0;
        regs.set(key, value); if (high) last = `warp ${warp} lane ${lane}: r${reg} = ${value | 0}`;
      } else if (kind === "addr") addr = token - 68608;
      else if (kind === "lo") lo = token;
      else if (addr < 127) { written[addr] = (lo | token << 16) >>> 0; last = `lane ${reg} writes memory word ${addr}`; }
    }
    return {t:token, block, warp, last, written};
  };
}
async function replayStep() {
  if (replayIndex + 3 > replayTrace.records.length) throw new Error("End of recording. Exit replay to return to the local runtime.");
  const phases = [];
  for (let i=0; i<3; i++) {
    const record = replayTrace.records[replayIndex++];
    consume({kind:"phase", data:{name:record.phase, program:record.program, host_tokens:record.host.length, host_memory_words:record.host_memory_words}});
    const feed = decoder(record.program);
    for (let j=0; j<record.generated.length; j++) {
      consume({kind:"token", data:feed(record.generated[j])});
      if (j % 8 === 0) await sleep(6);
    }
    consume({kind:"phase_done", data:{...record.summary, name:record.phase}}); phases.push(record.summary);
  }
  const last = replayTrace.records[replayIndex - 1];
  const summary = {generated_tokens:phases.reduce((n,p)=>n+p.generated_tokens,0), host_memory_words:phases.reduce((n,p)=>n+p.host_memory_words,0), seconds:phases.reduce((n,p)=>n+p.seconds,0)};
  consume({kind:"frame", data:{backend:replayTrace.backend, frame:last.frame, seed:replayTrace.seed, snapshot:last.snapshot, last:summary, device:"REPLAY"}});
}
async function step() {
  busy = true; notice(); updateControls();
  try {
    if (replayTrace) return await replayStep();
    const response = await api("/api/step", {}), reader = response.body.getReader(), decode = new TextDecoder();
    let pending = "", count = 0, completed = false;
    while (true) {
      const {done,value} = await reader.read();
      pending += decode.decode(value, {stream:!done});
      let end;
      while ((end = pending.indexOf("\n")) >= 0) {
        const line = pending.slice(0,end); pending = pending.slice(end+1);
        if (line) { const event = JSON.parse(line); consume(event); if (event.kind === "frame") completed = true; }
        if (++count % 32 === 0) await sleep(8);
      }
      if (done) break;
    }
    if (!completed) throw new Error("The stream ended before a complete frame. Reset before continuing.");
  } finally { busy = false; updateControls(); }
}
async function downloadTrace() {
  const trace = replayTrace || await (await api("/api/trace")).json();
  const blob = new Blob([JSON.stringify(trace,null,2)],{type:"application/json"}), url = URL.createObjectURL(blob);
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = `weightworld-${trace.backend}-seed${trace.seed}.json`; anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("step").onclick = () => step().catch(e=>notice(e.message));
$("run").onclick = async () => {
  if (auto) { auto=false; updateControls(); return; }
  auto=true; updateControls();
  const count = replayTrace ? Math.min(4,(replayTrace.records.length-replayIndex)/3) : 4;
  try { for (let i=0; i<count && auto; i++) { await step(); if (auto && i<count-1) await sleep(600); } }
  catch(e) { notice(e.message); }
  finally { auto=false; updateControls(); }
};
$("reset").onclick = async () => {
  try {
    if (replayTrace) { replayTrace=null; showStatus(await (await api("/api/status")).json()); notice(); return; }
    else { const seed=Number($("seed").value); if (!Number.isInteger(seed) || seed<0 || seed>1000000) throw new Error("Seed must be 0..1000000"); showStatus(await (await api("/api/reset",{seed})).json()); }
    tokens=0; $("tokens").textContent="0"; $("uploads").textContent="—"; $("resident-tag").textContent="WAITING";
    $("phase-state").textContent="AWAITING STEP"; $("warp").textContent="—"; $("block").textContent="—"; $("event").textContent="No execution yet.";
    document.querySelectorAll(".phase").forEach(el=>el.classList.remove("active","done"));
    $("verification").querySelector("strong").textContent="Ready for verification"; $("verification").querySelector(".check").textContent="○";
    $("verification").querySelector(":scope > div > span").textContent="Every launch is checked against the reference.";
    notice();
  } catch(e) { notice(e.message); }
};
$("export").onclick = () => downloadTrace().catch(e=>notice(e.message));
$("replay").onclick = () => $("trace-file").click();
$("trace-file").onchange = async event => {
  try {
    const file=event.target.files[0]; if (!file) return;
    if (file.size>4000000) throw new Error("Choose a trace smaller than 4 MB.");
    const trace=JSON.parse(await file.text());
    await api("/api/verify",trace);
    if (!trace.records.length || trace.records.length%3) throw new Error("Replay needs complete frames.");
    replayTrace=trace; replayIndex=0;
    const mem=trace.initial_memory, signed=v=>(v|0);
    showStatus({backend:trace.backend,frame:0,seed:trace.seed,device:"REPLAY",snapshot:{positions:Array.from({length:8},(_,i)=>signed(mem[i])),velocities:Array.from({length:8},(_,i)=>signed(mem[8+i])),pixels:Array(64).fill(0)}},true);
    notice("Recording checked against this checkout’s kernels. Use Replay next frame. Backend provenance is a claim in the file, not an authenticated attestation.");
  } catch(e) { notice(e.message); }
  finally { event.target.value=""; }
};
$("capture").onclick = () => { document.body.classList.toggle("focus"); $("capture").textContent=document.body.classList.contains("focus")?"Full view ↙":"Focus view ↗"; };
for(let i=0;i<80;i++){const cell=document.createElement("div");cell.className=`memory-cell${i<16?" state":""}`;cells.push(cell);$("memory").append(cell);}
render();
api("/api/status").then(r=>r.json()).then(showStatus).catch(e=>{notice(`Local runtime unavailable: ${e.message}`);$("backend").textContent="DISCONNECTED";["step","run","reset"].forEach(id=>$(id).disabled=true);});
