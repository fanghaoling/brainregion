"""GridWorld 场景查看页(Phase C+):专用页面渲染 env.step 事件的网格,实时 + 可回看。

区别于 BrainRegion 调试面板(/,显示脑区/模型调用,env.step 只作 JSON 进时间线),本页(/scene)
把每帧的网格**按字符着色渲染**成大格图(@ 蓝/G 绿/# 墙/. 地/? 灰),live SSE 实时刷新,并缓存所有帧
供播放/单步/拖动 —— run 结束 server 仍活时回看,或页面已加载则关 server 仍可拖。

复用 debug_server 的 SSE 事件流(/api/events + /api/events/stream);事件 payload 由 loop._emit_env_step
发(`frame`=env.render() 累积图,viewer 友好)。页面无 build-time meta(server 与 run 解耦),全部从事件
派生(网格大小/动作/reward/explored)。
"""
from __future__ import annotations

from html import escape


def build_env_scene_html() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GridWorld 场景查看</title>
<style>
:root {
  --bg: #f5f6f2; --panel: #ffffff; --ink: #17201a; --muted: #667065; --line: #dfe4dc;
  --blue: #34699a; --green: #2c7a4b; --walld: #4a4f4a; --floor: #eef2ea; --fog: #c7ccc4;
  --agent: #34699a; --goal: #2c7a4b;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; }
header { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 12px 20px; border-bottom: 1px solid var(--line); background: #eef2ea; }
h1 { margin: 0; font-size: 16px; font-weight: 700; }
.links { color: var(--muted); font-size: 12px; }
.links a { color: var(--blue); }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 14px; padding: 14px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
#grid { display: grid; gap: 2px; justify-content: start; }
.cell { width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;
  font: 600 16px ui-monospace, monospace; border-radius: 4px; }
.c-A { background: var(--agent); color: #fff; }      /* @ agent */
.c-G { background: var(--goal); color: #fff; }       /* G goal */
.c-W { background: var(--walld); color: #eee; }      /* # wall */
.c-F { background: var(--floor); color: transparent; } /* . floor */
.c-Q { background: var(--fog); color: transparent; }   /* ? fog */
.side h2 { margin: 0 0 8px; font-size: 12px; color: var(--muted); text-transform: uppercase; }
.meta { font-size: 13px; }
.meta .row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed var(--line); }
.meta b { font-variant-numeric: tabular-nums; }
.controls { display: grid; gap: 8px; margin-top: 10px; }
.btns { display: flex; gap: 6px; }
button { flex: 1; padding: 7px; border: 1px solid var(--line); border-radius: 6px; background: #fff;
  cursor: pointer; font: inherit; }
button:hover { background: #eef2ea; }
input[type=range] { width: 100%; }
.status { color: var(--muted); font-size: 12px; margin-top: 8px; min-height: 18px; }
.legend { color: var(--muted); font-size: 12px; margin-top: 10px; line-height: 1.7; }
.legend span { display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: middle; margin-right: 3px; }
@media (max-width: 800px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>GridWorld 场景查看</h1>
  <div class="links"><a href="/">调试面板</a> · <a href="/scene">场景</a> · <a href="/runs">过去场次</a></div>
</header>
<div class="layout">
  <div class="panel">
    <div id="grid"></div>
  </div>
  <div class="panel side">
    <h2>当前步</h2>
    <div class="meta" id="meta"><div class="row"><span>等待事件</span></div></div>
    <div class="controls">
      <div class="btns">
        <button id="prev" title="上一步">⏮</button>
        <button id="play" title="播放/暂停">▶</button>
        <button id="next" title="下一步">⏭</button>
      </div>
      <input id="scrub" type="range" min="0" max="0" value="0" />
    </div>
    <div class="status" id="status">等待 env.step 事件...</div>
    <div class="legend">
      图例:<br>
      <span style="background:var(--agent)"></span>@ 你(agent)
      <span style="background:var(--goal);margin-left:8px"></span>G 目标<br>
      <span style="background:var(--walld)"></span># 墙
      <span style="background:var(--floor);border:1px solid var(--line);margin-left:8px"></span>. 地<br>
      <span style="background:var(--fog)"></span>? 未探索/视野外
    </div>
  </div>
</div>
<script>
const CLS = { "@": "c-A", "G": "c-G", "#": "c-W", ".": "c-F", "?": "c-Q" };
const frames = [];   // {frame, action, reward, terminated, info}
let i = -1, timer = null, atLatest = true;
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), metaEl = $("meta"), gridEl = $("grid"), scrub = $("scrub");

function render() {
  if (i < 0 || i >= frames.length) { gridEl.innerHTML = ""; return; }
  const f = frames[i];
  const rows = f.frame.split("\\n");
  const cols = rows.length ? rows[0].length : 0;
  gridEl.style.gridTemplateColumns = `repeat(${cols}, 34px)`;
  let html = "";
  for (const row of rows) {
    for (const ch of row) {
      html += `<div class="cell ${CLS[ch] || "c-F"}">${ch === "@" || ch === "G" ? ch : ""}</div>`;
    }
  }
  gridEl.innerHTML = html;
  // explored = 非 ? 的格数
  const explored = f.frame.split("").filter(c => c !== "?" && c !== "\\n").length;
  const total = cols * rows.length;
  const terminated = f.terminated ? "是" : "否";
  const infoStr = f.info && Object.keys(f.info).length ? JSON.stringify(f.info) : "—";
  metaEl.innerHTML = `
    <div class="row"><span>步</span><b>${i + 1} / ${frames.length}</b></div>
    <div class="row"><span>动作</span><b>${escapeHtml(f.action || "")}</b></div>
    <div class="row"><span>reward</span><b>${f.reward}</b></div>
    <div class="row"><span>terminated</span><b>${terminated}</b></div>
    <div class="row"><span>info</span><b>${escapeHtml(infoStr)}</b></div>
    <div class="row"><span>网格</span><b>${cols}×${rows.length}</b></div>
    <div class="row"><span>已探索</span><b>${explored} / ${total}</b></div>`;
  scrub.value = i;
  statusEl.textContent = `帧 ${i + 1}/${frames.length}` + (atLatest ? " · 实时跟踪" : " · 已暂停跟踪");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

const seenSeq = new Set();   // 事件 sequence 去重(history/SSE 边界防双计)

function handleEvent(payload, seq) {
  if (seq !== undefined && seq !== null && seenSeq.has(seq)) return;  // 去重
  if (seq !== undefined && seq !== null) seenSeq.add(seq);
  frames.push(payload);
  if (atLatest) { i = frames.length - 1; render(); }
  scrub.max = frames.length - 1;
}

function resetFrames() {
  frames.length = 0; seenSeq.clear(); i = -1; atLatest = true; scrub.max = 0; render();
}

scrub.addEventListener("input", () => {
  atLatest = Number(scrub.value) === frames.length - 1;
  i = Number(scrub.value);
  render();
});
$("prev").onclick = () => { if (i > 0) { i--; atLatest = (i === frames.length - 1); render(); } };
$("next").onclick = () => { if (i < frames.length - 1) { i++; atLatest = (i === frames.length - 1); render(); } };
const playBtn = $("play");
playBtn.onclick = () => {
  if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "▶"; return; }
  if (frames.length < 2) return;
  playBtn.textContent = "⏸";
  timer = setInterval(() => {
    i = (i + 1) % frames.length;
    atLatest = (i === frames.length - 1);
    render();
  }, 400);
};

// 历史 + SSE —— 重连时清空旧 run 帧(防跨场次混)+ sequence 去重
function loadHistory() {
  return fetch("/api/events?limit=10000", { cache: "no-store" }).then(r => r.json()).then(data => {
    for (const ev of (data.events || [])) {
      if (ev.type === "env.step" && ev.payload && ev.payload.frame) handleEvent(ev.payload, ev.sequence);
    }
    if (frames.length === 0) statusEl.textContent = "尚无 env.step 事件(等 agent 开始 act)...";
    else if (atLatest) { i = frames.length - 1; render(); }
  }).catch(() => { /* ignore */ });
}

let connectedBefore = false;
function connectSSE() {
  if (!window.EventSource) { statusEl.textContent = "浏览器不支持 SSE"; return; }
  const src = new EventSource("/api/events/stream?after=0");
  src.onopen = () => {
    if (connectedBefore) { resetFrames(); }   // 重连(新 run/新进程)→ 清旧帧,只显当前 server 的 run
    connectedBefore = true;
    loadHistory();
  };
  src.onmessage = (m) => {
    try {
      const ev = JSON.parse(m.data);
      if (ev.type === "env.step" && ev.payload && ev.payload.frame) handleEvent(ev.payload, ev.sequence);
    } catch (e) { /* ignore */ }
  };
  src.onerror = () => { statusEl.textContent = "事件流断开/已结束(server 可能已停,已缓存帧仍可拖动回看)"; };
}

connectSSE();   // 首连 onopen 触发首次 loadHistory
</script>
</body>
</html>
"""
