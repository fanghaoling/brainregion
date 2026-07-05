"""Local debug dashboard for observing BrainRegion activation snapshots.

The first version is intentionally small: a stdlib HTTP server renders a
self-contained page and serves fresh ``BrainSnapshot`` JSON on every poll.
"""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .snapshot import build_snapshot


@dataclass(frozen=True)
class DebugDashboardOptions:
    host: str = "127.0.0.1"
    port: int = 8765
    goal: str = ""
    problem: str = ""
    context: str = ""
    gold_regions: tuple[str, ...] = ()
    run_id: str | None = None
    region: str | None = None
    judge_id: str | None = None
    history_limit: int = 20
    memory_preview_k: int = 5
    top_k: int = 5
    refresh_ms: int = 2000


def parse_gold_regions(value: str | None) -> tuple[str, ...]:
    return tuple(g.strip() for g in (value or "").split(",") if g.strip())


def _first(params: dict[str, list[str]], name: str, default: str | None = None) -> str | None:
    values = params.get(name)
    if not values:
        return default
    return values[-1]


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    raw = _first(params, name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def build_snapshot_payload(options: DebugDashboardOptions, params: dict[str, list[str]] | None = None) -> dict[str, Any]:
    params = params or {}
    gold_raw = _first(params, "gold_regions", ",".join(options.gold_regions))
    snap = build_snapshot(
        goal=_first(params, "goal", options.goal) or "",
        problem=_first(params, "problem", options.problem) or "",
        context=_first(params, "context", options.context) or "",
        gold_regions=list(parse_gold_regions(gold_raw)) or None,
        run_id=_first(params, "run", options.run_id) or None,
        region=_first(params, "region", options.region) or None,
        judge_id=_first(params, "judge", options.judge_id) or None,
        history_limit=_int_param(params, "history_limit", options.history_limit),
        memory_preview_k=_int_param(params, "memory_preview_k", options.memory_preview_k),
        top_k=_int_param(params, "top_k", options.top_k),
    )
    data = snap.to_dict()
    data["debug"] = {
        "generated_at_ms": int(time.time() * 1000),
        "refresh_ms": options.refresh_ms,
        "query": {
            "goal": _first(params, "goal", options.goal) or "",
            "problem": _first(params, "problem", options.problem) or "",
            "context": _first(params, "context", options.context) or "",
            "gold_regions": list(parse_gold_regions(gold_raw)),
            "top_k": _int_param(params, "top_k", options.top_k),
        },
    }
    return data


def build_debug_dashboard_html(options: DebugDashboardOptions) -> str:
    problem = escape(options.problem, quote=True)
    goal = escape(options.goal, quote=True)
    context = escape(options.context, quote=True)
    gold = escape(",".join(options.gold_regions), quote=True)
    refresh = max(500, options.refresh_ms)
    top_k = options.top_k
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainRegion 调试面板</title>
<style>
:root {{
  --bg: #f7f8f5;
  --panel: #ffffff;
  --ink: #17201a;
  --muted: #667065;
  --line: #dfe4dc;
  --ok: #2c7a4b;
  --warn: #b16d13;
  --bad: #b7423c;
  --blue: #34699a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: #eef2ea;
}}
h1 {{
  margin: 0;
  font-size: 18px;
  font-weight: 700;
}}
main {{
  display: grid;
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
  gap: 14px;
  padding: 14px;
}}
section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}}
h2 {{
  margin: 0 0 10px;
  font-size: 13px;
  color: var(--muted);
  text-transform: uppercase;
}}
label {{
  display: grid;
  gap: 5px;
  margin-bottom: 9px;
  color: var(--muted);
  font-size: 12px;
}}
input, textarea {{
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  color: var(--ink);
  background: #fff;
  font: inherit;
}}
textarea {{ min-height: 90px; resize: vertical; }}
button {{
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  padding: 7px 10px;
  cursor: pointer;
  font: inherit;
}}
button.primary {{
  border-color: var(--blue);
  background: var(--blue);
  color: #fff;
}}
.status {{
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 12px;
}}
.dot {{
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--warn);
}}
.dot.live {{ background: var(--ok); }}
.dot.error {{ background: var(--bad); }}
.grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 10px;
}}
.metric {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 9px;
}}
.metric b {{ display: block; font-size: 18px; margin-top: 4px; }}
.regions {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 10px;
}}
.region {{
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}}
.region .top {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
  margin-bottom: 7px;
}}
.phase {{
  border-radius: 999px;
  padding: 2px 7px;
  font-size: 12px;
  background: #eef2ea;
  color: var(--muted);
}}
.phase.woken, .phase.escalated, .phase.shadow_promoted {{ color: var(--ok); background: #e5f3ea; }}
.phase.missed, .phase.false_wake {{ color: var(--bad); background: #f8e8e6; }}
.bar {{
  height: 8px;
  border-radius: 999px;
  background: #edf0ea;
  overflow: hidden;
  margin: 8px 0;
}}
.bar span {{
  display: block;
  height: 100%;
  min-width: 2px;
  background: var(--blue);
}}
.bar.woken span, .bar.escalated span, .bar.shadow_promoted span {{ background: var(--ok); }}
.bar.missed span, .bar.false_wake span {{ background: var(--bad); }}
.small {{ color: var(--muted); font-size: 12px; }}
.tools {{ margin-top: 8px; color: var(--muted); font-size: 12px; word-break: break-word; }}
.row {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
.split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.log {{
  min-height: 180px;
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
  font-size: 12px;
}}
pre {{
  overflow: auto;
  margin: 0;
  max-height: 220px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #fbfcfa;
}}
@media (max-width: 900px) {{
  main {{ grid-template-columns: 1fr; }}
  .grid, .split {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body data-refresh-ms="{refresh}">
<header>
  <h1>BrainRegion 调试面板</h1>
  <div class="status"><span id="live-dot" class="dot"></span><span id="status">启动中</span></div>
</header>
<main>
  <aside>
    <section>
      <h2>查询输入</h2>
      <label>问题<input id="problem" value="{problem}"></label>
      <label>目标<input id="goal" value="{goal}"></label>
      <label>期望脑区<input id="gold" value="{gold}"></label>
      <label>召回数量 Top K<input id="top-k" type="number" min="1" max="32" value="{top_k}"></label>
      <label>上下文<textarea id="context">{context}</textarea></label>
      <div class="row">
        <button id="refresh" class="primary">刷新</button>
        <button id="pause">暂停</button>
      </div>
    </section>
    <section style="margin-top: 12px;">
      <h2>问题记录</h2>
      <div class="row" style="margin-bottom: 8px;">
        <button data-note="漏唤醒">漏唤醒</button>
        <button data-note="误唤醒">误唤醒</button>
        <button data-note="记忆未召回">记忆未召回</button>
        <button data-note="工具建议错误">工具建议错误</button>
      </div>
      <textarea id="notes" class="log"></textarea>
    </section>
  </aside>
  <div>
    <section>
      <h2>调用状态</h2>
      <div id="metrics" class="grid"></div>
    </section>
    <section style="margin-top: 12px;">
      <h2>脑区状态</h2>
      <div id="regions" class="regions"></div>
    </section>
    <section style="margin-top: 12px;">
      <h2>原始快照</h2>
      <pre id="raw">{{}}</pre>
    </section>
  </div>
</main>
<script>
const refreshMs = Number(document.body.dataset.refreshMs || 2000);
let paused = false;
let timer = null;
const $ = (id) => document.getElementById(id);
function esc(value) {{
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }})[ch]);
}}
const notes = $("notes");
notes.value = localStorage.getItem("brainregion.debug.notes") || "";
notes.addEventListener("input", () => localStorage.setItem("brainregion.debug.notes", notes.value));
document.querySelectorAll("[data-note]").forEach((button) => {{
  button.addEventListener("click", () => {{
    const stamp = new Date().toISOString();
    notes.value += `\\n[${{stamp}}] ${{button.dataset.note}}: `;
    notes.focus();
    localStorage.setItem("brainregion.debug.notes", notes.value);
  }});
}});
function params() {{
  const p = new URLSearchParams();
  p.set("problem", $("problem").value);
  p.set("goal", $("goal").value);
  p.set("context", $("context").value);
  p.set("gold_regions", $("gold").value);
  p.set("top_k", $("top-k").value);
  return p;
}}
function pct(region) {{
  const confidence = Number(region.confidence || 0);
  const score = Number(region.score || 0);
  return Math.max(0, Math.min(100, Math.round(Math.max(confidence, Math.min(score / 10, 1)) * 100)));
}}
function setStatus(text, mode) {{
  $("status").textContent = text;
  $("live-dot").className = "dot " + (mode || "");
}}
function renderMetrics(snapshot) {{
  const call = (snapshot.activation && snapshot.activation.call_status) || {{}};
  const items = [
    ["模型调用", call.models_called ? "是" : "否"],
    ["已唤醒", call.woken_count || 0],
    ["已升级", call.escalated_count || 0],
    ["建议动作", call.suggested_actions_count || 0],
    ["需批准", call.requires_user_approval_count || 0],
    ["评估状态", call.metrics_status || "unknown"],
    ["建议工具", (call.action_tools || []).join(", ") || "-"],
    ["Schema", snapshot.schema_version || "-"],
  ];
  $("metrics").innerHTML = items.map(([label, value]) => `<div class="metric"><span class="small">${{esc(label)}}</span><b>${{esc(value)}}</b></div>`).join("");
}}
function renderRegions(snapshot) {{
  const regions = snapshot.regions || [];
  $("regions").innerHTML = regions.map((r) => {{
    const phase = r.phase || r.woke || "unknown";
    const phaseClass = String(phase).replace(/[^a-zA-Z0-9_-]/g, "_");
    const width = pct(r);
    const tools = (r.action_tools || []).join(", ");
    return `<div class="region">
      <div class="top"><strong>${{esc(r.region)}}</strong><span class="phase ${{phaseClass}}">${{esc(phase)}}</span></div>
      <div class="bar ${{phaseClass}}"><span style="width:${{width}}%"></span></div>
      <div class="small">激活强度 ${{width}}% · 分数 ${{esc(r.score || 0)}} · 置信度 ${{esc(Number(r.confidence || 0).toFixed(2))}}</div>
      <div class="small">记忆 ${{esc(r.recallable || 0)}}/${{esc(r.total || 0)}} · 动作 ${{esc(r.suggested_actions || 0)}}</div>
      ${{tools ? `<div class="tools">${{esc(tools)}}</div>` : ""}}
    </div>`;
  }}).join("");
}}
async function loadSnapshot() {{
  if (paused) return;
  try {{
    const res = await fetch("/api/snapshot?" + params().toString(), {{cache: "no-store"}});
    if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
    const snapshot = await res.json();
    renderMetrics(snapshot);
    renderRegions(snapshot);
    $("raw").textContent = JSON.stringify(snapshot.activation || snapshot.debug || snapshot, null, 2);
    setStatus("实时刷新 " + new Date().toLocaleTimeString(), "live");
  }} catch (err) {{
    setStatus(String(err), "error");
  }}
}}
$("refresh").addEventListener("click", loadSnapshot);
$("pause").addEventListener("click", () => {{
  paused = !paused;
  $("pause").textContent = paused ? "继续" : "暂停";
  if (!paused) loadSnapshot();
}});
timer = setInterval(loadSnapshot, refreshMs);
loadSnapshot();
</script>
</body>
</html>"""


class _DebugHandler(BaseHTTPRequestHandler):
    server: "_DebugServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self._send_html(build_debug_dashboard_html(self.server.options))
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/snapshot":
            try:
                payload = build_snapshot_payload(self.server.options, parse_qs(parsed.query, keep_blank_values=True))
            except Exception as exc:  # noqa: BLE001 - debug endpoint should surface failures as JSON.
                self._send_json({"ok": False, "error": type(exc).__name__, "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def _send_html(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DebugServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], options: DebugDashboardOptions):
        super().__init__(server_address, _DebugHandler)
        self.options = options


def serve_debug_dashboard(options: DebugDashboardOptions, *, open_browser: bool = False) -> None:
    server = _DebugServer((options.host, options.port), options)
    url = f"http://{options.host}:{server.server_port}/"
    print(f"BrainRegion debug dashboard: {url}")
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBrainRegion debug dashboard stopped.")
    finally:
        server.server_close()
