"""过去场次索引页(/runs):扫描 .brain-region/sandbox/*.html(每 run 落盘的 replay),列出场次表。

每场次的 meta 嵌在 replay HTML 的 ``<script type="application/json" id="replay-data">`` 里(render_replay_html
写入),本模块抽取其 meta(model/size/solved/n_steps/fog/goal)渲染成表,点开 → /replay/<run_id>(debug_server
路由 serve 该 HTML 文件;server 死了也可 file:// 直开)。

仅 debug 观测用;扫盘解析在请求时做(场次少,可接受)。XSS:meta 值全 escape。
"""
from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

_REPLAY_DATA_RE = re.compile(r'id="replay-data">(.*?)</script>', re.DOTALL)


def list_env_runs(out_dir: str | Path | None = None) -> list[dict]:
    """扫 out_dir(默认 .brain-region/sandbox/)下 env-*.html,抽 meta,按 mtime 倒序返列表。"""
    out = Path(out_dir) if out_dir else Path(".brain-region") / "sandbox"
    if not out.is_dir():
        return []
    runs: list[dict] = []
    for html_path in out.glob("env-*.html"):
        meta: dict = {}
        try:
            text = html_path.read_text(encoding="utf-8")
            m = _REPLAY_DATA_RE.search(text)
            if m:
                meta = json.loads(m.group(1)).get("meta", {}) or {}
        except Exception:  # noqa: BLE001 — 损坏文件跳过,不崩索引页
            meta = {}
        runs.append({
            "run_id": html_path.stem,
            "meta": meta,
            "mtime": html_path.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def build_runs_index_html(runs: list[dict]) -> str:
    rows_html = ""
    if not runs:
        rows_html = '<tr><td colspan="8" class="small">还没有落盘的场次(跑完一个 sandbox env run 后这里会出现)。</td></tr>'
    for r in runs:
        meta = r.get("meta") or {}
        from datetime import datetime
        ts = datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M:%S")
        size = meta.get("size", "?")
        radius = meta.get("visibility_radius")
        fog = f"r={radius}" if radius is not None else "—"
        solved = meta.get("solved")
        solved_cell = "✅ 解出" if solved else ("❌ 未解" if solved is False else "?")
        reward = meta.get("total_reward", "?")
        goal = meta.get("goal_pos")
        goal_cell = f"({goal[0]},{goal[1]})" if isinstance(goal, (list, tuple)) and len(goal) == 2 else "—"
        rows_html += f"""<tr>
  <td>{escape(ts)}</td>
  <td><a href="/replay/{escape(r['run_id'])}" class="run-id">{escape(r['run_id'])}</a></td>
  <td>{escape(str(meta.get('model', '?')))}</td>
  <td class="num">{size}×{size}</td>
  <td>{escape(fog)}</td>
  <td class="num">{goal_cell}</td>
  <td>{solved_cell}</td>
  <td class="num">{meta.get('n_steps', '?')}</td>
  <td class="num">{reward}</td>
  <td>{escape(str(meta.get('termination', '?')))}</td>
</tr>"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>过去场次 · GridWorld</title>
<style>
:root {{ --bg:#f5f6f2; --panel:#fff; --ink:#17201a; --muted:#667065; --line:#dfe4dc; --blue:#34699a; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }}
header {{ display:flex; align-items:center; justify-content:space-between; padding:12px 20px;
  border-bottom:1px solid var(--line); background:#eef2ea; }}
h1 {{ margin:0; font-size:16px; font-weight:700; }}
.links a {{ color:var(--blue); text-decoration:none; font-weight:600; margin-left:14px; }}
main {{ padding:14px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.note {{ color:var(--muted); font-size:12px; margin-bottom:10px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; background:#f4f6f2; }}
tr:last-child td {{ border-bottom:0; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.run-id {{ color:var(--blue); font-family:ui-monospace,monospace; font-size:12px; }}
.run-id:hover {{ text-decoration:underline; }}
.small {{ color:var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>过去场次(GridWorld replay 归档)</h1>
  <div class="links"><a href="/scene">实时场景</a><a href="/">调试面板</a></div>
</header>
<main>
  <div class="panel">
    <div class="note">每跑完一个 <code>brain-region sandbox env</code> 会在这里归档一场(点 run_id 看回放;server 死了也可直接打开 .brain-region/sandbox/&lt;run_id&gt;.html)。</div>
    <table>
      <thead><tr>
        <th>时间</th><th>run_id</th><th>模型</th><th>网格</th><th>fog</th><th>goal</th>
        <th>结果</th><th>步数</th><th>reward</th><th>终止</th>
      </tr></thead>
      <tbody>{rows_html}
      </tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
