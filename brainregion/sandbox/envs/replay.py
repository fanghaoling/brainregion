"""GridWorld 回放页(Phase A):env.frames → 自包含 HTML(play/step),复用调试窗观感。

review 双强(2026-07-08)XSS 硬化(gpt #19):frames/meta 一律经 json.dumps 嵌入非执行
``<script type="application/json">`` 且 ``<`` → ``\\u003c``(防 ``</script>`` 破出);DOM 渲染用
textContent(非 innerHTML)。文件写显式 ``encoding="utf-8"``(opus #6,不靠 PYTHONIOENCODING)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_replay_html(frames: list[str], meta: dict[str, Any] | None = None) -> str:
    """frames(每帧网格文本)+ meta → 自包含回放 HTML(play/上一步/下一步)。"""
    data = {"frames": list(frames), "meta": dict(meta or {})}
    # 防 </script> 破出:JSON 里所有 < 转义(放进非执行 script 标签仍需)。JSON.parse 还原。
    data_json = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GridWorld 回放</title>
<style>
body { font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; background: #f7f8f5; color: #17201a; margin: 0; padding: 14px; }
h3 { margin: 0 0 10px; font-size: 16px; }
pre { background: #fff; border: 1px solid #dfe4dc; border-radius: 6px; padding: 10px;
      font: 14px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre; }
.controls { display: flex; gap: 8px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
button { padding: 6px 10px; border: 1px solid #dfe4dc; border-radius: 6px; background: #fff; cursor: pointer; font: inherit; }
button:hover { background: #eef2ea; }
.info { color: #667065; font-size: 12px; }
.meta { color: #667065; font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<h3>GridWorld 回放</h3>
<div class="controls">
  <button id="prev">⏮ 上一步</button>
  <button id="play">▶ 播放</button>
  <button id="next">下一步 ⏭</button>
  <span class="info" id="info"></span>
</div>
<pre id="frame"></pre>
<div class="meta" id="meta"></div>
<script type="application/json" id="replay-data">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('replay-data').textContent);
  const frames = D.frames || [];
  const meta = D.meta || {};
  let i = 0; let timer = null;
  const frameEl = document.getElementById('frame');
  const infoEl = document.getElementById('info');
  const metaEl = document.getElementById('meta');
  function render() {
    frameEl.textContent = frames[i] != null ? frames[i] : '(无帧)';
    infoEl.textContent = '帧 ' + (i + 1) + ' / ' + frames.length;
    metaEl.textContent = Object.entries(meta).map(([k, v]) => k + ': ' + v).join('  |  ');
  }
  document.getElementById('prev').onclick = () => { if (i > 0) { i--; render(); } };
  document.getElementById('next').onclick = () => { if (i < frames.length - 1) { i++; render(); } };
  const playBtn = document.getElementById('play');
  playBtn.onclick = () => {
    if (timer) { clearInterval(timer); timer = null; playBtn.textContent = '▶ 播放'; return; }
    if (frames.length < 2) return;
    playBtn.textContent = '⏸ 暂停';
    timer = setInterval(() => { i = (i + 1) % frames.length; render(); }, 500);
  };
  render();
})();
</script>
</body>
</html>
""".replace("__DATA__", data_json)


def write_replay_html(
    path: str | Path, frames: list[str], meta: dict[str, Any] | None = None,
) -> Path:
    """写回放 HTML 到 path(显式 utf-8,不靠 PYTHONIOENCODING;opus #6)。"""
    out = Path(path)
    out.write_text(render_replay_html(frames, meta), encoding="utf-8")
    return out
