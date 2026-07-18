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

from brainregion.runtime import canonical_model_name, emit_event, list_events, wait_events

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


def _nonnegative_int_param(params: dict[str, list[str]], name: str, default: int = 0) -> int:
    raw = _first(params, name)
    if raw is None:
        return max(0, default)
    try:
        return max(0, int(raw))
    except ValueError:
        return max(0, default)


def _emit_snapshot_events(data: dict[str, Any]) -> None:
    debug = data.get("debug") or {}
    query = debug.get("query") or {}
    activation = data.get("activation") or {}
    call_status = activation.get("call_status") or {}
    regions = data.get("regions") or []
    emit_event(
        "dashboard.snapshot_built",
        payload={
            "has_query": data.get("has_query"),
            "query": query,
            "region_count": len(regions),
            "woken_count": call_status.get("woken_count", 0),
            "suggested_actions_count": call_status.get("suggested_actions_count", 0),
        },
    )
    if call_status:
        emit_event("dashboard.call_status", payload=call_status)
    for region in regions:
        phase = region.get("phase") or region.get("woke") or "unknown"
        if phase == "quiet" and not region.get("score") and not region.get("confidence"):
            continue
        emit_event(
            "region.activation",
            region_id=region.get("region"),
            payload={
                "phase": phase,
                "score": region.get("score", 0),
                "confidence": region.get("confidence", 0.0),
                "suggested_actions": region.get("suggested_actions", 0),
                "action_tools": region.get("action_tools", []),
            },
        )


_MODEL_EVENT_TYPES = {"model.call_started", "model.call_finished", "model.call_failed"}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _model_identity(event: dict[str, Any]) -> dict[str, str | None]:
    payload = _event_payload(event)
    model = payload.get("model") or event.get("model")
    resolved_model = payload.get("resolved_model") or model
    canonical = payload.get("canonical_model") or canonical_model_name(resolved_model or model)
    provider = payload.get("provider") or event.get("provider")
    endpoint_id = payload.get("endpoint_id")
    route = endpoint_id or provider or "official"
    label = f"{route}/{canonical}" if route != "official" and canonical else canonical or model or "unknown"
    return {
        "key": f"{route}:{canonical or model or 'unknown'}",
        "label": label,
        "model": model,
        "resolved_model": resolved_model,
        "canonical_model": canonical,
        "provider": provider,
        "endpoint_id": endpoint_id,
        "route": route,
    }


def _empty_model_stats(identity: dict[str, str | None]) -> dict[str, Any]:
    return {
        **identity,
        "started_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "known_cost_calls": 0,
        "missing_cost_calls": 0,
        "latency_ms_total": 0.0,
        "latency_samples": 0,
        "avg_latency_ms": None,
        "last_seen": None,
        "last_sequence": 0,
        "last_status": None,
        "last_error": None,
        "cost_sources": set(),
    }


def _add_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    target["input_tokens"] += _safe_int(usage.get("input_tokens"))
    target["output_tokens"] += _safe_int(usage.get("output_tokens"))
    target["total_tokens"] += _safe_int(usage.get("total_tokens"))
    target["cached_tokens"] += _safe_int(usage.get("cached_tokens"))
    target["reasoning_tokens"] += _safe_int(usage.get("reasoning_tokens"))


def summarize_model_events(
    events: list[dict[str, Any]],
    *,
    recent_limit: int = 20,
) -> dict[str, Any]:
    """Aggregate model call telemetry for the debug dashboard."""
    model_events = [event for event in events if event.get("type") in _MODEL_EVENT_TYPES]
    totals: dict[str, Any] = {
        "event_count": len(model_events),
        "started_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "in_flight_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "known_cost_calls": 0,
        "missing_cost_calls": 0,
        "latency_ms_total": 0.0,
        "latency_samples": 0,
        "avg_latency_ms": None,
    }
    by_model: dict[str, dict[str, Any]] = {}

    for event in sorted(model_events, key=lambda item: int(item.get("sequence", 0) or 0)):
        event_type = str(event.get("type") or "")
        payload = _event_payload(event)
        identity = _model_identity(event)
        stats = by_model.setdefault(identity["key"] or "unknown", _empty_model_stats(identity))
        stats["last_seen"] = event.get("timestamp")
        stats["last_sequence"] = int(event.get("sequence", 0) or 0)
        status = payload.get("status") or ("started" if event_type == "model.call_started" else "unknown")
        stats["last_status"] = status

        if event_type == "model.call_started":
            stats["started_calls"] += 1
            totals["started_calls"] += 1
            continue
        if event_type == "model.call_failed":
            stats["failed_calls"] += 1
            totals["failed_calls"] += 1
            stats["last_error"] = payload.get("error")
        elif event_type == "model.call_finished":
            stats["successful_calls"] += 1
            totals["successful_calls"] += 1

        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        _add_usage(stats, usage)
        _add_usage(totals, usage)

        cost = _safe_float(payload.get("cost_usd"))
        if cost is None:
            stats["missing_cost_calls"] += 1
            totals["missing_cost_calls"] += 1
        else:
            stats["cost_usd"] += cost
            totals["cost_usd"] += cost
            stats["known_cost_calls"] += 1
            totals["known_cost_calls"] += 1
        if payload.get("cost_source"):
            stats["cost_sources"].add(str(payload["cost_source"]))

        latency = _safe_float(payload.get("latency_ms"))
        if latency is not None:
            stats["latency_ms_total"] += latency
            stats["latency_samples"] += 1
            totals["latency_ms_total"] += latency
            totals["latency_samples"] += 1

    totals["in_flight_calls"] = max(0, totals["started_calls"] - totals["successful_calls"] - totals["failed_calls"])
    if totals["latency_samples"]:
        totals["avg_latency_ms"] = round(totals["latency_ms_total"] / totals["latency_samples"], 3)

    models: list[dict[str, Any]] = []
    for stats in by_model.values():
        if stats["latency_samples"]:
            stats["avg_latency_ms"] = round(stats["latency_ms_total"] / stats["latency_samples"], 3)
        stats["cost_usd"] = round(stats["cost_usd"], 8)
        stats["cost_sources"] = sorted(stats["cost_sources"])
        stats.pop("latency_ms_total", None)
        models.append(stats)
    models.sort(key=lambda item: (item["last_sequence"], item["total_tokens"], item["cost_usd"]), reverse=True)

    recent: list[dict[str, Any]] = []
    for event in sorted(model_events, key=lambda item: int(item.get("sequence", 0) or 0), reverse=True)[: max(1, recent_limit)]:
        payload = _event_payload(event)
        identity = _model_identity(event)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        recent.append(
            {
                "sequence": event.get("sequence"),
                "timestamp": event.get("timestamp"),
                "type": event.get("type"),
                "status": payload.get("status") or ("started" if event.get("type") == "model.call_started" else "unknown"),
                "label": identity["label"],
                "model": identity["model"],
                "canonical_model": identity["canonical_model"],
                "provider": identity["provider"],
                "endpoint_id": identity["endpoint_id"],
                "usage": usage,
                "cost_usd": payload.get("cost_usd"),
                "cost_source": payload.get("cost_source"),
                "latency_ms": payload.get("latency_ms"),
                "error": payload.get("error"),
            }
        )

    totals["cost_usd"] = round(totals["cost_usd"], 8)
    totals.pop("latency_ms_total", None)
    return {"ok": True, "totals": totals, "models": models, "recent": recent}


def build_model_calls_payload(params: dict[str, list[str]] | None = None) -> dict[str, Any]:
    params = params or {}
    after = _nonnegative_int_param(params, "after", 0)
    limit = min(_int_param(params, "limit", 5000), 5000)
    recent_limit = min(_int_param(params, "recent", 20), 200)
    events = list_events(after_sequence=after, limit=limit)
    data = summarize_model_events(events, recent_limit=recent_limit)
    data["debug"] = {
        "generated_at_ms": int(time.time() * 1000),
        "after_sequence": after,
        "event_limit": limit,
        "recent_limit": recent_limit,
    }
    return data


def summarize_context_pressure_events(
    events: list[dict[str, Any]],
    *,
    recent_limit: int = 20,
) -> dict[str, Any]:
    """Aggregate content-free context pressure observations by region and model."""
    pressure_events = [
        event for event in events if event.get("type") == "context.pressure_observed"
    ]
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    totals = {
        "sample_count": len(pressure_events),
        "high_pressure_samples": 0,
        "truncation_count": 0,
        "capacity_known_count": 0,
        "model_call_count": 0,
    }
    recent: list[dict[str, Any]] = []
    for event in sorted(
        pressure_events, key=lambda item: int(item.get("sequence", 0) or 0)
    ):
        payload = _event_payload(event)
        region = str(payload.get("region") or event.get("region_id") or "unknown")
        model = str(payload.get("model") or event.get("model") or "unknown")
        endpoint = str(payload.get("endpoint_id") or event.get("endpoint_id") or "")
        key = (region, model, endpoint)
        band = str(payload.get("pressure_band") or "normal")
        high_pressure = band in {"strained", "saturated"}
        totals["high_pressure_samples"] += int(high_pressure)
        totals["truncation_count"] += int(bool(payload.get("context_truncated")))
        model_called = bool(payload.get("model_called"))
        totals["capacity_known_count"] += int(
            model_called and bool(payload.get("model_capacity_known"))
        )
        totals["model_call_count"] += int(model_called)
        stats = grouped.setdefault(
            key,
            {
                "region": region,
                "model": model,
                "endpoint_id": endpoint or None,
                "sample_count": 0,
                "high_pressure_samples": 0,
                "truncation_count": 0,
                "peak_pressure_score": 0.0,
                "last_sequence": 0,
            },
        )
        stats["sample_count"] += 1
        stats["high_pressure_samples"] += int(high_pressure)
        stats["truncation_count"] += int(bool(payload.get("context_truncated")))
        pressure_score = _safe_float(payload.get("pressure_score")) or 0.0
        stats["peak_pressure_score"] = max(
            float(stats["peak_pressure_score"]), pressure_score
        )
        stats.update(
            {
                "latest_pressure_score": pressure_score,
                "latest_pressure_band": band,
                "context_fill_ratio": payload.get("context_fill_ratio"),
                "model_window_fill_ratio": payload.get(
                    "model_window_fill_ratio"
                ),
                "model_capacity_known": bool(
                    payload.get("model_capacity_known")
                ),
                "input_tokens": _safe_int(payload.get("input_tokens")),
                "high_pressure_streak": _safe_int(
                    payload.get("high_pressure_streak")
                ),
                "signals": list(payload.get("signals") or ()),
                "last_sequence": int(event.get("sequence", 0) or 0),
                "last_seen": event.get("timestamp"),
            }
        )
    rows = sorted(
        grouped.values(),
        key=lambda item: (
            float(item.get("latest_pressure_score") or 0.0),
            int(item.get("last_sequence") or 0),
        ),
        reverse=True,
    )
    for event in sorted(
        pressure_events,
        key=lambda item: int(item.get("sequence", 0) or 0),
        reverse=True,
    )[: max(1, recent_limit)]:
        payload = _event_payload(event)
        recent.append(
            {
                "sequence": event.get("sequence"),
                "timestamp": event.get("timestamp"),
                "task_id": event.get("task_id"),
                "assignment_id": event.get("assignment_id"),
                "region": payload.get("region") or event.get("region_id"),
                "model": payload.get("model") or event.get("model"),
                "endpoint_id": payload.get("endpoint_id")
                or event.get("endpoint_id"),
                "pressure_score": payload.get("pressure_score"),
                "pressure_band": payload.get("pressure_band"),
                "context_fill_ratio": payload.get("context_fill_ratio"),
                "model_window_fill_ratio": payload.get(
                    "model_window_fill_ratio"
                ),
                "input_tokens": payload.get("input_tokens"),
                "signals": list(payload.get("signals") or ()),
            }
        )
    totals["capacity_coverage_rate"] = (
        totals["capacity_known_count"] / totals["model_call_count"]
        if totals["model_call_count"]
        else None
    )
    return {
        "ok": True,
        "mode": "shadow",
        "totals": totals,
        "region_models": rows,
        "recent": recent,
        "score_interpretation": "risk_proxy_not_measured_model_fatigue",
        "contains_context_content": False,
    }


def build_context_pressure_payload(
    params: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    params = params or {}
    after = _nonnegative_int_param(params, "after", 0)
    limit = min(_int_param(params, "limit", 5000), 5000)
    recent_limit = min(_int_param(params, "recent", 20), 200)
    events = list_events(after_sequence=after, limit=limit)
    data = summarize_context_pressure_events(events, recent_limit=recent_limit)
    data["debug"] = {
        "generated_at_ms": int(time.time() * 1000),
        "after_sequence": after,
        "event_limit": limit,
        "recent_limit": recent_limit,
    }
    return data


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
    _emit_snapshot_events(data)
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
.timeline {{
  display: grid;
  gap: 7px;
  max-height: 280px;
  overflow: auto;
}}
.table-wrap {{
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  min-width: 760px;
  font-size: 12px;
}}
th, td {{
  padding: 7px 8px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 600;
  background: #f4f6f2;
}}
tr:last-child td {{ border-bottom: 0; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.model-name {{ font-weight: 700; color: var(--ink); }}
.error-text {{ color: var(--bad); }}
.event {{
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 7px 8px;
  background: #fbfcfa;
}}
.event .event-top {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  color: var(--muted);
  font-size: 12px;
}}
.event code {{
  color: var(--blue);
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
  <div class="status">
    <a href="/scene" style="color:var(--blue);text-decoration:none;font-weight:600;margin-right:14px">🎮 场景 /scene</a>
    <a href="/runs" style="color:var(--blue);text-decoration:none;font-weight:600;margin-right:14px">📜 过去场次 /runs</a>
    <span id="live-dot" class="dot"></span><span id="status">启动中</span>
  </div>
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
      <h2>模型调用面板</h2>
      <div id="model-summary" class="grid"></div>
      <div class="table-wrap" style="margin-top: 10px;">
        <table>
          <thead>
            <tr>
              <th>模型</th>
              <th class="num">成功</th>
              <th class="num">失败</th>
              <th class="num">输入</th>
              <th class="num">输出</th>
              <th class="num">总 Token</th>
              <th class="num">成本 USD</th>
              <th class="num">平均延迟</th>
              <th>价格来源</th>
              <th>最近状态</th>
            </tr>
          </thead>
          <tbody id="model-rows"></tbody>
        </table>
      </div>
      <div class="table-wrap" style="margin-top: 10px;">
        <table>
          <thead>
            <tr>
              <th>最近调用</th>
              <th>状态</th>
              <th class="num">Token</th>
              <th class="num">成本 USD</th>
              <th class="num">延迟</th>
              <th>错误</th>
            </tr>
          </thead>
          <tbody id="recent-model-calls"></tbody>
        </table>
      </div>
    </section>
    <section style="margin-top: 12px;">
      <h2>脑区状态</h2>
      <h3>上下文压力（只读影子指标）</h3>
      <div id="context-pressure-summary" class="grid"></div>
      <div class="table-wrap" style="margin: 10px 0 14px;">
        <table>
          <thead>
            <tr>
              <th>脑区 / 模型</th>
              <th>状态</th>
              <th class="num">压力</th>
              <th class="num">私有上下文</th>
              <th class="num">模型窗口</th>
              <th class="num">输入 Token</th>
              <th>信号</th>
            </tr>
          </thead>
          <tbody id="context-pressure-rows"></tbody>
        </table>
      </div>
      <div id="regions" class="regions"></div>
    </section>
    <section style="margin-top: 12px;">
      <h2>实时事件</h2>
      <div id="events" class="timeline"></div>
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
function fmtInt(value) {{
  return Number(value || 0).toLocaleString("zh-CN");
}}
function fmtCost(value) {{
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  if (num === 0) return "0";
  return Math.abs(num) < 0.0001 ? num.toExponential(2) : num.toFixed(6);
}}
function fmtLatency(value) {{
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num >= 1000 ? (num / 1000).toFixed(2) + "s" : Math.round(num) + "ms";
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
function renderModelCalls(data) {{
  const totals = data.totals || {{}};
  const items = [
    ["总调用", (totals.successful_calls || 0) + (totals.failed_calls || 0)],
    ["进行中", totals.in_flight_calls || 0],
    ["失败", totals.failed_calls || 0],
    ["总 Token", fmtInt(totals.total_tokens)],
    ["输入 Token", fmtInt(totals.input_tokens)],
    ["输出 Token", fmtInt(totals.output_tokens)],
    ["推理 Token", fmtInt(totals.reasoning_tokens)],
    ["总成本", "$" + fmtCost(totals.cost_usd)],
    ["平均延迟", fmtLatency(totals.avg_latency_ms)],
    ["缺价格", totals.missing_cost_calls || 0],
  ];
  $("model-summary").innerHTML = items.map(([label, value]) => `<div class="metric"><span class="small">${{esc(label)}}</span><b>${{esc(value)}}</b></div>`).join("");

  const models = data.models || [];
  $("model-rows").innerHTML = models.length ? models.map((m) => {{
    const sources = (m.cost_sources || []).join(", ") || "-";
    const lastStatus = m.last_error ? `${{m.last_status || "-"}} · ${{m.last_error}}` : (m.last_status || "-");
    return `<tr>
      <td><div class="model-name">${{esc(m.label || m.canonical_model || m.model || "unknown")}}</div><div class="small">${{esc(m.resolved_model || "")}}</div></td>
      <td class="num">${{fmtInt(m.successful_calls)}}</td>
      <td class="num">${{fmtInt(m.failed_calls)}}</td>
      <td class="num">${{fmtInt(m.input_tokens)}}</td>
      <td class="num">${{fmtInt(m.output_tokens)}}</td>
      <td class="num">${{fmtInt(m.total_tokens)}}</td>
      <td class="num">${{fmtCost(m.cost_usd)}}</td>
      <td class="num">${{fmtLatency(m.avg_latency_ms)}}</td>
      <td>${{esc(sources)}}</td>
      <td>${{m.last_error ? `<span class="error-text">${{esc(lastStatus)}}</span>` : esc(lastStatus)}}</td>
    </tr>`;
  }}).join("") : `<tr><td colspan="10" class="small">还没有模型调用事件</td></tr>`;

  const recent = data.recent || [];
  $("recent-model-calls").innerHTML = recent.length ? recent.map((call) => {{
    const usage = call.usage || {{}};
    const tokenText = fmtInt(usage.total_tokens || 0);
    const error = call.error ? String(call.error).slice(0, 160) : "";
    return `<tr>
      <td><div class="model-name">${{esc(call.label || call.model || "unknown")}}</div><div class="small">#${{esc(call.sequence || "-")}} · ${{esc(call.timestamp || "")}}</div></td>
      <td>${{esc(call.status || call.type || "-")}}</td>
      <td class="num">${{tokenText}}</td>
      <td class="num">${{fmtCost(call.cost_usd)}}</td>
      <td class="num">${{fmtLatency(call.latency_ms)}}</td>
      <td>${{error ? `<span class="error-text">${{esc(error)}}</span>` : "-"}}</td>
    </tr>`;
  }}).join("") : `<tr><td colspan="6" class="small">还没有最近调用</td></tr>`;
}}
function renderContextPressure(data) {{
  const totals = data.totals || {{}};
  const coverage = totals.capacity_coverage_rate;
  const items = [
    ["观察样本", fmtInt(totals.sample_count || 0)],
    ["高压力", fmtInt(totals.high_pressure_samples || 0)],
    ["发生截断", fmtInt(totals.truncation_count || 0)],
    ["容量覆盖", coverage == null ? "未知" : `${{Math.round(Number(coverage) * 100)}}%`],
  ];
  $("context-pressure-summary").innerHTML = items.map(([label, value]) => `<div class="metric"><span class="small">${{esc(label)}}</span><b>${{esc(value)}}</b></div>`).join("");
  const ratioText = (value) => value == null ? "未知" : `${{Math.round(Number(value) * 100)}}%`;
  const rows = data.region_models || [];
  $("context-pressure-rows").innerHTML = rows.length ? rows.map((row) => {{
    const route = row.endpoint_id ? `${{row.endpoint_id}}/${{row.model}}` : row.model;
    const signals = (row.signals || []).join(", ") || "-";
    return `<tr>
      <td><div class="model-name">${{esc(row.region || "unknown")}}</div><div class="small">${{esc(route || "unknown")}}</div></td>
      <td>${{esc(row.latest_pressure_band || "normal")}}</td>
      <td class="num">${{esc(Number(row.latest_pressure_score || 0).toFixed(2))}}</td>
      <td class="num">${{esc(ratioText(row.context_fill_ratio))}}</td>
      <td class="num">${{esc(ratioText(row.model_window_fill_ratio))}}</td>
      <td class="num">${{fmtInt(row.input_tokens || 0)}}</td>
      <td>${{esc(signals)}}</td>
    </tr>`;
  }}).join("") : `<tr><td colspan="7" class="small">还没有脑区上下文压力样本</td></tr>`;
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
let lastEventSequence = 0;
const eventWindow = [];
let modelRefreshPending = false;
let contextPressureRefreshPending = false;
function pushEvent(event) {{
  const seq = Number(event.sequence || 0);
  if (seq && seq <= lastEventSequence) return;
  if (seq) lastEventSequence = seq;
  eventWindow.unshift(event);
  while (eventWindow.length > 100) eventWindow.pop();
  renderEvents();
  if (String(event.type || "").startsWith("model.call")) scheduleModelRefresh();
  if (String(event.type || "") === "context.pressure_observed") scheduleContextPressureRefresh();
}}
function renderEvents() {{
  const box = $("events");
  if (!box) return;
  if (!eventWindow.length) {{
    box.innerHTML = `<div class="small">等待运行事件...</div>`;
    return;
  }}
  box.innerHTML = eventWindow.map((event) => {{
    const payload = event.payload ? JSON.stringify(event.payload) : "";
    const ts = String(event.timestamp || "").replace("T", " ").replace("+00:00", "Z");
    return `<div class="event">
      <div class="event-top"><code>${{esc(event.type || "event")}}</code><span>#${{esc(event.sequence || "-")}} · ${{esc(ts)}}</span></div>
      <div class="small">${{event.region_id ? `region=${{esc(event.region_id)}}` : ""}}</div>
      ${{payload ? `<pre>${{esc(payload)}}</pre>` : ""}}
    </div>`;
  }}).join("");
}}
async function loadInitialEvents() {{
  try {{
    const res = await fetch("/api/events?limit=50", {{cache: "no-store"}});
    if (!res.ok) return;
    const data = await res.json();
    (data.events || []).forEach(pushEvent);
  }} catch (err) {{
    return;
  }}
}}
function scheduleModelRefresh() {{
  if (modelRefreshPending) return;
  modelRefreshPending = true;
  setTimeout(() => {{
    modelRefreshPending = false;
    loadModelCalls();
  }}, 150);
}}
async function loadModelCalls() {{
  if (paused) return;
  try {{
    const res = await fetch("/api/models?limit=5000&recent=20", {{cache: "no-store"}});
    if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
    renderModelCalls(await res.json());
  }} catch (err) {{
    $("model-summary").innerHTML = `<div class="metric"><span class="small">模型面板错误</span><b>${{esc(String(err))}}</b></div>`;
  }}
}}
function scheduleContextPressureRefresh() {{
  if (contextPressureRefreshPending) return;
  contextPressureRefreshPending = true;
  setTimeout(() => {{
    contextPressureRefreshPending = false;
    loadContextPressure();
  }}, 150);
}}
async function loadContextPressure() {{
  if (paused) return;
  try {{
    const res = await fetch("/api/context-pressure?limit=5000&recent=20", {{cache: "no-store"}});
    if (!res.ok) throw new Error(`HTTP ${{res.status}}`);
    renderContextPressure(await res.json());
  }} catch (err) {{
    $("context-pressure-summary").innerHTML = `<div class="metric"><span class="small">上下文压力面板错误</span><b>${{esc(String(err))}}</b></div>`;
  }}
}}
function connectEvents() {{
  if (!window.EventSource) {{
    pushEvent({{type: "dashboard.sse_unavailable", payload: {{"message": "EventSource unavailable"}}}});
    return;
  }}
  const source = new EventSource("/api/events/stream?after=" + encodeURIComponent(String(lastEventSequence)));
  source.onmessage = (message) => {{
    try {{
      pushEvent(JSON.parse(message.data));
    }} catch (err) {{
      pushEvent({{type: "dashboard.sse_parse_error", payload: {{"message": String(err)}}}});
    }}
  }};
  source.onerror = () => {{
    setStatus("事件流重连中 " + new Date().toLocaleTimeString(), "error");
  }};
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
loadInitialEvents().then(connectEvents);
timer = setInterval(() => {{
  loadSnapshot();
  loadModelCalls();
  loadContextPressure();
}}, refreshMs);
loadSnapshot();
loadModelCalls();
loadContextPressure();
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
        if parsed.path == "/scene":
            from .env_scene import build_env_scene_html  # 专用场景查看页(env.step 网格渲染)
            self._send_html(build_env_scene_html())
            return
        if parsed.path == "/runs":
            from .runs_index import build_runs_index_html, list_env_runs  # 过去场次归档索引
            self._send_html(build_runs_index_html(list_env_runs()))
            return
        if parsed.path.startswith("/replay/"):
            import re as _re
            from pathlib import Path as _P
            run_id = parsed.path[len("/replay/"):]
            if not _re.fullmatch(r"[A-Za-z0-9_-]+", run_id):  # 防 path traversal
                self.send_error(HTTPStatus.NOT_FOUND, "bad run_id")
                return
            replay_file = _P(".brain-region") / "sandbox" / f"{run_id}.html"
            if not replay_file.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "run not found")
                return
            self._send_html(replay_file.read_text(encoding="utf-8"))
            return
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/events":
            params = parse_qs(parsed.query, keep_blank_values=True)
            after = _nonnegative_int_param(params, "after", 0)
            limit = _int_param(params, "limit", 100)
            self._send_json({"ok": True, "events": list_events(after_sequence=after, limit=limit)})
            return
        if parsed.path == "/api/events/stream":
            self._send_sse(parse_qs(parsed.query, keep_blank_values=True))
            return
        if parsed.path == "/api/models":
            self._send_json(build_model_calls_payload(parse_qs(parsed.query, keep_blank_values=True)))
            return
        if parsed.path == "/api/context-pressure":
            self._send_json(
                build_context_pressure_payload(
                    parse_qs(parsed.query, keep_blank_values=True)
                )
            )
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

    def _send_sse(self, params: dict[str, list[str]]) -> None:
        after = _nonnegative_int_param(params, "after", 0)
        last_id = self.headers.get("Last-Event-ID")
        if last_id:
            try:
                after = int(last_id)
            except ValueError:
                pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        sequence = max(0, after)
        try:
            while True:
                events = wait_events(after_sequence=sequence, timeout=15.0, limit=100)
                if not events:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
                for event in events:
                    sequence = max(sequence, int(event.get("sequence", 0)))
                    data = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
                    self.wfile.write(f"id: {sequence}\n".encode("utf-8"))
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8")
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


def serve_debug_dashboard(
    options: DebugDashboardOptions, *, open_browser: bool = False, open_path: str = "/"
) -> None:
    server = _DebugServer((options.host, options.port), options)
    base = f"http://{options.host}:{server.server_port}"
    print(f"BrainRegion debug dashboard: {base}/  (场景查看: {base}/scene)")
    if open_browser:
        path = open_path if open_path.startswith("/") else "/" + open_path
        threading.Timer(0.2, lambda: webbrowser.open(f"{base}{path}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBrainRegion debug dashboard stopped.")
    finally:
        server.server_close()
