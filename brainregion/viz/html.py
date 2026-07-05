"""HtmlRenderer:BrainSnapshot → 自包含静态 HTML dashboard(可视化 Phase 1 唯一 renderer)。

- **自包含**:内联 <style>,无外部请求、无 <script src>、零 JS(纯静态最安全)。
- **XSS 安全**:所有插值经 html.escape()(stdlib)——memory summary / region 名 / explain / reasons
  全是用户或内部内容,等同 core/context.py 的 data-fencing 思路。
- region-centric:hero 是 region snapshots 网格;默认(无查询)无 Activation 段。
- **界面文案统一中文**(方便调试);数据值(gate 决策 GO/NO_GO、run_id、status 枚举)保留原样。
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

from ..inspector.render import status_symbol
from .diff import (
    ADDED, BrainDiff, MULTI, REGION_CHANGED, REMOVED, STATUS_CHANGED,
    SUMMARY_CHANGED, TRIGGER_CHANGED,
)
from .snapshot import BrainSnapshot

_DOCTYPE = "<!DOCTYPE html>"

# ── 中文标签映射(数据枚举原样存,显示时本地化;CSS class 仍用原始枚举值)──────────
_GOV_STATUS_LABELS = {"active": "活跃", "pending": "待核实",
                      "superseded": "已覆盖", "wrong": "错误"}
_STAGE_LABELS = {"wake": "唤醒", "retrieve": "检索", "memory": "记忆",
                 "consult": "会诊", "judge": "评审"}
_STAGE_STATUS_LABELS = {"SUCCESS": "成功", "FAILED": "失败", "SKIPPED": "跳过",
                        "UNKNOWN": "未知", "NOT_INSTRUMENTED": "未埋点"}
_PHASE_LABELS = {
    "woken": "已唤醒",
    "escalated": "已升级",
    "shadow_promoted": "shadow 唤醒",
    "shadow": "接近阈值",
    "retrieved": "已检索",
    "missed": "漏唤醒",
    "false_wake": "误唤醒",
    "quiet": "静默",
    "unknown": "未知",
}


def _esc(v) -> str:
    """HTML 转义任意值(防 XSS)。None → 空串。"""
    return html.escape("" if v is None else str(v), quote=True)


def _gov_label(status: str) -> str:
    return _GOV_STATUS_LABELS.get(status or "", status or "")


def _stage_label(name: str) -> str:
    return _STAGE_LABELS.get(name or "", name or "")


def _stage_status_label(name: str) -> str:
    return _STAGE_STATUS_LABELS.get(name or "", name or "")


def _phase_label(name: str) -> str:
    return _PHASE_LABELS.get(name or "", name or "")


def _fmt_ts(iso: str) -> str:
    """ISO 时间戳 → 可读(失败原样返回)。"""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:  # noqa: BLE001
        return iso


_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #f5f6f8; color: #1f2329; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 24px 20px 60px; }
header { margin-bottom: 20px; }
header h1 { margin: 0 0 4px; font-size: 22px; }
header .meta { color: #8a9099; font-size: 13px; }
section { background: #fff; border: 1px solid #e8eaed; border-radius: 10px;
          padding: 16px 18px; margin-bottom: 16px; }
section h2 { margin: 0 0 12px; font-size: 15px; color: #4a5159;
             text-transform: uppercase; letter-spacing: .04em; }
/* KPI 行 */
.kpis { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.kpi { flex: 1 1 200px; background: #fff; border: 1px solid #e8eaed; border-left: 4px solid #c9ced6;
       border-radius: 10px; padding: 14px 16px; }
.kpi .label { font-size: 12px; color: #8a9099; text-transform: uppercase; letter-spacing: .04em; }
.kpi .value { font-size: 22px; font-weight: 600; margin: 4px 0 2px; }
.kpi .hint { font-size: 12px; color: #8a9099; }
.kpi.ok { border-left-color: #2ea44f; }
.kpi.warn { border-left-color: #d9a300; }
.kpi.bad { border-left-color: #d73a49; }
/* region 网格(hero)*/
.regions { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
.region { border: 1px solid #e8eaed; border-radius: 8px; padding: 12px; background: #fafbfc; }
.region .name { font-weight: 600; font-size: 14px; margin-bottom: 6px; word-break: break-all; }
.region .nums { font-size: 12px; color: #5a626c; }
.region .nums b { color: #1f2329; }
.region .act { margin-top: 8px; font-size: 12px; color: #5a626c; }
.bar { height: 7px; background: #e8eaed; border-radius: 999px; overflow: hidden; margin: 6px 0 4px; }
.bar span { display: block; height: 100%; background: #0969da; border-radius: 999px; }
.bar.missed span { background: #cf222e; }
.bar.false_wake span { background: #b08800; }
.bar.quiet span { background: #c9ced6; }
.badge { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 10px;
         font-weight: 600; margin-left: 6px; vertical-align: middle; }
.badge.woke { background: #e6f4ea; color: #1a7f37; }
.badge.quiet { background: #f0f1f3; color: #8a9099; }
.badge.missed { background: #ffebe9; color: #cf222e; }
.badge.false_wake { background: #fff8c5; color: #7d5e00; }
/* 通用表 */
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eef0f2; }
th { color: #8a9099; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
td.mono, th.mono { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-size: 12px; }
/* 状态色 */
.dec { font-weight: 600; }
.dec-go, .dec-OK { color: #1a7f37; }
.dec-no_go, .dec-FAIL, .dec-fail { color: #cf222e; }
.dec-inconclusive { color: #b08800; }
.dec-neutral { color: #8a9099; }
.chip { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 8px; margin-right: 4px; }
.chip.active { background: #e6f4ea; color: #1a7f37; }
.chip.pending { background: #fff8c5; color: #7d5e00; }
.chip.superseded, .chip.wrong { background: #ffebe9; color: #cf222e; }
.muted { color: #8a9099; }
.timeline td.sym { text-align: center; font-size: 14px; }
.empty { color: #8a9099; font-size: 13px; padding: 8px 0; }
.explain { font-size: 13px; line-height: 1.6; color: #4a5159; }
/* diff 专用 */
.stats { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
.stat { background: #fff; border: 1px solid #e8eaed; border-radius: 8px; padding: 10px 14px; min-width: 92px; }
.stat .n { font-size: 22px; font-weight: 700; }
.stat .l { font-size: 12px; color: #8a9099; }
.stat.add .n { color: #1a7f37; } .stat.del .n { color: #cf222e; }
.stat.chg .n { color: #b08800; } .stat.neu .n { color: #5a626c; }
.ck { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 10px; font-weight: 600; margin-right: 6px; }
.ck.add { background: #e6f4ea; color: #1a7f37; }
.ck.del { background: #ffebe9; color: #cf222e; }
.ck.status { background: #fff1e0; color: #bc4a00; }
.ck.summary { background: #ddf4ff; color: #0969da; }
.ck.trigger { background: #fff8c5; color: #7d5e00; }
.ck.region { background: #f1e8fd; color: #8250df; }
.ck.multi { background: #f0f1f3; color: #5a626c; }
.arrow { color: #8a9099; margin: 0 4px; }
.del-line { color: #8a9099; }   /* removed 行弱化 */
"""


class HtmlRenderer:
    """BrainSnapshot → 自包含 HTML 字符串。"""

    def render(self, snapshot: BrainSnapshot) -> str:
        parts = [
            _DOCTYPE,
            "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>{_esc('BrainRegion 快照')}</title>",
            f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
            self._header(snapshot),
            self._kpis(snapshot.kpis),
            self._regions(snapshot.regions, snapshot.has_query),
            self._memory(snapshot.memory),
            self._runs(snapshot.runs),
            self._calibration(snapshot.calibration),
        ]
        if snapshot.activation is not None:
            parts.append(self._activation(snapshot.activation))
        parts.append("</div></body></html>")
        return "".join(parts)

    # ── 段 ────────────────────────────────────────────────────────────────────
    def _header(self, s: BrainSnapshot) -> str:
        return (
            "<header><h1>🧠 BrainRegion 脑状态快照</h1>"
            f"<div class=\"meta\">生成于 {_esc(_fmt_ts(s.generated_at))}"
            f" · brainregion {_esc(s.brainregion_version)}"
            f" · snapshot schema {_esc(s.schema_version)}</div></header>"
        )

    def _kpis(self, kpis) -> str:
        cards = []
        for k in kpis:
            cards.append(
                f"<div class=\"kpi {_esc(k.status)}\"><div class=\"label\">{_esc(k.label)}</div>"
                f"<div class=\"value\">{_esc(k.value)}</div>"
                f"<div class=\"hint\">{_esc(k.hint)}</div></div>"
            )
        return f"<div class=\"kpis\">{''.join(cards)}</div>"

    def _regions(self, regions, has_query: bool) -> str:
        if not regions:
            return "<section><h2>脑区</h2><div class=\"empty\">暂无脑区</div></section>"
        cards = []
        for r in regions:
            inactive = max(0, r.total - r.recallable)
            if r.woke == "yes":
                badge = "<span class=\"badge woke\">已唤醒</span>"
            elif has_query:
                badge = "<span class=\"badge quiet\">—</span>"
            else:
                badge = ""
            nums = f"<b>{_esc(r.total)}</b> 条 · <b>{_esc(r.recallable)}</b> 可召回"
            if inactive:
                nums += f" · <span class=\"muted\">{_esc(inactive)} 失效</span>"
            act = ""
            if has_query:
                pct = int(max(0.0, min(1.0, float(getattr(r, "confidence", 0.0)))) * 100)
                phase = getattr(r, "phase", "") or ("woken" if r.woke == "yes" else "quiet")
                tools = ", ".join(getattr(r, "action_tools", ()) or ())
                action_hint = f" · actions {getattr(r, 'suggested_actions', 0)}" if getattr(r, "suggested_actions", 0) else ""
                tools_hint = f" · {_esc(tools)}" if tools else ""
                act = (
                    f"<div class=\"act\"><div class=\"bar {_esc(phase)}\"><span style=\"width:{_esc(pct)}%\"></span></div>"
                    f"{_esc(_phase_label(phase))} · intensity {_esc(pct)}% · score {_esc(getattr(r, 'score', 0))}"
                    f"{action_hint}{tools_hint}</div>"
                )
            cards.append(f"<div class=\"region\"><div class=\"name\">{_esc(r.region)}{badge}</div>"
                         f"<div class=\"nums\">{nums}</div>{act}</div>")
        return f"<section><h2>脑区</h2><div class=\"regions\">{''.join(cards)}</div></section>"

    def _memory(self, memory: dict) -> str:
        if not memory:
            return ""
        health = memory.get("health") or {}
        by_status = health.get("by_status") or {}
        status_chips = "".join(
            f"<span class=\"chip {_esc(s)}\">{_esc(_gov_label(s))} {_esc(n)}</span>"
            for s, n in sorted(by_status.items()) if n
        )
        recallable = health.get("recallable", 0)
        non_recallable = health.get("non_recallable", 0)
        expired = health.get("expired_count", 0)
        parts = [
            "<section><h2>记忆健康</h2>",
            f"<div class=\"explain\">{_esc(recallable)} 可召回 · {_esc(non_recallable)} 失效"
            f" · {_esc(expired)} 过期 · 共 {_esc(memory.get('total', 0))}</div>",
            f"<div style=\"margin:10px 0\">{status_chips}</div>" if status_chips else "",
        ]
        preview = memory.get("preview") or []
        if preview:
            parts.append("<table><tbody>")
            for e in preview:
                st = e.get("status", "active")
                parts.append(
                    "<tr>"
                    f"<td class=\"mono\">{_esc(e.get('region') or '(global)')}</td>"
                    f"<td>{_esc(e.get('summary'))}</td>"
                    f"<td><span class=\"chip {_esc(st)}\">{_esc(_gov_label(st))}</span></td>"
                    f"<td class=\"muted\">{_esc(e.get('age_days'))} 天</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
        parts.append("</section>")
        return "".join(parts)

    def _runs(self, runs: dict) -> str:
        if not runs:
            return "<section><h2>最近 Run</h2><div class=\"empty\">无 Run</div></section>"
        if "gate" in runs or "timeline" in runs:  # run_id 单 run 详情
            return self._run_detail(runs)
        return self._run_history(runs)

    def _run_history(self, runs: dict) -> str:
        history = runs.get("history") or []
        if not history:
            return "<section><h2>最近 Run</h2><div class=\"empty\">无 Run</div></section>"
        rows = []
        for r in history:
            rows.append(
                "<tr>"
                f"<td class=\"mono\">{_esc(r.get('run_id'))}</td>"
                f"<td>{_esc(_fmt_ts(r.get('date')))}</td>"
                f"<td><span class=\"dec {_dec_class(r.get('status'))}\">{_esc(r.get('status'))}</span></td>"
                f"<td>{_esc(_fmt_cost(r.get('cost_usd')))}</td>"
                f"<td>{_esc(r.get('n_tasks'))}</td>"
                "</tr>"
            )
        return ("<section><h2>最近 Run</h2><table><thead><tr>"
                "<th>Run</th><th>时间</th><th>闸门</th><th>成本</th><th>任务</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></section>")

    def _run_detail(self, runs: dict) -> str:
        run = runs.get("run") or {}
        gate = runs.get("gate") or {}
        decision = gate.get("decision")
        parts = [
            "<section><h2>Run 详情</h2>",
            f"<div class=\"explain\"><b>{_esc(run.get('run_id'))}</b>"
            f" · {_esc(run.get('n_tasks'))} 个任务 · {_esc(_fmt_ts(run.get('date')))}</div>",
            f"<div style=\"margin:8px 0\">闸门：<span class=\"dec {_dec_class(decision)}\">{_esc(decision or '—')}</span></div>",
        ]
        timeline = runs.get("timeline") or []
        if timeline:
            stage_names = list((timeline[0].get("stages") or {}).keys())
            head = "".join(f"<th class=\"sym\">{_esc(_stage_label(s))}</th>" for s in stage_names)
            body = []
            for row in timeline:
                syms = row.get("symbols") or {}
                stages = row.get("stages") or {}
                cells = "".join(
                    f"<td class=\"sym\" title=\"{_esc(_stage_status_label(stages.get(s, '')))}\">"
                    f"{_esc(syms.get(s, '?'))}</td>"
                    for s in stage_names
                )
                body.append(
                    f"<tr><td class=\"mono\">{_esc(row.get('task_id'))}</td>"
                    f"<td class=\"mono\">{_esc(row.get('variant'))}</td>{cells}</tr>"
                )
            parts.append(
                "<table class=\"timeline\"><thead><tr><th>任务</th><th>变体</th>"
                f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            )
            # 图例
            legend = " ".join(
                f"{_esc(status_symbol(s))}={_esc(_stage_status_label(s))}"
                for s in ("SUCCESS", "FAILED", "SKIPPED", "UNKNOWN", "NOT_INSTRUMENTED")
            )
            parts.append(f"<div class=\"muted\" style=\"margin-top:6px;font-size:12px\">{legend}</div>")
        parts.append("</section>")
        return "".join(parts)

    def _calibration(self, cal: dict) -> str:
        if not cal:
            return ""
        blocked = cal.get("am_i_blocked")
        badge = ("<span class=\"badge\" style=\"background:#ffebe9;color:#cf222e\">阻塞</span>"
                 if blocked else "<span class=\"badge woke\">通过</span>")
        parts = [
            f"<section><h2>校准 {badge}</h2>",
            f"<div class=\"explain\">{_esc(cal.get('passed_count', 0))}/{_esc(cal.get('n', 0))} 个 judge 已校准</div>",
        ]
        not_passed = cal.get("not_passed") or []
        if not_passed:
            parts.append("<table><thead><tr><th>judge</th><th>模型</th><th>Wilson 下界</th><th>阈值</th></tr></thead><tbody>")
            for r in not_passed:
                parts.append(
                    "<tr>"
                    f"<td class=\"mono\">{_esc(r.get('judge_id'))}</td>"
                    f"<td>{_esc(r.get('judge_model'))}</td>"
                    f"<td>{_esc(_fmt_num(r.get('wilson_lower')))}</td>"
                    f"<td>{_esc(_fmt_num(r.get('threshold')))}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
        parts.append("</section>")
        return "".join(parts)

    def _activation(self, act: dict) -> str:
        metrics = act.get("wake_metrics") or {}
        woken = act.get("woken") or []
        call = act.get("call_status") or {}
        rows = [
            ("唤醒", ", ".join(woken)),
            ("命中", ", ".join(metrics.get("hit") or [])),
            ("漏唤醒", ", ".join(metrics.get("missed") or [])),
            ("误唤醒", ", ".join(metrics.get("false_wake") or [])),
            ("评分状态", metrics.get("metrics_status")),
        ]
        rows.extend([
            ("models_called", call.get("models_called")),
            ("suggested_actions", call.get("suggested_actions_count")),
            ("requires_user_approval", call.get("requires_user_approval_count")),
            ("action_tools", ", ".join(call.get("action_tools") or [])),
        ])
        body = "".join(
            f"<tr><td>{_esc(label)}</td><td>{_esc(val) or '—'}</td></tr>" for label, val in rows
        )
        return (
            "<section><h2>激活</h2>"
            f"<div class=\"explain\">{_esc(act.get('explain'))}</div>"
            f"<table><thead><tr><th>指标</th><th>值</th></tr></thead><tbody>{body}</tbody></table></section>"
        )


class DiffHtmlRenderer:
    """BrainDiff → 自包含 HTML(可视化 Phase 2)。零业务逻辑,全读 diff.summary / diff.* 预算值。"""

    _KIND_BADGE = {
        ADDED: ("新增", "add"), REMOVED: ("移除", "del"),
        STATUS_CHANGED: ("状态变", "status"), SUMMARY_CHANGED: ("摘要变", "summary"),
        TRIGGER_CHANGED: ("Trigger 变", "trigger"), REGION_CHANGED: ("脑区变", "region"),
        MULTI: ("多项变", "multi"),
    }

    def render(self, diff: BrainDiff) -> str:
        parts = [
            _DOCTYPE,
            "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"<title>{_esc('BrainRegion Diff')}</title>",
            f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
            self._header(diff),
            self._summary_hero(diff.summary),
            self._kpis(diff.kpis_a, diff.kpis_b),
            self._memory(diff),
            self._regions(diff.regions),
            self._runs(diff.runs, diff.summary),
            self._calibration(diff.calibration),
        ]
        if diff.notes:
            parts.append("<section><h2>说明</h2><div class=\"explain\">"
                         + "<br>".join(_esc(n) for n in diff.notes) + "</div></section>")
        parts.append("</div></body></html>")
        return "".join(parts)

    # ── 段(renderer 只读不算)────────────────────────────────────────────────
    def _header(self, d: BrainDiff) -> str:
        a, b = d.a_meta, d.b_meta
        return (
            "<header><h1>🔄 BrainRegion Diff</h1>"
            f"<div class=\"meta\">{_esc(a.get('label','A'))} {_esc(_fmt_ts(a.get('generated_at')))}"
            f"{_ql(a)} <span class=\"arrow\">→</span> "
            f"{_esc(b.get('label','B'))} {_esc(_fmt_ts(b.get('generated_at')))}{_ql(b)}</div></header>"
        )

    def _summary_hero(self, s: dict) -> str:
        """Executive Summary:3 秒看完脑变化(GPT r1⑤)。直读 s,不自算。"""
        chips = []
        def chip(n, label, cls):
            if n:
                chips.append(f"<div class=\"stat {cls}\"><div class=\"n\">{_esc(n)}</div>"
                             f"<div class=\"l\">{_esc(label)}</div></div>")
        chip(f"+{s.get('added',0)}", "新增记忆", "add")
        chip(f"{s.get('changed',0)}", "变化记忆", "chg")
        chip(f"−{s.get('removed',0)}", "移除记忆", "del")
        chip(f"+{s.get('regions_added',0)}" if s.get('regions_added',0) else "",
             "新增脑区", "neu")
        chip(f"+{s.get('new_runs',0)}", "新 Run", "neu")
        if s.get("total_a", 0) != s.get("total_b", 0):
            chips.append(f"<div class=\"stat neu\"><div class=\"n\">{_esc(s.get('total_a',0))}"
                         f"<span class=\"arrow\">→</span>{_esc(s.get('total_b',0))}</div>"
                         "<div class=\"l\">记忆总数</div></div>")
        if s.get("gate_a") != s.get("gate_b"):
            chips.append(f"<div class=\"stat neu\"><div class=\"n dec {_dec_class(s.get('gate_a'))}\">"
                         f"{_esc(s.get('gate_a'))}<span class=\"arrow\">→</span>"
                         f"<span class=\"dec {_dec_class(s.get('gate_b'))}\">{_esc(s.get('gate_b'))}</span></div>"
                         "<div class=\"l\">闸门</div></div>")
        if s.get("blocked_a") != s.get("blocked_b"):
            chips.append("<div class=\"stat neu\"><div class=\"n\">校准</div><div class=\"l\">"
                         + ("通过→阻塞" if s.get("blocked_b") else "阻塞→通过") + "</div></div>")
        if not chips:
            return "<section><div class=\"stat neu\" style=\"display:inline-block\"><div class=\"n\">无变化</div></div></section>"
        return f"<section><h2>Executive Summary</h2><div class=\"stats\">{''.join(chips)}</div></section>"

    def _kpis(self, kpis_a, kpis_b) -> str:
        b_by_label = {k.label: k for k in kpis_b}
        rows = []
        for ka in kpis_a:
            kb = b_by_label.get(ka.label)
            if kb is None:
                continue
            rows.append(
                f"<tr><td>{_esc(ka.label)}</td>"
                f"<td class=\"dec {_dec_class_value(ka.status)}\">{_esc(ka.value)}</td>"
                f"<td class=\"arrow\">→</td>"
                f"<td class=\"dec {_dec_class_value(kb.status)}\">{_esc(kb.value)}</td></tr>"
            )
        return ("<section><h2>KPI A → B</h2><table><thead><tr><th>指标</th><th>A</th><th></th>"
                f"<th>B</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>")

    def _memory(self, d: BrainDiff) -> str:
        mem = d.memory
        parts = ["<section><h2>记忆变化</h2>"]
        changed = mem.get("changed") or []
        added = mem.get("added") or []
        removed = mem.get("removed") or []
        if not (changed or added or removed):
            parts.append("<div class=\"empty\">无 experience 级变化</div></section>")
            return "".join(parts)
        # 顺序 Changed → Added → Removed(GPT r2⑥)
        if changed:
            parts.append("<h2 style=\"font-size:13px\">变化</h2><table><tbody>")
            for c in changed:
                parts.append(self._changed_row(c))
            parts.append("</tbody></table>")
        if added:
            parts.append("<h2 style=\"font-size:13px\">新增</h2><table><tbody>")
            for c in added:
                parts.append(self._addremove_row(c, "add"))
            parts.append("</tbody></table>")
        if removed:
            parts.append("<h2 style=\"font-size:13px\">移除</h2><table><tbody>")
            for c in removed:
                parts.append(self._addremove_row(c, "del"))
            parts.append("</tbody></table>")
        parts.append("</section>")
        return "".join(parts)

    def _changed_row(self, c) -> str:
        label, cls = self._KIND_BADGE.get(c.change_kind, (c.change_kind or "?", "multi"))
        identity = (c.after or c.before).summary
        detail = self._change_detail(c)
        return (f"<tr><td><span class=\"ck {cls}\">{_esc(label)}</span>{_esc(identity)}</td>"
                f"<td>{detail}</td></tr>")

    def _change_detail(self, c) -> str:
        b, a, kind = c.before, c.after, c.change_kind
        segs = []
        if kind in (STATUS_CHANGED, MULTI):
            segs.append(f"状态 <span class=\"ck status\">{_esc(_gov_label(b.status))}</span>"
                        f"<span class=\"arrow\">→</span><span class=\"ck status\">{_esc(_gov_label(a.status))}</span>")
        if kind in (TRIGGER_CHANGED, MULTI):
            segs.append(f"trigger <span class=\"mono\">{_esc(', '.join(b.triggers)) or '∅'}</span>"
                        f"<span class=\"arrow\">→</span><span class=\"mono\">{_esc(', '.join(a.triggers)) or '∅'}</span>")
        if kind in (SUMMARY_CHANGED, MULTI):
            segs.append(f"摘要 {_esc(b.summary)}<span class=\"arrow\">→</span>{_esc(a.summary)}")
        if kind in (REGION_CHANGED, MULTI):
            segs.append(f"脑区 {_esc(b.region)}<span class=\"arrow\">→</span>{_esc(a.region)}")
        return " ".join(segs)

    def _addremove_row(self, c, cls) -> str:
        e = c.after or c.before
        label = self._KIND_BADGE.get(c.change_kind, (c.change_kind, cls))[0]
        return (f"<tr class=\"{'del-line' if cls == 'del' else ''}\">"
                f"<td><span class=\"ck {cls}\">{_esc(label)}</span>{_esc(e.summary)}</td>"
                f"<td class=\"mono\">{_esc(e.region)}</td>"
                f"<td><span class=\"ck {_gov(e.status)}\">{_esc(_gov_label(e.status))}</span></td>"
                f"<td class=\"mono muted\">{_esc(', '.join(e.triggers)) or '∅'}</td></tr>")

    def _regions(self, regions: dict) -> str:
        added = regions.get("added") or []
        removed = regions.get("removed") or []
        deltas = regions.get("deltas") or []
        if not added and not removed and not deltas:
            return ""
        parts = ["<section><h2>脑区变化</h2><table><thead><tr><th>脑区</th><th>A</th><th></th><th>B</th></tr></thead><tbody>"]
        for r in added:
            parts.append(f"<tr><td>{_esc(r)}</td><td>—</td><td class=\"arrow\">→</td>"
                         f"<td><b>{'+新增'}</b></td></tr>")
        for r in removed:
            parts.append(f"<tr class=\"del-line\"><td>{_esc(r)}</td><td><b>消失</b></td>"
                         f"<td class=\"arrow\">→</td><td>—</td></tr>")
        for dlt in deltas:
            a, b = dlt.get("a", 0), dlt.get("b", 0)
            sign = "+" if b > a else ""
            parts.append(f"<tr><td>{_esc(dlt.get('region'))}</td><td>{_esc(a)}</td>"
                         f"<td class=\"arrow\">→</td><td><b>{sign}{_esc(b)}</b></td></tr>")
        parts.append("</tbody></table></section>")
        return "".join(parts)

    def _runs(self, runs: dict, summary: dict) -> str:
        new = runs.get("new") or []
        if not new:
            return ""
        rows = "".join(
            f"<tr><td class=\"mono\">{_esc(r.get('run_id'))}</td>"
            f"<td><span class=\"dec {_dec_class(r.get('status'))}\">{_esc(r.get('status'))}</span></td></tr>"
            for r in new
        )
        return ("<section><h2>新增 Run(B 有 A 无)</h2><table><thead><tr><th>Run</th><th>闸门</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></section>")

    def _calibration(self, cal: dict) -> str:
        blocked_a = cal.get("blocked_a")
        blocked_b = cal.get("blocked_b")
        new_np = cal.get("new_not_passed") or []
        if blocked_a == blocked_b and not new_np:
            return ""
        parts = ["<section><h2>校准变化</h2>"]
        if blocked_a != blocked_b:
            parts.append(f"<div class=\"explain\">校准状态: "
                         f"{'通过' if not blocked_a else '阻塞'}<span class=\"arrow\">→</span>"
                         f"{'通过' if not blocked_b else '阻塞'}</div>")
        if new_np:
            parts.append("<table><thead><tr><th>新增未通过 judge</th><th>模型</th></tr></thead><tbody>")
            for r in new_np:
                parts.append(f"<tr><td class=\"mono\">{_esc(r.get('judge_id'))}</td>"
                             f"<td>{_esc(r.get('judge_model'))}</td></tr>")
            parts.append("</tbody></table>")
        parts.append("</section>")
        return "".join(parts)


def _ql(meta: dict) -> str:
    """query_label 片段(有则显示)。"""
    ql = meta.get("query_label") or ""
    return f" · {_esc(ql)}" if ql else ""


def _gov(status: str) -> str:
    """governance status → chip CSS class(原始枚举即 class,同 Phase 1)。"""
    return status or ""


def _dec_class_value(status: str) -> str:
    """KPI status(ok/warn/bad/neutral)→ dec class(复用 _dec_class 的色彩思路但独立)。"""
    return {"ok": "dec-go", "bad": "dec-no_go", "warn": "dec-inconclusive"}.get(status or "", "dec-neutral")


def _dec_class(dec) -> str:
    """gate/run decision → CSS class。"""
    d = (dec or "").upper()
    if d == "GO" or d == "OK":
        return "dec-go"
    if "NO_GO" in d or "FAIL" in d:
        return "dec-no_go"
    if "INCONCLUSIVE" in d:
        return "dec-inconclusive"
    return "dec-neutral"


def _fmt_cost(v) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):.4f}"
    except Exception:  # noqa: BLE001
        return str(v)


def _fmt_num(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except Exception:  # noqa: BLE001
        return str(v)
