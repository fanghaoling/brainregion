"""External consultation engine."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from ..errors import classify_error
from ..stages.review import select_jobs_within_budget
from .guard import prepare_request, summarize_unparseable_output
from .parse import parse_advice
from .prompt import render_consult_prompt
from .report import ConsultReport, ConsultRequest
from .synthesize import synthesize_report
from ..consultants.loader import load_consultant, list_consultants
from ..context import render_context_blocks


class ConsultEngine:
    """Run consultant roles across a model panel and synthesize the result."""

    def __init__(self, *, backend: Any, consultants_dir: str | Path) -> None:
        self.backend = backend
        self.consultants_dir = Path(consultants_dir)

    async def consult(
        self,
        request: ConsultRequest,
        *,
        panel: list[dict],
        consultants: list[str],
        max_input_chars: int = 24000,
        max_cost_usd: float | None = None,
        effort: str | None = None,
        consultation_id: str | None = None,
        context_blocks: list | None = None,
    ) -> ConsultReport:
        consultation_id = consultation_id or f"consult-{uuid.uuid4().hex[:12]}"
        sanitized, guard_meta = prepare_request(request, max_input_chars=max_input_chars)
        # ContextProvider 召回块（memory/git/…,config memory_inject 门控 consult provider-loop 注入）：
        # 渲染一次（跨 panel×consultant 共用），cap 到 max_input_chars//4 防 budget 挤压。
        context_block_str = ""
        if context_blocks:
            rendered = render_context_blocks(list(context_blocks))
            cap = max(0, max_input_chars // 4)
            if cap and len(rendered) > cap:
                rendered = rendered[:cap]
            context_block_str = rendered
            guard_meta["context_blocks"] = len(context_blocks)
        available = set(list_consultants(self.consultants_dir))
        unknown = [name for name in consultants if name not in available]
        if unknown:
            raise ValueError(f"未知 consultant: {unknown}，可用: {sorted(available)}")

        roles = {name: load_consultant(name, self.consultants_dir) for name in consultants}
        jobs: list[dict] = []
        for entry in panel:
            for consultant, role in roles.items():
                system, user = render_consult_prompt(sanitized, role, context_block=context_block_str)
                jobs.append(
                    {
                        "model": entry["model"],
                        "label": entry["label"],
                        "endpoint_id": entry.get("endpoint_id"),
                        "consultant": consultant,
                        "system": system,
                        "user": user,
                        "temperature": float(role.get("temperature", 0.2)),
                        "top_p": float(role.get("top_p", 0.95)),
                        "max_tokens": int(role.get("max_tokens", 2048)),
                    }
                )

        jobs_total = len(jobs)
        estimated_cost_usd = 0.0
        budget_exhausted = False
        if max_cost_usd is not None and jobs:
            jobs, estimated_cost_usd, budget_exhausted = select_jobs_within_budget(jobs, float(max_cost_usd))
        jobs_run = len(jobs)

        budget = {
            "max_usd": max_cost_usd,
            "estimated_usd": estimated_cost_usd,
            "jobs_run": jobs_run,
            "jobs_total": jobs_total,
            "exhausted": budget_exhausted,
        }
        if not jobs:
            return synthesize_report(
                consultation_id=consultation_id,
                advice=[],
                failed_models=[],
                usage={"total_tokens": 0, "cost_usd": 0.0},
                budget=budget,
                guard=guard_meta,
            )

        async def _one(job: dict) -> dict:
            resp = await self.backend.complete(
                model=job["model"],
                system=job["system"],
                user=job["user"],
                temperature=job["temperature"],
                top_p=job["top_p"],
                max_tokens=job["max_tokens"],
                effort=effort,
                endpoint_id=job.get("endpoint_id"),
            )
            return {"job": job, "response": resp}

        raw_results = await asyncio.gather(*(_one(job) for job in jobs), return_exceptions=True)
        advice = []
        failed_models: list[dict] = []
        total_tokens = 0
        cost_usd = 0.0
        advice_index = 0

        for raw in raw_results:
            if isinstance(raw, Exception):
                failed_models.append(
                    {"model": "", "consultant": "", "error": f"{type(raw).__name__}: {raw}", "type": "unknown"}
                )
                continue
            job = raw["job"]
            resp = raw["response"]
            label = job["label"]
            consultant = job["consultant"]
            if getattr(resp, "usage", None):
                total_tokens += int(resp.usage.get("total_tokens") or 0)
            if getattr(resp, "cost_usd", None):
                cost_usd += float(resp.cost_usd or 0.0)
            if not resp.ok:
                classified = classify_error(resp.error or "")
                failed_models.append(
                    {
                        "model": label,
                        "consultant": consultant,
                        "error": resp.error,
                        "type": classified["type"],
                        "hint": classified["hint"],
                    }
                )
                continue
            parsed = parse_advice(
                resp.content,
                model=label,
                consultant=consultant,
                advice_id=f"{consultation_id}-{advice_index}",
            )
            if parsed is None:
                diagnostics = summarize_unparseable_output(resp.content)
                if diagnostics.get("has_object_start"):
                    hint = "输出含 JSON 但解析失败（可能被截断）；提高 max_tokens 或降低 effort 后重试。"
                else:
                    hint = (
                        "模型未输出 JSON 对象（可能 reasoning 思考耗尽 max_tokens 或未遵循格式）；"
                        "提高 max_tokens、降低 effort、或换非 reasoning 模型。"
                    )
                failed_models.append(
                    {
                        "model": label,
                        "consultant": consultant,
                        "error": "输出无法解析为 consult JSON",
                        "type": "parse_error",
                        "hint": hint,
                        "diagnostics": diagnostics,
                    }
                )
            else:
                advice.append(parsed)
                advice_index += 1

        return synthesize_report(
            consultation_id=consultation_id,
            advice=advice,
            failed_models=failed_models,
            usage={"total_tokens": total_tokens, "cost_usd": round(cost_usd, 6)},
            budget=budget,
            guard=guard_meta,
        )
