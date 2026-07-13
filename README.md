# 脑区 BrainRegion

[English](README.md) | [简体中文](README.zh-CN.md)

![CI](https://github.com/fanghaoling/brainregion/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/fanghaoling/brainregion.svg)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)

BrainRegion is AI collaboration infrastructure for review, consultation, planning, and memory.

This project was formerly `design-review-mcp`. The internal Python package has moved to `brainregion`, while the old
CLI command aliases remain available during the rename.

The current MCP server and CLI can fan out a plan, source change, or document to multiple LLM reviewer roles, retrieve
project-specific knowledge, normalize duplicate findings, and return consensus-oriented reports that are easier to act
on. It also includes external consultation tools for asking another expert model when the main assistant is stuck.

The core pipeline is project-agnostic. Project-specific behavior lives in adapters, so the default experience stays
useful for general product, architecture, code, and document design reviews. Optional adapters can add domain knowledge
without changing the core pipeline.

## Highlights

- Review plans, code, Markdown, ADRs, RFCs, and config documents.
- Use reviewer roles such as `planner`, `safety`, `architecture`, `performance`, `feasibility`, and `visionary`.
- Run one model or a panel of models, including official LiteLLM providers and OpenAI/Anthropic-compatible gateways.
- Retrieve framework and project-local knowledge before review.
- Normalize findings into canonical buckets and separate consensus, majority, and individual issues.
- Render JSON, Markdown, and SARIF output.
- Track review memory with `mark_finding` so accepted/rejected findings can influence later confidence calibration.
- Ask external consultant models with `consult_problem` and record useful advice with `mark_advice`.
- Generate executable task plans with `plan_task`, then review the plan before implementation.
- Route a goal/problem to likely Brain Regions with `route_regions` as a local, deterministic precursor to context scheduling.
- Suggest explicit manual next steps with `suggest_workflow` without auto-calling tools or models.
- Inspect model routing with `list_model_routes` so bare model names and endpoint-backed models are not confused.
- Attach model profile metadata such as `cheap`, `fast`, `flagship`, `sleep`, or `awake` for preflight visibility.
- Recommend a model panel with `suggest_panel` from profile tags and cost/speed/quality scores without calling models.
- Plan auditable Skill/Region wakes with `plan_region_activation`, including hard prerequisites, deny conditions, and bounded context requests.
- Merge defaults from builtin values, global config, project config, environment variables, and explicit call arguments.

## Architecture

Most pieces are swappable. Adapter-specific behavior stays out of `core/`.

| Layer | Contract | Default implementation |
|---|---|---|
| `ModelBackend` | `async complete(...)` | `LiteLLMBackend` |
| `KnowledgeProvider` | `retrieve/list_cases/add_case` | `YamlKnowledgeProvider` |
| `ProjectAdapter` | `read_context/version/convention + reviewers/knowledge` | `GenericAdapter`, optional domain adapters |
| `ReportRenderer` | `render(ReviewReport)` | Markdown / JSON / SARIF renderers |
| `Stage` | `process(ctx) -> ctx` | retrieve, context, prompt, review, parse, normalize, consensus, score |

Adding another project type should usually mean adding a new adapter package, not changing the core pipeline.

## Review Pipeline

```text
ReviewDocument
  -> RetrieveStage
  -> ContextStage
  -> PromptStage
  -> ReviewStage      # fan-out across panel x dimensions
  -> ParseStage
  -> NormalizeStage   # canonical finding buckets
  -> ConsensusStage
  -> ScoreStage
  -> ReviewReport
```

The pipeline is designed to reduce "confident but unsupported" feedback:

- Findings need evidence quotes.
- Knowledge retrieval can inject project gotchas and version-specific cases.
- Reviewer prompts are role-specific.
- Canonical normalization reduces duplicate phrasing across models.
- Calibrated confidence combines model agreement, severity, retrieval hits, and review memory.

## External Consultation

`consult_problem` is for moments when the main assistant is stuck, uncertain, repeatedly debugging the same issue, or
needs another expert perspective. It does not execute commands or edit files; it returns structured advice, hypotheses,
next experiments, risks, and a recommended plan.

```python
consult_problem(
    problem="FlowField updates occasionally deadlock",
    context="Unity ECS project; double buffering and JobHandle.CombineDependencies were already tried.",
    logs="Occasionally stalls near CompleteDependency()",
    attempts=["double buffering", "combined JobHandle dependencies"],
    mode="architecture",
)
```

Common modes:

- `debugging`: root-cause diagnosis.
- `architecture`: boundaries, state flow, and maintainability.
- `performance`: latency, throughput, token/API cost.
- `simplicity`: YAGNI and smaller MVP slices.
- `game_design`: gameplay and player experience.
- `challenge`: adversarial challenge to the current thinking.
- `planning`: task decomposition, risks, and acceptance criteria.

Recommended config:

```jsonc
{
  "consult_panel": ["modelbridge_openai/gpt-5.4-mini"],
  "consult_consultants": ["debugger", "critic"],
  "consult_max_cost_usd": 0.03,
  "consult_max_input_chars": 24000
}
```

If `consult_panel` is not configured, consultation falls back to `panel`. For day-to-day use, keep a cheaper/faster
consult panel so one consultation does not expand the full review panel.

`consult_problem` returns a `consultation_id`, and each item in `individual` has a stable advice `id`. Mark useful or
unhelpful advice with Advice Memory:

```python
mark_advice(
    advice_id="consult-abc123-0",
    consultation_id="consult-abc123",
    decision="accepted",
    reason="identified the real race condition",
    outcome="added the suggested minimal reproduction test",
)
```

`decision` is one of `accepted`, `rejected`, `partial`, or `unknown`. The database stores advice metadata and user
feedback, not the original prompt, problem text, or full advice body.

## Planning

`plan_task` turns a goal into a structured, reviewable implementation plan. It is intentionally a thin Planner MVP:
it does not execute commands, does not edit files, and does not run a multi-model debate. It tries the configured model
panel in order and returns the first parseable plan.

```python
plan_task(
    goal="Add a Planner MVP to BrainRegion",
    context="Python MCP server with existing consult_problem and review_plan tools.",
    constraints=[
        "Do not auto-execute tasks.",
        "Reuse existing budget and input guardrails.",
    ],
    success_criteria=[
        "The MCP tool returns milestones, tasks, risks, acceptance criteria, and tests.",
        "Unit tests cover parsing and routing.",
    ],
)
```

Recommended flow:

```text
Goal -> plan_task -> review_plan -> implement -> review_code -> mark_finding / mark_advice
```

Optional config:

```jsonc
{
  "planner_panel": ["modelbridge_openai/gpt-5.4-mini"],
  "planner_max_cost_usd": 0.03,
  "planner_max_input_chars": 24000
}
```

If `planner_panel` is not configured, planning falls back to `consult_panel`, then `panel`.

## Brain Regions

`route_regions` is the first small step toward region-based context scheduling. It is deliberately local and
deterministic: it does not call models, read memory, or trigger review/consult/planner tools. It only ranks static
region definitions by explicit triggers and returns an activation trace.

```python
route_regions(
    goal="Optimize a Unity ECS FlowField system that allocates too much memory",
    files={
        "Assets/Scripts/FlowFieldSystem.cs": "...",
    },
    top_k=3,
)
```

Example result shape:

```jsonc
{
  "selected": [
    {"id": "unity_ecs", "score": 4, "matched_triggers": [...]},
    {"id": "performance", "score": 4, "matched_triggers": [...]}
  ],
  "trace": {
    "strategy": "deterministic_keyword_v1",
    "input": {"file_contents_used": false}
  }
}
```

Built-in regions currently include `planning`, `review`, `debugging`, `performance`, `security`, `memory`, `research`,
and `unity_ecs`. This tool is advisory; future schedulers must explicitly decide whether to consume its result.

### Structured Activation Contracts

plan_region_activation is the next deterministic layer after candidate routing. Existing Skill manifests may declare
an activation contract with positive signals, deny conditions, required tools/capabilities, context selectors, and
budgets. Manifests without a contract remain inert.

    plan_region_activation(
        task_intents=["diagnose_failure"],
        events=["test_failed"],
        available_tools=["git"],
        max_regions=2,
        max_context_tokens=2000,
    )

The result records every Skill as wake, skip, or defer. A wake includes a bounded context_request; a skip includes
the failed prerequisite or deny reason. Defer is reserved for a future cheap semantic gate. The hard gate itself never
calls a model, retrieves memory, or executes a tool.

`load_region_context` consumes the same hard-gate plan and materializes only activated provider Skills into
short-lived `ContextBlock` values:

    load_region_context(
        query="Why did the parser keep failing after several repair attempts?",
        events=["repeated_attempt_failed"],
        scope_regions=["debugging"],
        max_context_tokens=2000,
    )

Provider failures are isolated, and both block count and estimated tokens are bounded. Advisory/action Skills remain
visible in the activation trace but are not executed by this call. Retrieval does not call a model, write memory, or
retain the returned context in the runtime.

### Cognitive Workspace Delivery

`stage_region_context` loads activated provider context into a process-local task workspace and returns only a
delivery receipt. Region-private block contents are not included in the staging result:

    stage_region_context(
        task_id="parser-debug",
        query="Repeated parser failures after several repair attempts",
        events=["repeated_attempt_failed"],
        audience="region",
        target_region="debugging",
        ttl_steps=3,
    )

`workspace_context` provides four lifecycle operations through one MCP tool:

- `read`: main consumers see `main + shared`; region consumers see `their region + shared`.
- `inspect`: returns delivery metadata and evidence references without context contents.
- `advance`: decrements explicit task-step TTL and unloads expired entries.
- `clear`: unloads every workspace entry for one task.

The workspace stores evidence and structured task context, not model chain of thought. It is an architectural routing
boundary rather than an authorization boundary; process restart also clears it. Read operations apply a fresh block
and estimated-token budget before returning context.

### Region Status and Escalation

A successful stage now records a `RegionContextReceipt` containing observable retrieval state: provider outcomes,
block/token counts, evidence references, removed stale candidates, conflicts, and selector coverage. Coverage remains
`unverified` until a provider explicitly confirms selectors; retrieval success is not treated as proof that a model
understands the task.

Experts publish structured reports through the existing workspace tool:

    workspace_context(
        task_id="parser-debug",
        operation="publish_report",
        report={
            "region": "debugging",
            "state": "needs_decision",
            "summary": "Historical constraints change the parser design.",
            "evidence_refs": ["memory:id:exp-parser"],
            "context_state": "ready",
            "decision_scope": "architecture",
            "risk": "medium",
            "memory_impact": "decision_changing",
            "reversible": True,
        },
    )

The escalation policy is deterministic. Routine, reversible work returns `continue`; insufficient context returns
`request_context`; blocked/decision states, decision-changing or contradictory memory, conflicted/stale context,
architecture/user/cross-region scope, high risk, irreversible actions, repeated failure, or user choice return
`notify_main`. `status` exposes compact region state, while `inbox` contains only `notify_main` reports. Neither view
contains private `ContextBlock` contents or model chain of thought.

### Expert Region Execution

`run_region_expert` reads only the selected region's private workspace view, invokes one configured model, validates
its structured `RegionReport`, and publishes it through the deterministic escalation policy:

    run_region_expert(
        task_id="parser-debug",
        region="debugging",
        task="Determine the next grounded parser debugging action.",
        model="modelbridge_anthropic/claude-sonnet-4-6",
        max_context_tokens=2000,
        max_cost_usd=0.05,
    )

The expert prompt fences workspace blocks as untrusted data and never requests chain of thought. Runtime validation
rejects evidence references not present in the private view, decision-changing memory without an evidence reference,
and long verbatim copies of private context. The caller receives only the validated report, escalation decision,
usage/cost telemetry, routing metadata, and context counts; raw model output and private blocks are omitted.

If the region has no private context, the runtime publishes `request_context` without calling a model. A conservative
single-job preflight skips the call when `max_cost_usd` cannot cover the configured model and output cap.

Expert context export has an independent three-state authorization gate. It never rewrites allowed context:

```json
{
  "context_export_policy": {
    "mode": "audit",
    "endpoint_trust": {
      "trusted_gateway": "trusted"
    },
    "source_sensitivity": {
      "public_docs": "public"
    },
    "default_sensitivity": "private"
  }
}
```

- `off` is the default and a true bypass: blocks are not classified and model prompts are unchanged.
- `audit` records `allow` or `would_deny` metadata but still sends the byte-identical prompt.
- `enforce` sends the original context when allowed or skips the model call before prompt construction.

Endpoint trust is `external`, `trusted`, or `local`. It can be configured in the policy map above or inline as
`endpoints.<id>.context_trust`; policy overrides inline metadata. Official/bare model routes default to `external`.
Block sensitivity is `public`, `project`, `private`, or `secret`. A block-level `metadata.sensitivity` wins, followed by
`source_sensitivity`, then `default_sensitivity`. Built-in source defaults classify Git as `project`, memory as `private`,
and unknown sources conservatively as `private`.

The fixed authorization matrix is deliberately small: external endpoints receive only public blocks, trusted endpoints
receive public/project/private blocks, and local endpoints may also receive secret blocks. Export telemetry contains only
mode, action, trust/sensitivity classes, and block counts; it never contains context text.

## Workflow Suggestions

`suggest_workflow` builds on `route_regions` and returns explicit next tool-call suggestions for the main assistant or
user to approve. It is still local and deterministic: it does not call models, run review/consult/planner tools, read
memory, or edit files.

```python
suggest_workflow(
    goal="Optimize a Unity ECS FlowField system and review the implementation plan",
    files={"Assets/Scripts/FlowFieldSystem.cs": "..."},
)
```

Example actions may include `plan_task`, `consult_problem`, `review_document`, or `review_code`. Every action includes
`requires_user_approval: true`, a short reason, suggested arguments, source regions, and trace metadata. This is the
safe bridge between region routing and a future Context Scheduler.

## Knowledge Base

Review quality depends heavily on project knowledge. Built-in adapter packages may ship seed cases, but the most useful
architecture decisions, historical bugs, and team conventions usually live in project-local knowledge files.

Recommended project-local location:

```text
<project-root>/.brain-region/knowledge/*.yaml
```

The legacy `.design-review/knowledge/` directory is still loaded first for compatibility. New `.brain-region/knowledge/`
cases load after it and can override legacy cases with the same `id`.

Example:

```yaml
- id: API-001
  title: "Keep breaking API changes behind a migration path"
  version: {service: ">=2.0"}
  triggers: ["breaking change", "API contract", "migration"]
  category: compatibility
  bad_pattern: "Change a public request or response shape without a versioned fallback or migration notes."
  recommended_pattern: "Add a compatible path, document the migration window, and test old and new clients."
  source: "ADR-014#api-versioning"
```

Tips:

- Write one concrete, reproducible gotcha per case.
- Put words that will appear in plans or code into `triggers`.
- Keep sensitive project knowledge local and ignored by git.
- Use `list_knowledge` to inspect the loaded framework and local cases.

## Installation

```bash
cd <path-to-brain-region-mcp>
uv sync --extra dev
```

Run the test suite:

```bash
uv run pytest tests/ -q
uv run --extra dev ruff check .
```

## MCP Setup

Register the stdio server in Codex, Claude Code, or another MCP client:

```jsonc
{
  "type": "stdio",
  "command": "uv",
  "args": [
    "run",
    "--directory",
    "<path-to-brain-region-mcp>",
    "brain-region-mcp"
  ],
  "env": {
    "UNITY_PROJECT_ROOT": "<path-to-project-root>",
    "BRAIN_REGION_CONFIG": "<path-to-brain-region-mcp>/brain_region_config.json"
  }
}
```

`UNITY_PROJECT_ROOT` is a historical project-root environment variable name. Point it at the project you want reviewed.
Keep API keys in `.env` or process environment variables. Do not commit `.env` or local `brain_region_config.json`.

## CLI

The `brain-region` CLI uses the same pipeline as the MCP server. The legacy `design-review` command is still available
as an alias during the rename.

```bash
uv run brain-region plan path/to/plan.md --output markdown
cat plan.md | uv run brain-region plan -
uv run brain-region plan --text "# Plan" --dimensions planner feasibility
uv run brain-region code src/a.py src/b.py --output sarif --output-file review.sarif
uv run brain-region doc docs/rfc.md --type rfc --output markdown
```

Common options:

- `--panel`: model list or endpoint shortcuts.
- `--dimensions`: reviewer dimensions.
- `--adapter`: `auto`, `generic`, or another installed domain adapter.
- `--retrieve-top-k`: number of knowledge cases to retrieve.
- `--effort`: reasoning/thinking effort where supported.
- `--max-cost-usd`: preflight budget cap.
- `--timeout`: per-model timeout.

## Configuration

Defaults are resolved in this order:

```text
builtin < global config < project config < env < explicit tool args
```

See:

- [Config precedence](docs/config_precedence.md)
- [Endpoint configuration](docs/endpoint_config.md)

Typical local config path:

```text
<path-to-brain-region-mcp>/brain_region_config.json
```

`brain_region_config.json` can hold defaults such as:

- `panel`
- `dimensions`
- `retrieve_top_k`
- `timeout`
- `normalizer_model`
- `effort`
- `max_cost_usd`
- `endpoints`
- `model_profiles`
- `privacy_policy`
- `context_modes`

## Custom Gateway Endpoints

Use `endpoints` for OpenAI-compatible or Anthropic-compatible gateways such as New API, one-api, OpenRouter-style
proxies, or internal model bridges. Use one endpoint per wire protocol.

```json
{
  "endpoints": {
    "modelbridge_openai": {
      "provider": "openai",
      "base_url": "https://www.modelbridge.cloud/v1",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": ["gpt-5.5", "gpt-5.4-mini"]
    },
    "modelbridge_anthropic": {
      "provider": "anthropic",
      "base_url": "https://www.modelbridge.cloud",
      "api_key_env": "MODEBRIDGE_API_KEY",
      "models": ["claude-haiku-4-5", "claude-opus-4-8"]
    }
  },
  "panel": ["modelbridge_openai/gpt-5.5", "modelbridge_anthropic/claude-opus-4-8"]
}
```

Panel shortcuts:

- `"endpoints"` expands every declared model under every endpoint.
- `"endpoint_id"` expands every model under one endpoint.
- `"endpoint_id/model"` runs one model through one endpoint.
- Native LiteLLM strings such as `"gpt-4o"` or `"deepseek/deepseek-chat"` bypass endpoint config and use provider env vars.

For example, `"claude-opus-4-8"` is a bare official-provider route and usually needs `ANTHROPIC_API_KEY`, while
`"modelbridge_anthropic/claude-opus-4-8"` uses the configured gateway and its `MODEBRIDGE_API_KEY`. Run
`list_model_routes` when you want to inspect the exact route before spending tokens.

Model profile metadata is optional and descriptive. It is shown in `list_model_routes`, `suggest_panel`, and tool
`routing` metadata:

```jsonc
{
  "model_profiles": {
    "modelbridge_openai/gpt-5.4-mini": {
      "activation_role": "sleep",
      "tier": "economy",
      "cost": "low",
      "latency": "fast",
      "tags": ["cheap", "fast"],
      "quality_score": 0.65,
      "cost_score": 0.9,
      "speed_score": 0.85
    },
    "modelbridge_anthropic/claude-opus-4-8": {
      "activation_role": "awake",
      "tier": "flagship",
      "cost": "high",
      "tags": ["deep_reasoning", "architecture"],
      "quality_score": 0.98,
      "cost_score": 0.2
    }
  }
}
```

`suggest_panel(strategy="cheap_fast" | "best_reasoning" | "balanced" | "sleep" | "awake" | "structured_output")`
ranks configured routes by this profile metadata and returns `selected_panel`. It does not call models or automatically
execute downstream tools.

## Cost And Effort Controls

Two optional controls are available:

- `max_cost_usd`: preflight cost cap for a review. Jobs are kept in panel order until the estimate would exceed the cap.
- `effort`: reasoning/thinking intensity for providers that support it. Unsupported providers ignore it.

The report includes estimated budget information and actual usage/cost where the provider returns it.

## Privacy Mode

By default, every model in the panel receives the review document. For sensitive plan reviews, `privacy_policy` can enable
a strict mode where a trusted model sees the full document, adversarial reviewers see a redacted summary, and the trusted
model later mediates evidence.

```json
{
  "privacy_policy": {
    "policy": "strict",
    "trusted": {"endpoint": "trusted_gateway", "model": "trusted-model", "label": "trusted"},
    "min_coverage": 0.5
  }
}
```

Strict privacy is most useful for plan review. Code review can lose too much semantic detail after redaction.

## Review Memory

Use `mark_finding` to record whether a finding was useful:

```text
mark_finding(finding_id="gpt-4o-3", decision="accepted", params_hash="...")
```

Valid decisions are `accepted`, `rejected`, and `partial`. Feedback is stored in the local SQLite review database and is
used to calibrate future confidence per `(model, dimension)`.

## Output

Reports include:

- `consensus`: findings all models agreed on.
- `majority`: findings supported by multiple models.
- `individual`: one-model findings.
- `failed_models`: isolated model failures.
- `budget`, `usage`, `risk`, and `context_compression` metadata.

SARIF output can be uploaded to GitHub Code Scanning or consumed by IDEs.

## Project Layout

```text
brainregion/
  server.py              # MCP server entry point
  cli.py                 # brain-region CLI
  core/                  # pipeline, stages, schemas, report models
  adapters/              # generic and optional domain adapters
  providers/             # LLM backends
  knowledge/             # retrieval providers
  privacy/               # privacy policies
  output/                # renderers
tests/                   # pytest coverage
docs/                    # focused docs
```

## Security Notes

- Do not commit `.env`, `.env.local`, API keys, generated databases, or local `brain_region_config.json` files.
- Legacy `design_review_config.json` files are still supported but should not be committed either.
- Prefer `api_key_env` over plaintext `api_key`.
- Generated review databases such as `brain_region_reviews.db` are local data and should not be used in tests. Legacy
  `design_reviews.db` files are still read when present.

## License

Apache-2.0
