# Worktree RegionReport Utilization Evaluation

`worktree-report-eval` isolates whether a main model can turn a validated expert
report into repository actions. For every repeat, one scoped expert reads the
allowlisted source and memory in a private workspace and produces one validated
RegionReport. The same report semantics feed three fresh worktree arms:

- `no_report`: no advisory context.
- `full_report`: all public RegionReport fields.
- `decision_card`: bounded action-facing fields only: assignment, region,
  summary, implication, recommended action, uncertainty, evidence, and risk.

The expert is called once per repeat, not once per arm. This prevents report
sampling differences from being mistaken for delivery-format effects. The main
model never receives private source snapshots or memory directly.

The task spec is the same as `worktree-memory-eval` and requires
`expert_context_paths`, `protected_paths`, and matching `seed_memory`.

```powershell
brain-region --config brain_region_config.json --env-file .env `
  sandbox worktree-report-eval `
  --task-spec .brain-region/tasks/historical-consult-endpoint.json `
  --main-brain buzz_anthropic/claude-haiku-4-5-20251001 `
  --expert-model buzz_anthropic/claude-haiku-4-5-20251001 `
  --repeats 1
```

Reports retain only content-free delivery length, actions, parse-error counts,
output-cap saturation, workspace effects, verification, tokens, cost, and solve
status. They do not persist source, memory, RegionReport text, diffs, tool
results, or reasoning.

The primary contrast is `decision_card - full_report`. `no_report` is a lower
bound for separating report-presence value. Treat a one-repeat result as a
protocol pilot, not as evidence of general expert or memory value.
