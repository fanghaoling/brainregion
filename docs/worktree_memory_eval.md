# Worktree Scoped-Memory Evaluation

`worktree-memory-eval` measures whether a focused expert and expert-private project memory help on a real repository task.
Every arm runs in a fresh temporary Git worktree and uses the same main model, task, test command, and budgets.

The three arms are:

- `main_only`: the main model edits and verifies without an expert.
- `expert_without_memory`: an expert reads only the explicitly listed repository files and returns a validated RegionReport.
- `expert_with_scoped_memory`: the same expert also receives memory records matching its region.

Memory is never injected directly into the main model. The main model receives only the public RegionReport and must
verify it against repository files and tests. Reports do not persist source, memory, model responses, diffs, tool results,
or reasoning content.

Reports retain content-free main-brain diagnostics: operation counts, error-kind counts, unique-target count, and how many
responses reached the configured output-token cap. These distinguish navigation/inspection stalls from parser failures
without storing paths, tool output, or model text.

## Task Spec

```json
{
  "id": "parser-regression",
  "goal": "Fix the parser regression and make tests/test_parser.py pass.",
  "repo_path": "D:/Projects/example",
  "base_ref": "tasks/parser-regression",
  "test_args": ["tests/test_parser.py", "-q"],
  "bootstrap_commands": [],
  "expert_context_paths": [
    {"path": "src/parser.py", "start_line": 40, "end_line": 140},
    "tests/test_parser.py"
  ],
  "protected_paths": ["tests/test_parser.py"],
  "seed_memory": [
    {
      "id": "parser-wrapper-lesson",
      "region": "debugging",
      "status": "active",
      "summary": "Previous wrapped responses required extracting the first complete JSON object."
    }
  ]
}
```

`expert_context_paths` is an allowlist. Paths outside the worktree, missing files, duplicate paths, oversized files, and
oversized aggregate context fail before an expert call. Memory is selected deterministically by exact region or `shared`.
An entry may be an object with inclusive `start_line` and `end_line` bounds so a large source file can be staged without
spending the expert's context budget on unrelated sections.

`protected_paths` is required. The harness snapshots these files after bootstrap and marks a run unsolved when the main
model changes or deletes any of them, even if pytest turns green. Historical replay tasks therefore cannot pass by
weakening their regression tests.

## Run

```powershell
brain-region sandbox worktree-memory-eval `
  --task-spec .brain-region/tasks/parser-regression.json `
  --main-brain buzz_anthropic/claude-haiku-4-5-20251001 `
  --expert-model buzz_anthropic/claude-haiku-4-5-20251001
```

This command makes real model calls and can incur cost. Start with one task and `--repeats 1`; interpret results as a
pilot until multiple tasks and repeats agree. Primary comparison is `expert_with_scoped_memory` minus
`expert_without_memory`; `main_only` separately measures expert-presence value.
