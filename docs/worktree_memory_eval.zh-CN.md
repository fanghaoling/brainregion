# 真实仓库定向记忆评测

`worktree-memory-eval` 用真实仓库任务测量“聚焦专家”和“专家私有项目记忆”是否有价值。每个臂都在新的临时 Git worktree 中运行，并使用相同的主模型、任务、测试命令与预算。

三个臂分别是：

- `main_only`：主模型独立修改并验证。
- `expert_without_memory`：专家只读取任务明确列出的仓库文件，返回经过校验的 RegionReport。
- `expert_with_scoped_memory`：同一个专家额外获得与自身 region 精确匹配的记忆。

记忆不会直接注入主模型。主模型只能收到公开 RegionReport，并且必须使用仓库文件和测试重新验证。评测报告不保存源码、记忆正文、模型响应、diff、工具结果或推理内容。

报告会保留无内容的主脑诊断指标：操作类型计数、错误类型计数、唯一目标数，以及触及输出 token 上限的调用次数。这样可以区分“反复读搜但不修改”和“连续解析失败”，同时不保存路径、工具输出或模型原文。

## 任务规格

```json
{
  "id": "parser-regression",
  "goal": "修复 parser 回归并让 tests/test_parser.py 通过。",
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
      "summary": "历史经验表明，带包装文本的响应需要提取第一个完整 JSON 对象。"
    }
  ]
}
```

`expert_context_paths` 是文件白名单。路径逃逸、文件不存在、重复路径、单文件或总上下文超限都会在专家调用前失败。记忆只通过精确 region 或 `shared` 确定性选择，不让模型先判断召回范围。
条目也可以用包含 `path`、`start_line`、`end_line` 的对象表示闭区间，只装载大文件中与任务相关的代码，避免无关部分占用专家上下文。

`protected_paths` 为必填项。harness 会在 bootstrap 后记录这些文件的摘要；主模型只要修改或删除其中任何文件，即使 pytest 转绿也会判定该臂未解决。历史回放因此不能靠削弱回归测试过关。

## 运行

```powershell
brain-region sandbox worktree-memory-eval `
  --task-spec .brain-region/tasks/parser-regression.json `
  --main-brain buzz_anthropic/claude-haiku-4-5-20251001 `
  --expert-model buzz_anthropic/claude-haiku-4-5-20251001
```

该命令会真实调用模型并产生费用。首次应只跑一个任务和 `--repeats 1`，在多个任务和重复实验方向一致前只把结果视为 pilot。主要比较是 `expert_with_scoped_memory - expert_without_memory`；`main_only` 只用于分离“专家存在本身”的价值。
