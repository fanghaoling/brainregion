# 真实仓库定向记忆评测

`worktree-memory-eval` 用真实仓库任务测量“聚焦专家”和“专家私有项目记忆”是否有价值。每个臂都在新的临时 Git worktree 中运行，并使用相同的主模型、任务、测试命令与预算。

三个臂分别是：

- `main_only`：主模型独立修改并验证。
- `expert_without_memory`：专家只读取任务明确列出的仓库文件，返回经过校验的 RegionReport。
- `expert_with_scoped_memory`：同一个专家额外获得与自身 region 精确匹配的记忆。

记忆不会直接注入主模型。主模型只能收到公开 RegionReport，并且必须使用仓库文件和测试重新验证。评测报告不保存源码、记忆正文、模型响应、diff、工具结果或推理内容。

## 任务规格

```json
{
  "id": "parser-regression",
  "goal": "修复 parser 回归并让 tests/test_parser.py 通过。",
  "repo_path": "D:/Projects/example",
  "base_ref": "tasks/parser-regression",
  "test_args": ["tests/test_parser.py", "-q"],
  "bootstrap_commands": [],
  "expert_context_paths": ["src/parser.py", "tests/test_parser.py"],
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

## 运行

```powershell
brain-region sandbox worktree-memory-eval `
  --task-spec .brain-region/tasks/parser-regression.json `
  --main-brain buzz_anthropic/claude-haiku-4-5-20251001 `
  --expert-model buzz_anthropic/claude-haiku-4-5-20251001
```

该命令会真实调用模型并产生费用。首次应只跑一个任务和 `--repeats 1`，在多个任务和重复实验方向一致前只把结果视为 pilot。主要比较是 `expert_with_scoped_memory - expert_without_memory`；`main_only` 只用于分离“专家存在本身”的价值。
