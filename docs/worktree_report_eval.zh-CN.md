# 真实工作树 RegionReport 利用率评测

`worktree-report-eval` 用于隔离“主脑能否把专家报告转化为仓库操作”。每个 repeat 只调用一次 scoped 专家：专家在私有工作区读取白名单源码和记忆，生成一份经过校验的 RegionReport；随后同一份报告语义进入三个全新 worktree：

- `no_report`：不注入建议。
- `full_report`：注入全部公共 RegionReport 字段。
- `decision_card`：只注入有界的行动字段，包括 assignment、region、摘要、影响、建议动作、不确定性、证据和风险。

专家每个 repeat 只调用一次，而不是每个臂各调用一次，避免把专家采样差异误判为报告格式影响。主脑始终不会直接收到专家私有源码快照或记忆。

任务规格与 `worktree-memory-eval` 相同，必须包含 `expert_context_paths`、`protected_paths` 和能匹配专家 region 的 `seed_memory`。

```powershell
brain-region --config brain_region_config.json --env-file .env `
  sandbox worktree-report-eval `
  --task-spec .brain-region/tasks/historical-consult-endpoint.json `
  --main-brain buzz_anthropic/claude-haiku-4-5-20251001 `
  --expert-model buzz_anthropic/claude-haiku-4-5-20251001 `
  --repeats 1
```

评测报告只保留无内容的交付长度、动作、解析错误、输出上限饱和、工作区修改、验证、Token、成本和解决状态；不保存源码、记忆、RegionReport 正文、diff、工具结果或推理。

主要对比是 `decision_card - full_report`，`no_report` 只作为“存在报告本身是否有价值”的下界。单次结果只能视为协议 pilot，不能证明专家或记忆具有一般价值。
