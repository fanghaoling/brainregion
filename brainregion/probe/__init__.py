"""模型指纹探针:检测端点是否被偷换模型 / 降智。

- fingerprint: usage(tokenizer 计数)指纹 + behavior(随机数分布 JSD)指纹的采集与对比。
- storage: 基线与运行历史的 SQLite 存储(对齐 eval/store 模式)。

方法出处与阈值依据见 docs/model_fingerprint.zh-CN.md。
"""
