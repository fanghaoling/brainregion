"""GitEvent:git 提交历史模型(Phase 6;极简,无治理——git 历史不可变)。

镜像 ExperienceEvent 的「纯数据」面,但无 v6 governance 字段(git 历史不退役/不过期)。
subject = 提交首行(git 惯例单行 → log 解析稳);files = 改动文件路径。未来加 body/parents =
frozen dataclass 加性默认,非 breaking(参考 ExperienceEvent 长 4 个治理字段的同样方式)。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GitEvent:
    """一条 git 提交(git-package 内部;经 GitProvider 转 ContextBlock 后才跨 ContextProvider 边界)。"""

    sha: str
    subject: str
    author: str
    date: str  # ISO 8601(%aI)
    files: tuple[str, ...] = ()  # 改动文件路径
