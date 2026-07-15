"""Git 历史存储 —— git log 子进程包装 + 解析/搜索纯函数(Phase 6;镜像 memory/store.py)。

``GitRunner`` 可注入(测试 / 未来换 pygit2);``_real_runner`` 走 subprocess。降级规范镜像 memory store:
git 缺(``FileNotFoundError``)/ 非 repo 或空仓库(returncode≠0)→ ``([], meta)`` **不抛**。
``search_commits`` 是纯函数(eval 用,不跑子进程 = 防伪)。

解析用 ASCII RS/US 分隔:git format placeholder ``%x1e``/``%x1f`` 让 git 输出真实 0x1e/0x1e 字节;
本模块用真实字节 ``\\x1e``/``\\x1f`` 切分。subject 单行(git 惯例)→ 字段切分稳。
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Callable

from ..core.regions.loader import _contains, _normalize
from .base import GitEvent

logger = logging.getLogger("brainregion.git.store")

# 切分用的真实字节(Record/Unit Separator);git 经 %x1e/%x1f placeholder 输出。
_RS = "\x1e"
_US = "\x1f"
# git format placeholders(git 把 %x1e 渲染成真实 0x1e 字节)。subject(%s)单行故无换行歧义。
_FMT = "%x1e%H%x1f%s%x1f%an%x1f%aI"
_GIT_TIMEOUT_SECONDS = 10.0

# query tokenization 停用词(去常见噪声;镜像 router.py ManifestRouter._terms 的极小集)。
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "via",
        "over",
        "to",
        "of",
        "in",
        "on",
        "a",
        "an",
        "is",
        "are",
        "be",
        "by",
        "or",
        "as",
        "at",
        "it",
        "its",
        "from",
        "into",
        "use",
        "used",
        "uses",
        "using",
        "can",
        "will",
        "not",
        "but",
        "which",
        "when",
    }
)

# runner 契约:(args, cwd) -> (returncode, stdout, stderr)。注入用;默认 _real_runner 走 subprocess。
GitRunner = Callable[[list[str], str], tuple[int, str, str]]


def _real_runner(args: list[str], cwd: str) -> tuple[int, str, str]:
    """默认 runner:走 subprocess.run(git)。git 不在 PATH → raise FileNotFoundError(caller 捕获)。

    强制 UTF-8 解码(git 输出恒 UTF-8);Windows 默认 gbk 会在中文 commit message 上崩。
    """
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_GIT_TIMEOUT_SECONDS,
    )
    return proc.returncode, proc.stdout, proc.stderr


@dataclass
class GitStore:
    """git 历史源(cwd/max_n/runner 可注入)。``list_commits`` 跑一次 git log → 解析为 GitEvent 列表。"""

    cwd: str = "."
    max_n: int = 200
    runner: GitRunner | None = None  # None → _real_runner

    def _log_args(self) -> list[str]:
        return [
            "log",
            "--no-merges",
            "-n",
            str(int(self.max_n)),
            "--name-only",
            f"--format={_FMT}",
        ]

    def list_commits(self) -> tuple[list[GitEvent], dict]:
        runner = self.runner or _real_runner
        try:
            rc, stdout, stderr = runner(self._log_args(), self.cwd)
        except subprocess.TimeoutExpired as e:
            logger.warning("GitStore.list_commits: git log timed out: %s", e)
            return [], {
                "git_available": True,
                "commits_found": 0,
                "timed_out": True,
                "error": f"git log timed out after {e.timeout}s",
            }
        except (FileNotFoundError, OSError) as e:
            logger.warning("GitStore.list_commits: git 不可用: %s", e)
            return [], {"git_available": False, "commits_found": 0, "error": str(e)}
        if rc != 0:
            # 非 repo / 空仓库 / 其他 git 错 → git 在,只是无历史(降级不抛)。
            logger.warning("GitStore.list_commits: git log rc=%s stderr=%s", rc, (stderr or "").strip()[:200])
            return [], {"git_available": True, "commits_found": 0, "error": (stderr or "").strip()[:200]}
        events = _parse_log(stdout)
        return events, {
            "git_available": True,
            "commits_found": len(events),
            "parsed": len(events),
            "truncated": len(events) >= int(self.max_n),
        }


def _parse_log(stdout: str) -> list[GitEvent]:
    """解析 ``git log --format=\\x1e...\\x1f... --name-only`` 输出 → GitEvent 列表(subject 单行保稳)。"""
    events: list[GitEvent] = []
    for record in stdout.split(_RS):
        record = record.lstrip("\n")
        if not record:
            continue
        meta_part, _, files_part = record.partition("\n")
        fields = meta_part.split(_US)
        if len(fields) < 4:
            continue
        sha, subject, author, date = fields[0], fields[1], fields[2], fields[3]
        files = tuple(f for f in files_part.split("\n") if f.strip())
        events.append(GitEvent(sha=sha, subject=subject, author=author, date=date, files=files))
    return events


def _query_terms(text: str) -> list[str]:
    """tokenize query(len>=3 + 去停用词;镜像 ManifestRouter._terms)。"""
    out: list[str] = []
    for tok in _normalize(text).split(" "):
        if len(tok) >= 3 and tok not in _STOPWORDS and tok not in out:
            out.append(tok)
    return out


def _event_text(event: GitEvent) -> str:
    return _normalize(event.subject + " " + " ".join(event.files))


def search_commits(events: list[GitEvent], text: str, top_k: int = 5) -> list[GitEvent]:
    """纯函数召回(query 词命中 event 的 subject+files;按命中数降序、原序 tie-break;top_k)。空 query → []。"""
    terms = _query_terms(text or "")
    if not terms:
        return []
    hits: list[tuple[int, int, GitEvent]] = []
    for i, e in enumerate(events):
        et = _event_text(e)
        score = sum(1 for t in terms if _contains(et, t))
        if score > 0:
            hits.append((score, i, e))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return [e for _, _, e in hits[: max(0, int(top_k))]]
