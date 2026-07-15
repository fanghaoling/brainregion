"""Phase 6:GitProvider —— 第二个 ContextProvider 测试。

覆盖:git log 解析(\x1e/\x1f + name-only)、search_commits(纯函数召回)、GitProvider 双构造
(from_repo 惰性 / from_events 防伪)、降级(git 缺 / 非 repo 不抛)、GitStore 命令正确、
抽象闭环(ProviderResolver 零改动直通 git)、list_skills MCP、真实 git 冒烟。
"""

from __future__ import annotations

import subprocess

import pytest

from brainregion.core.context import ContextBlock, ContextQuery, ProviderRegistry
from brainregion.core.skills import (
    SkillManifest,
    UnsupportedSkillKind,
    load_skill,
    resolve_skill_body,
    setup_resolvers,
)
from brainregion.core.skills.loader import SKILLS_DIR
from brainregion.git import GitEvent, GitProvider
from brainregion.git.store import GitStore, _parse_log, search_commits


# ── helpers ───────────────────────────────────────────────────────────────────


def _log_stdout(commits: list[dict]) -> str:
    """构造匹配真实 ``git log --format=\\x1e... --name-only`` 的 stdout(subject 单行)。"""
    chunks: list[str] = []
    for c in commits:
        meta = f"\x1e{c['sha']}\x1f{c['subject']}\x1f{c['author']}\x1f{c['date']}"
        files_block = "\n".join(c.get("files", []))
        chunks.append(meta + ("\n" + files_block if files_block else ""))
    return "\n".join(chunks) + ("\n" if chunks else "")


def _ev(sha: str, subject: str, files: tuple[str, ...] = (), author: str = "a", date: str = "2026-07-01") -> GitEvent:
    return GitEvent(sha=sha, subject=subject, author=author, date=date, files=files)


# ── 解析 ───────────────────────────────────────────────────────────────────────


def test_parse_log_two_commits_with_files():
    out = _log_stdout(
        [
            {
                "sha": "aaa111",
                "subject": "fix wake_gate routing",
                "author": "alice",
                "date": "2026-07-01T10:00:00+08:00",
                "files": ["brainregion/core/wake/gate.py", "brainregion/core/regions/router.py"],
            },
            {
                "sha": "bbb222",
                "subject": "docs: roadmap update",
                "author": "bob",
                "date": "2026-07-02T11:00:00+08:00",
                "files": ["docs/roadmap.md"],
            },
        ]
    )
    events = _parse_log(out)
    assert len(events) == 2
    e0, e1 = events
    assert e0.sha == "aaa111" and e0.subject == "fix wake_gate routing" and e0.author == "alice"
    assert e0.files == ("brainregion/core/wake/gate.py", "brainregion/core/regions/router.py")
    assert e1.sha == "bbb222" and e1.files == ("docs/roadmap.md",)


def test_parse_log_empty_and_no_files():
    assert _parse_log("") == []  # 空输出 → []
    e = _parse_log("\x1esha\x1fsubj\x1fauth\x1fdate\n")[0]  # 无文件
    assert e.files == ()


def test_parse_log_skips_malformed_record():
    # 字段 < 4 的 record 跳过(不崩)。
    out = "\x1egarbage\x1fonlytwo\n\x1egood\x1fs\x1fa\x1fd\nf.py\n"
    events = _parse_log(out)
    assert len(events) == 1 and events[0].sha == "good"


# ── search_commits(纯函数召回)──────────────────────────────────────────────────


def test_search_commits_matches_subject_and_files_orders_by_score():
    # query "wake_gate routing" → terms [wake, gate, routing](_ normalize 成空格)。
    events = [
        _ev("1", "fix wake_gate routing bug", ("gate.py",)),  # 3 命中(wake, gate, routing)
        _ev("2", "update gate config", ()),  # 1 命中(gate)
        _ev("3", "unrelated docs", ("readme.md",)),  # 0 命中
    ]
    hits = search_commits(events, "wake_gate routing", top_k=5)
    assert [h.sha for h in hits] == ["1", "2"]  # doc 不命中;score 降序
    assert hits[0].sha == "1"  # 3 命中 > 1 命中
    assert search_commits(events, "wake_gate routing", top_k=1) == [events[0]]  # top_k 截断


def test_search_commits_matches_file_paths():
    events = [_ev("1", "misc", ("brainregion/core/wake/gate.py",))]
    assert [h.sha for h in search_commits(events, "wake gate", top_k=3)] == ["1"]


def test_search_commits_empty_query_and_topk():
    events = [_ev("1", "wake_gate")]
    assert search_commits(events, "", top_k=5) == []  # 空 query → []
    assert search_commits(events, "wake_gate", top_k=0) == []  # top_k=0 → []
    # 短词 + 停用词被过滤(len<3 / 停用词表)
    assert search_commits(events, "the a of", top_k=5) == []


# ── GitProvider.from_events(防伪;镜像 MemoryProvider.from_records)──────────────


def test_provider_from_events_retrieve_blocks_and_meta():
    events = [_ev("aaa1111", "fix wake_gate routing", ("gate.py",))]
    p = GitProvider.from_events(events)
    rr = p.retrieve(ContextQuery(text="wake_gate routing", top_k=3))
    assert rr.provider == "git" and len(rr.blocks) == 1
    b = rr.blocks[0]
    assert isinstance(b, ContextBlock)
    assert b.source == "git" and b.framing == "data"  # framing=data(防注入)
    assert b.title == "aaa1111 fix wake_gate routing"
    assert b.metadata["sha"] == "aaa1111" and b.metadata["files"] == ["gate.py"]
    assert rr.meta["source"] == "injected"
    assert rr.meta["candidates_before_top_k"] == 1 and rr.meta["returned"] == 1


def test_provider_from_events_empty_returns_empty_blocks():
    rr = GitProvider.from_events([]).retrieve(ContextQuery(text="anything", top_k=3))
    assert rr.blocks == [] and rr.meta["candidates_before_top_k"] == 0


# ── 降级(git 缺 / 非 repo 不抛;镜像 store accessor try/except)──────────────────


def test_store_git_missing_degrades_to_empty():
    def _missing_runner(args, cwd):
        raise FileNotFoundError("git not on PATH")

    events, meta = GitStore(runner=_missing_runner).list_commits()
    assert events == [] and meta["git_available"] is False  # git 缺 → 空 + flag


def test_store_non_repo_degrades_to_empty_git_available():
    def _err_runner(args, cwd):
        return 128, "", "fatal: not a git repository (or any of the parent directories): .git"

    events, meta = GitStore(runner=_err_runner).list_commits()
    assert events == [] and meta["git_available"] is True and meta["commits_found"] == 0


def test_store_timeout_degrades_with_observable_reason():
    def _timeout_runner(args, cwd):
        raise subprocess.TimeoutExpired(["git", *args], timeout=10.0)

    events, meta = GitStore(runner=_timeout_runner).list_commits()

    assert events == []
    assert meta["git_available"] is True
    assert meta["timed_out"] is True
    assert "timed out" in meta["error"]


def test_provider_store_degradation_no_crash():
    def _missing_runner(args, cwd):
        raise FileNotFoundError("git")

    rr = GitProvider(store=GitStore(runner=_missing_runner)).retrieve(ContextQuery(text="x", top_k=3))
    assert rr.blocks == [] and rr.meta["git_available"] is False  # 降级不抛


# ── GitStore 命令正确(runner 收到合法 git log 参数)──────────────────────────────


def test_store_log_args_shape():
    seen: dict = {}

    def _spy_runner(args, cwd):
        seen["args"] = args
        seen["cwd"] = cwd
        return 0, "", ""

    GitStore(cwd="/tmp/repo", max_n=50, runner=_spy_runner).list_commits()
    assert seen["cwd"] == "/tmp/repo"
    a = seen["args"]
    assert a[0] == "log" and "--no-merges" in a and "-n" in a  # log / --no-merges / -n
    assert a[a.index("-n") + 1] == "50"  # max_n 紧跟 -n
    assert "--name-only" in a
    assert any(str(x).startswith("--format=") for x in a)  # --format placeholder


# ── 抽象闭环(ProviderResolver 零改动直通 git)──────────────────────────────────


def test_resolve_skill_body_git_via_provider_resolver():
    """git-recall(kind=provider, ref=git) 经现 ProviderResolver 直通 → RetrieveResult provider='git'。"""
    reg = ProviderRegistry()
    reg.register("git", GitProvider.from_events([_ev("aaa1111", "fix wake_gate routing", ("gate.py",))]))
    resolvers = setup_resolvers(provider_registry=reg)
    m = SkillManifest(id="git-recall", name="Git History Recall", region="review", kind="provider", ref="git")
    rr = resolve_skill_body(m, ContextQuery(text="wake_gate routing", top_k=3), resolvers=resolvers)
    assert rr.provider == "git" and rr.blocks  # 零 resolver 改动


def test_git_recall_yaml_loads_and_resolves():
    """真实 git-recall.yaml 经 loader 加载(过 review region + git provider 校验)→ 可 resolve。"""
    reg = ProviderRegistry()
    reg.register("git", GitProvider.from_events([_ev("s", "subj")]))
    m = load_skill(
        "git-recall",
        SKILLS_DIR,
        region_exists=lambda r: r == "review",
        provider_exists=lambda n: n == "git",
    )
    assert m.kind == "provider" and m.ref == "git" and m.region == "review"
    resolvers = setup_resolvers(provider_registry=reg)
    rr = resolve_skill_body(m, ContextQuery(text="subj", top_k=3), resolvers=resolvers)
    assert rr.provider == "git"


def test_consultant_seed_still_unresolvable_after_git_added():
    """回归:consultant seed 仍 UnsupportedSkillKind(git 加入不破 kind 分发)。"""
    resolvers = setup_resolvers(provider_registry=ProviderRegistry())
    m = SkillManifest(id="debugger", name="D", region="debugging", kind="consultant", ref="debugger")
    with pytest.raises(UnsupportedSkillKind):
        resolve_skill_body(m, ContextQuery(text="x"), resolvers=resolvers)


# ── list_skills MCP(含 git-recall;sanitized)───────────────────────────────────


def test_list_skills_mcp_includes_git_recall():
    from brainregion.server import list_skills as list_skills_mcp

    out = list_skills_mcp()
    ids = {s["id"] for s in out["skills"]}
    assert "git-recall" in ids
    g = next(s for s in out["skills"] if s["id"] == "git-recall")
    assert g["region"] == "review" and g["kind"] == "provider"
    assert "ref" not in g  # MCP 输出不泄 ref


def test_bootstrap_registers_git_provider():
    """drift:_skill_registry bootstrap 注册了 GitProvider(ProviderRegistry 有 'git')。"""
    from brainregion.core.context import default_provider_registry
    from brainregion.server import _skill_registry

    _skill_registry()  # 触发 bootstrap
    assert default_provider_registry.has("git")


# ── 真实 git 冒烟(本 repo;catch format 回归)───────────────────────────────────


def test_real_git_log_parses_in_repo():
    """真实 git log 冒烟:本 repo 有提交 → 解析非空 GitEvent,字段非空。非 repo/git 缺 → skip。"""
    events, meta = GitStore(cwd=".").list_commits()  # real _real_runner
    if not meta.get("git_available"):
        pytest.skip("git unavailable in this environment")
    if meta.get("commits_found", 0) == 0:
        pytest.skip("no commits in cwd")
    assert meta["git_available"] is True and meta["commits_found"] > 0
    e = events[0]
    assert e.sha and e.subject and e.date  # 字段非空(格式回归守卫)
    assert isinstance(e.files, tuple)


def test_real_git_provider_retrieve_in_repo():
    """真实 from_repo().retrieve 在本 repo 跑通(query 命中近期 router/wake 提交)。"""
    rr = GitProvider.from_repo().retrieve(ContextQuery(text="router wake_gate", top_k=3))
    assert rr.provider == "git"
    assert rr.meta.get("git_available") is True
    # 不强断具体提交(历史会变);只断 shape + framing 一致。
    for b in rr.blocks:
        assert b.source == "git" and b.framing == "data"
