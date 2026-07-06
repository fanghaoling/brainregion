from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from brainregion.workspace.files import (
    MAX_READ_FILE_BYTES,
    apply_text_patch,
    inspect_file,
    list_allowed_roots,
    read_text,
    search_text,
)


@pytest.fixture()
def workspace_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_REGION_WORKSPACE_ROOTS", str(tmp_path))
    monkeypatch.setattr("brainregion.workspace.files.emit_event", lambda *a, **k: {"ok": True})
    return tmp_path


def test_list_allowed_roots_uses_workspace_env(workspace_root):
    result = list_allowed_roots()

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["roots"][0]["path"] == str(workspace_root.resolve())
    assert result["roots"][0]["source"] == "env:BRAIN_REGION_WORKSPACE_ROOTS"


def test_inspect_file_returns_metadata_without_contents(workspace_root):
    p = workspace_root / "notes.md"
    p.write_text("# Title\n中文\n", encoding="utf-8")

    result = inspect_file("notes.md")

    assert result["ok"] is True
    assert result["relative_path"] == "notes.md"
    assert result["is_file"] is True
    assert result["is_text"] is True
    assert result["encoding"] == "utf-8"
    assert result["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert "text" not in result


def test_read_text_keeps_utf8_and_line_window(workspace_root):
    (workspace_root / "src").mkdir()
    p = workspace_root / "src" / "demo.py"
    p.write_text("line 1\n中文第二行\nline 3\n", encoding="utf-8", newline="\n")

    result = read_text("src/demo.py", start_line=2, end_line=3)

    assert result["ok"] is True
    assert result["relative_path"] == str(Path("src") / "demo.py")
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["total_lines"] == 3
    assert result["text"] == "中文第二行\nline 3\n"


def test_read_text_caps_output_bytes(workspace_root):
    (workspace_root / "long.txt").write_text("abcdef", encoding="utf-8")

    result = read_text("long.txt", max_bytes=3)

    assert result["text"] == "abc"
    assert result["truncated"] is True


def test_read_text_rejects_path_outside_allowed_root(workspace_root, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside allowed workspace roots"):
        read_text(str(outside))


def test_read_text_rejects_sensitive_files(workspace_root):
    (workspace_root / ".env").write_text("API_KEY=secret", encoding="utf-8")

    with pytest.raises(PermissionError, match="sensitive env file"):
        read_text(".env")


def test_read_text_rejects_binary_files(workspace_root):
    (workspace_root / "image.bin").write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="binary files"):
        read_text("image.bin")


def test_read_text_rejects_oversized_files(workspace_root):
    (workspace_root / "huge.txt").write_bytes(b"a" * (MAX_READ_FILE_BYTES + 1))

    with pytest.raises(ValueError, match="too large"):
        read_text("huge.txt")


def test_apply_text_patch_defaults_to_dry_run(workspace_root):
    p = workspace_root / "edit.txt"
    p.write_text("alpha\nbeta\n", encoding="utf-8", newline="\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    result = apply_text_patch(
        "edit.txt",
        expected_sha256=sha,
        replacements=[{"old_text": "beta", "new_text": "gamma"}],
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["changed"] is True
    assert result["old_sha256"] == sha
    assert result["new_sha256"] != sha
    assert "-beta" in result["diff"]
    assert "+gamma" in result["diff"]
    assert p.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_apply_text_patch_writes_when_dry_run_false(workspace_root):
    p = workspace_root / "edit.txt"
    p.write_text("alpha\nbeta\n", encoding="utf-8", newline="\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    result = apply_text_patch(
        "edit.txt",
        expected_sha256=sha,
        replacements=[{"old_text": "beta", "new_text": "gamma"}],
        dry_run=False,
    )

    assert result["dry_run"] is False
    assert p.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == result["new_sha256"]


def test_apply_text_patch_rejects_hash_mismatch(workspace_root):
    p = workspace_root / "edit.txt"
    p.write_text("alpha\nbeta\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="expected_sha256"):
        apply_text_patch(
            "edit.txt",
            expected_sha256="0" * 64,
            replacements=[{"old_text": "beta", "new_text": "gamma"}],
            dry_run=False,
        )

    assert p.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_apply_text_patch_rejects_ambiguous_or_missing_old_text(workspace_root):
    p = workspace_root / "edit.txt"
    p.write_text("same\nsame\n", encoding="utf-8", newline="\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="ambiguous"):
        apply_text_patch("edit.txt", expected_sha256=sha, replacements=[{"old_text": "same", "new_text": "new"}])
    with pytest.raises(ValueError, match="not found"):
        apply_text_patch("edit.txt", expected_sha256=sha, replacements=[{"old_text": "missing", "new_text": "new"}])


def test_apply_text_patch_rejects_sensitive_files(workspace_root):
    p = workspace_root / ".env"
    p.write_text("API_KEY=secret\n", encoding="utf-8")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    with pytest.raises(PermissionError, match="sensitive env file"):
        apply_text_patch(".env", expected_sha256=sha, replacements=[{"old_text": "secret", "new_text": "redacted"}])


def test_apply_text_patch_event_redacts_patch_content(workspace_root, monkeypatch):
    emitted = []
    monkeypatch.setattr("brainregion.workspace.files.emit_event", lambda event_type, **fields: emitted.append((event_type, fields)))
    p = workspace_root / "edit.txt"
    p.write_text("secret-old\n", encoding="utf-8", newline="\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    result = apply_text_patch(
        "edit.txt",
        expected_sha256=sha,
        replacements=[{"old_text": "secret-old", "new_text": "secret-new"}],
        dry_run=False,
    )

    assert result["changed"] is True
    assert emitted[0][0] == "workspace.file_patch_applied"
    payload_text = str(emitted[0][1]["payload"])
    assert "secret-old" not in payload_text
    assert "secret-new" not in payload_text
    assert "diff_sha256" in emitted[0][1]["payload"]


def test_search_text_finds_utf8_with_globs_and_context(workspace_root):
    (workspace_root / "src").mkdir()
    (workspace_root / "docs").mkdir()
    (workspace_root / "src" / "main.py").write_text(
        "before\n目标函数()\nafter\n", encoding="utf-8", newline="\n"
    )
    (workspace_root / "docs" / "note.md").write_text("目标函数 in docs\n", encoding="utf-8", newline="\n")

    result = search_text("目标函数", include_globs=["*.py"], context_lines=1)

    assert result["ok"] is True
    assert result["count"] == 1
    match = result["matches"][0]
    assert match["relative_path"] == "src/main.py"
    assert match["line"] == 2
    assert match["text"] == "目标函数()"
    assert [item["line"] for item in match["context"]] == [1, 2, 3]


def test_search_text_respects_exclude_and_skips_denied_paths(workspace_root):
    (workspace_root / "src").mkdir()
    (workspace_root / "node_modules").mkdir()
    (workspace_root / "src" / "keep.txt").write_text("needle\n", encoding="utf-8")
    (workspace_root / "src" / "skip.log").write_text("needle\n", encoding="utf-8")
    (workspace_root / "node_modules" / "package.txt").write_text("needle\n", encoding="utf-8")
    (workspace_root / ".env").write_text("needle=secret\n", encoding="utf-8")

    result = search_text("needle", exclude_globs=["*.log"])

    assert [m["relative_path"] for m in result["matches"]] == ["src/keep.txt"]
    assert result["skipped"]["excluded"] >= 1
    assert result["skipped"]["denied"] >= 2


def test_search_text_regex_case_sensitive_and_limit(workspace_root):
    (workspace_root / "a.txt").write_text("Needle\nneedle\n", encoding="utf-8", newline="\n")
    (workspace_root / "b.txt").write_text("needle\n", encoding="utf-8", newline="\n")

    result = search_text(r"^needle$", regex=True, case_sensitive=True, max_results=1)

    assert result["count"] == 1
    assert result["truncated"] is True
    assert result["matches"][0]["text"] == "needle"


def test_search_text_rejects_invalid_regex(workspace_root):
    (workspace_root / "a.txt").write_text("x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid regex"):
        search_text("[", regex=True)


def test_search_text_event_redacts_query(workspace_root, monkeypatch):
    emitted = []
    monkeypatch.setattr("brainregion.workspace.files.emit_event", lambda event_type, **fields: emitted.append((event_type, fields)))
    (workspace_root / "a.txt").write_text("api_key=sk-secret needle\n", encoding="utf-8")

    result = search_text("api_key=sk-secret needle")

    assert result["count"] == 1
    assert emitted[0][0] == "workspace.text_search"
    payload = emitted[0][1]["payload"]
    assert "query_sha256" in payload
    assert "api_key" not in str(payload)


def test_server_mcp_tools_delegate_to_workspace(workspace_root):
    from brainregion import server

    (workspace_root / "tool.txt").write_text("hello\n", encoding="utf-8", newline="\n")

    roots = server.list_allowed_roots()
    inspected = server.inspect_file("tool.txt")
    read = server.read_text("tool.txt")
    searched = server.search_text("hello")
    patched = server.apply_text_patch(
        "tool.txt",
        expected_sha256=read["sha256"],
        replacements=[{"old_text": "hello", "new_text": "hi"}],
    )

    assert roots["count"] == 1
    assert inspected["relative_path"] == "tool.txt"
    assert read["text"] == "hello\n"
    assert searched["matches"][0]["relative_path"] == "tool.txt"
    assert patched["dry_run"] is True
    assert "hi" in patched["diff"]
