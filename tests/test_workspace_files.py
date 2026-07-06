from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from brainregion.workspace.files import MAX_READ_FILE_BYTES, inspect_file, list_allowed_roots, read_text


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


def test_server_mcp_tools_delegate_to_workspace(workspace_root):
    from brainregion import server

    (workspace_root / "tool.txt").write_text("hello\n", encoding="utf-8", newline="\n")

    roots = server.list_allowed_roots()
    inspected = server.inspect_file("tool.txt")
    read = server.read_text("tool.txt")

    assert roots["count"] == 1
    assert inspected["relative_path"] == "tool.txt"
    assert read["text"] == "hello\n"
