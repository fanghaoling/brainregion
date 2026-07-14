from __future__ import annotations

import json
import os

import pytest

from brainregion import defaults
from brainregion.cli import _apply_cli_bootstrap, build_parser


def test_cli_bootstrap_loads_explicit_config_and_env_without_overwriting_process_env(
    tmp_path, monkeypatch,
):
    config_path = tmp_path / "brain_region_config.json"
    config_path.write_text(
        json.dumps({"endpoints": {"buzz_anthropic": {"provider": "anthropic"}}}),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BUZZ_API_KEY=from-file\nEXISTING_KEY=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BRAIN_REGION_CONFIG", str(tmp_path / "old.json"))
    monkeypatch.setenv("EXISTING_KEY", "from-process")
    monkeypatch.delenv("BUZZ_API_KEY", raising=False)

    args = build_parser().parse_args([
        "--config", str(config_path), "--env-file", str(env_path),
        "plan", "--text", "check startup",
    ])
    _apply_cli_bootstrap(args)

    assert os.environ["BRAIN_REGION_CONFIG"] == str(config_path.resolve())
    assert os.environ["BUZZ_API_KEY"] == "from-file"
    assert os.environ["EXISTING_KEY"] == "from-process"
    assert "buzz_anthropic" in defaults.apply()["endpoints"]


@pytest.mark.parametrize("flag", ["--config", "--env-file"])
def test_cli_bootstrap_rejects_missing_startup_file(flag, tmp_path):
    args = build_parser().parse_args([flag, str(tmp_path / "missing"), "plan", "--text", "x"])

    with pytest.raises(SystemExit, match="file not found"):
        _apply_cli_bootstrap(args)
