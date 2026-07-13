from __future__ import annotations

import json
from pathlib import Path

from brainregion import defaults


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_context_export_policy_defaults_to_off(monkeypatch):
    monkeypatch.setattr(defaults, "_global_config_paths", lambda: [])
    monkeypatch.setattr(defaults, "_project_config_paths", lambda: [])

    assert defaults.apply()["context_export_policy"] == {"mode": "off"}


def test_global_config_is_loaded_from_explicit_path(tmp_path, monkeypatch):
    global_config = tmp_path / "global.json"
    _write_json(
        global_config,
        {
            "panel": ["openai/gpt-4.1-mini"],
            "max_cost_usd": 0.25,
        },
    )
    monkeypatch.setenv("DESIGN_REVIEW_CONFIG", str(global_config))

    got = defaults.get_all()

    assert got["panel"]["value"] == ["openai/gpt-4.1-mini"]
    assert got["panel"]["source"] == "global_config"
    assert got["max_cost_usd"]["value"] == 0.25


def test_brain_region_config_takes_precedence_over_legacy_explicit_path(tmp_path, monkeypatch):
    legacy_config = tmp_path / "legacy.json"
    brain_region_config = tmp_path / "brain_region.json"
    _write_json(legacy_config, {"panel": ["legacy-model"]})
    _write_json(brain_region_config, {"panel": ["brain-region-model"]})
    monkeypatch.setenv("DESIGN_REVIEW_CONFIG", str(legacy_config))
    monkeypatch.setenv("BRAIN_REGION_CONFIG", str(brain_region_config))

    got = defaults.get_all()

    assert got["panel"]["value"] == ["brain-region-model"]
    assert got["panel"]["source"] == "global_config"


def test_brain_region_config_accepts_utf8_bom(tmp_path, monkeypatch):
    brain_region_config = tmp_path / "brain_region.json"
    brain_region_config.write_text(
        json.dumps(
            {
                "panel": ["modelbridge_anthropic/claude-sonnet-4-6"],
                "endpoints": {"modelbridge_anthropic": {"provider": "anthropic"}},
            }
        ),
        encoding="utf-8-sig",
    )
    monkeypatch.setenv("BRAIN_REGION_CONFIG", str(brain_region_config))

    got = defaults.get_all()

    assert got["panel"]["value"] == ["modelbridge_anthropic/claude-sonnet-4-6"]
    assert "modelbridge_anthropic" in got["endpoints"]["value"]


def test_project_config_overrides_and_merges_global_config(tmp_path, monkeypatch):
    global_config = tmp_path / "global.json"
    _write_json(
        global_config,
        {
            "panel": ["global-model"],
            "endpoints": {
                "global": {
                    "provider": "openai",
                    "base_url": "https://global.example/v1",
                    "api_key_env": "GLOBAL_KEY",
                    "models": ["global-model"],
                }
            },
            "context_modes": {"visionary": "compressed"},
        },
    )
    project_config = tmp_path / "Assets" / "Generated" / "AIGenerated" / "design_review_config.json"
    _write_json(
        project_config,
        {
            "panel": ["project-model"],
            "endpoints": {
                "project": {
                    "provider": "anthropic",
                    "base_url": "https://project.example",
                    "api_key_env": "PROJECT_KEY",
                    "models": ["project-model"],
                }
            },
            "context_modes": {"feasibility": "minimal"},
        },
    )
    monkeypatch.setenv("DESIGN_REVIEW_CONFIG", str(global_config))

    got = defaults.get_all()

    assert got["panel"]["value"] == ["project-model"]
    assert got["panel"]["source"] == "project_config"
    assert sorted(got["endpoints"]["value"]) == ["global", "project"]
    assert got["context_modes"]["value"] == {
        "visionary": "compressed",
        "feasibility": "minimal",
    }


def test_brain_region_project_config_overrides_legacy_project_config(tmp_path, monkeypatch):
    legacy_project_config = tmp_path / "Assets" / "Generated" / "AIGenerated" / "design_review_config.json"
    brain_region_project_config = tmp_path / "Assets" / "Generated" / "AIGenerated" / "brain_region_config.json"
    _write_json(legacy_project_config, {"panel": ["legacy-project-model"]})
    _write_json(brain_region_project_config, {"panel": ["brain-region-project-model"]})
    monkeypatch.setenv("UNITY_PROJECT_ROOT", str(tmp_path))

    got = defaults.get_all()

    assert got["panel"]["value"] == ["brain-region-project-model"]
    assert got["panel"]["source"] == "project_config"
    assert got["panel"]["path"] == str(brain_region_project_config)


def test_env_still_overrides_config(tmp_path, monkeypatch):
    global_config = tmp_path / "global.json"
    _write_json(global_config, {"timeout": 180})
    monkeypatch.setenv("DESIGN_REVIEW_CONFIG", str(global_config))
    monkeypatch.setenv("DESIGN_REVIEW_DEFAULT_TIMEOUT", "30")

    got = defaults.get_all()

    assert got["timeout"] == {"value": 30.0, "source": "env"}


def test_brain_region_env_overrides_legacy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DESIGN_REVIEW_DEFAULT_TIMEOUT", "60")
    monkeypatch.setenv("BRAIN_REGION_DEFAULT_TIMEOUT", "15")

    got = defaults.get_all()

    assert got["timeout"] == {"value": 15.0, "source": "env"}


def test_apply_only_uses_non_none_explicit_overrides(tmp_path, monkeypatch):
    global_config = tmp_path / "global.json"
    _write_json(global_config, {"retrieve_top_k": 9, "output_format": "markdown"})
    monkeypatch.setenv("DESIGN_REVIEW_CONFIG", str(global_config))

    got = defaults.apply(retrieve_top_k=None, output_format=None)

    assert got["retrieve_top_k"] == 9
    assert got["output_format"] == "markdown"
