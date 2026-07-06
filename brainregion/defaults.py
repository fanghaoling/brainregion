"""Layered defaults for BrainRegion.

Precedence:
    builtin < global config < project config < env < explicit overrides

Environment value overrides use ``BRAIN_REGION_DEFAULT_<KEY>`` or the legacy
``DESIGN_REVIEW_DEFAULT_<KEY>``, for example ``BRAIN_REGION_DEFAULT_PANEL``.

Global config is loaded from ``BRAIN_REGION_CONFIG`` when set, then the legacy
``DESIGN_REVIEW_CONFIG``. Without an explicit path, both BrainRegion and legacy
design-review config paths are checked.

Project config stays at
``$UNITY_PROJECT_ROOT/Assets/Generated/AIGenerated/brain_region_config.json``.
The legacy ``design_review_config.json`` path is still loaded for compatibility.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("brainregion.defaults")

_ENV_PREFIX = "BRAIN_REGION_DEFAULT_"
_LEGACY_ENV_PREFIX = "DESIGN_REVIEW_DEFAULT_"

# memory_scope 合法值（2 mode 用常量集够；第 3 个 mode 出现再升 enum）。
MEMORY_SCOPE_MODES = {"woken", "none"}

_BUILTINS = {
    "panel": ["claude-opus-4-8", "gpt-5"],
    "dimensions": [],
    "temperature": 0.3,
    "max_tokens": 4096,
    "consensus_threshold": 2,
    "retrieve_top_k": 5,
    "timeout": 90,
    "normalizer_model": "claude-opus-4-8",
    "output_format": "json",
    "effort": None,
    "max_cost_usd": None,
    "endpoints": {},
    "model_prices": {},
    "model_profiles": {},
    "workspace_roots": [],
    "privacy_policy": None,
    "context_modes": {},
    "min_compressed_chars": 50,
    "model_reliability_prior": {"mode": "builtin", "custom": {}},
    "consult_panel": [],
    "consult_consultants": ["debugger", "architect", "critic"],
    "consult_mode": None,
    "consult_max_input_chars": 24000,
    "consult_max_cost_usd": None,
    "memory_inject": False,
    "context_top_k": 5,  # Phase 7:每 ContextProvider 召回 top_k(memory/git/…共用;曾名 memory_recall_top_k)
    "memory_scope": "woken",  # Phase A：memory 召回按 wake 的 region scope（"none"=unscoped 消融）
    "planner_panel": [],
    "planner_max_input_chars": 24000,
    "planner_max_cost_usd": None,
    # 沙盒(agent loop)配置 —— opt-in(CLI 触发,默认不跑;详见 sandbox/)
    "sandbox_max_steps": 10,
    "sandbox_max_cost_usd": 0.5,
    "sandbox_main_brain": "",
    "sandbox_consecutive_error_limit": 3,
    "sandbox_transcript_token_cap": 24000,
    "sandbox_temperature": 0.0,
}


def _project_config_paths() -> list[Path]:
    root = os.environ.get("UNITY_PROJECT_ROOT", ".")
    generated = Path(root) / "Assets" / "Generated" / "AIGenerated"
    return [
        generated / "design_review_config.json",
        generated / "brain_region_config.json",
    ]


def _global_config_paths() -> list[Path]:
    explicit = os.environ.get("BRAIN_REGION_CONFIG")
    if explicit:
        return [Path(explicit)]

    explicit = os.environ.get("DESIGN_REVIEW_CONFIG")
    if explicit:
        return [Path(explicit)]

    paths: list[Path] = []
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        paths.append(Path(codex_home) / "design_review_config.json")
        paths.append(Path(codex_home) / "brain_region_config.json")
    paths.append(Path.home() / ".codex" / "design_review_config.json")
    paths.append(Path.home() / ".codex" / "brain_region_config.json")

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        paths.append(Path(xdg_config) / "design-review" / "config.json")
        paths.append(Path(xdg_config) / "brain-region" / "config.json")
    else:
        paths.append(Path.home() / ".config" / "design-review" / "config.json")
        paths.append(Path.home() / ".config" / "brain-region" / "config.json")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _load_json_config(path: Path) -> dict[str, Any]:
    p = path.expanduser()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ignoring invalid BrainRegion config %s: %s", p, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _merge_value(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge_value(merged.get(key), value)
        return merged
    return override


def _apply_config_layer(result: dict[str, dict[str, Any]], cfg: dict[str, Any], source: str, path: Path | None) -> None:
    for key, value in cfg.items():
        if key not in _BUILTINS:
            continue
        old_value = result[key]["value"]
        result[key] = {
            "value": _merge_value(old_value, value),
            "source": source,
        }
        if path is not None:
            result[key]["path"] = str(path.expanduser())


def _coerce(key: str, val: str):
    if key == "memory_inject":
        return val.strip().lower() in ("1", "true", "yes", "on")
    if key == "memory_scope":
        v = val.strip().lower()
        return v if v in MEMORY_SCOPE_MODES else "woken"
    if key in (
        "temperature",
        "timeout",
        "max_cost_usd",
        "consult_max_cost_usd",
        "planner_max_cost_usd",
        "sandbox_max_cost_usd",
        "sandbox_temperature",
    ):
        try:
            return float(val)
        except ValueError:
            return val
    if key in (
        "max_tokens",
        "consensus_threshold",
        "retrieve_top_k",
        "min_compressed_chars",
        "consult_max_input_chars",
        "planner_max_input_chars",
        "context_top_k",
        "sandbox_max_steps",
        "sandbox_consecutive_error_limit",
        "sandbox_transcript_token_cap",
    ):
        try:
            return int(val)
        except ValueError:
            return val
    return val


def get_all() -> dict:
    """Return ``{key: {value, source}}`` with layered config provenance."""
    result = {key: {"value": value, "source": "builtin"} for key, value in _BUILTINS.items()}

    for path in _global_config_paths():
        _apply_config_layer(result, _load_json_config(path), "global_config", path)

    for project_path in _project_config_paths():
        _apply_config_layer(result, _load_json_config(project_path), "project_config", project_path)

    for key in _BUILTINS:
        legacy_ev = os.environ.get(f"{_LEGACY_ENV_PREFIX}{key.upper()}")
        if legacy_ev is not None:
            result[key] = {"value": _coerce(key, legacy_ev), "source": "env"}
        ev = os.environ.get(f"{_ENV_PREFIX}{key.upper()}")
        if ev is not None:
            result[key] = {"value": _coerce(key, ev), "source": "env"}
    return result


def apply(**overrides) -> dict:
    """Merge builtin < global config < project config < env < explicit overrides."""
    merged = {key: value["value"] for key, value in get_all().items()}
    for key, value in overrides.items():
        if value is not None and key in merged:
            merged[key] = value
    return merged
