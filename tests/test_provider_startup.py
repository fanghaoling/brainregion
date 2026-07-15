"""Startup boundaries for optional heavyweight model-provider dependencies."""

from __future__ import annotations

import sys
from types import ModuleType

from brainregion.providers import litellm as backend


def test_litellm_backend_defaults_to_local_cost_map_without_overriding_operator(
    monkeypatch,
):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    backend._configure_litellm_environment()
    assert backend.os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")
    backend._configure_litellm_environment()
    assert backend.os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "False"


def test_litellm_loader_configures_dependency_only_when_called(monkeypatch):
    fake = ModuleType("litellm")
    fake.suppress_debug_info = False
    monkeypatch.setitem(sys.modules, "litellm", fake)

    loaded = backend._load_litellm()

    assert loaded is fake
    assert fake.suppress_debug_info is True
