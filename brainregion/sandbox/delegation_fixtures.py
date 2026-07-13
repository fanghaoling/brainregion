"""Opt-in multi-file fixtures for delegation difficulty calibration.

These tasks are intentionally excluded from the default sandbox suite. They
test whether an expert report can reduce search and context-acquisition work
for a weaker main model without changing the objective pytest acceptance rule.
"""

from __future__ import annotations

from .task import SandboxTask


_TENANT_CACHE_FILES = {
    "app/__init__.py": "",
    "app/models.py": """from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    tenant_id: str
    user_id: int
    display_name: str
""",
    "app/cache.py": """class MemoryCache:
    def __init__(self):
        self._values = {}

    def get(self, key):
        return self._values.get(key)

    def put(self, key, value):
        self._values[key] = value
""",
    "app/repository.py": """class ProfileRepository:
    def __init__(self, records):
        self._records = dict(records)
        self.load_calls = []

    def load(self, tenant_id, user_id):
        self.load_calls.append((tenant_id, user_id))
        return self._records[(tenant_id, user_id)]
""",
    "app/service.py": """class ProfileService:
    def __init__(self, repository, cache):
        self._repository = repository
        self._cache = cache

    def get_profile(self, tenant_id, user_id):
        key = self._cache_key(user_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        profile = self._repository.load(tenant_id, user_id)
        self._cache.put(key, profile)
        return profile

    @staticmethod
    def _cache_key(user_id):
        return f"profile:{user_id}"
""",
    "app/audit.py": """def profile_access_event(tenant_id, user_id):
    return {"tenant_id": tenant_id, "user_id": user_id}
""",
}

_TENANT_CACHE_TESTS = {
    "test_profile_service.py": """from app.cache import MemoryCache
from app.models import Profile
from app.repository import ProfileRepository
from app.service import ProfileService


def test_same_user_id_is_cached_independently_per_tenant():
    records = {
        ("alpha", 7): Profile("alpha", 7, "Ada"),
        ("beta", 7): Profile("beta", 7, "Grace"),
    }
    repository = ProfileRepository(records)
    service = ProfileService(repository, MemoryCache())

    assert service.get_profile("alpha", 7).display_name == "Ada"
    assert service.get_profile("beta", 7).display_name == "Grace"
    assert service.get_profile("alpha", 7).display_name == "Ada"
    assert service.get_profile("beta", 7).display_name == "Grace"
    assert repository.load_calls == [("alpha", 7), ("beta", 7)]
""",
}


_SETTINGS_FILES = {
    "settings/__init__.py": "",
    "settings/defaults.py": """DEFAULTS = {
    "timeout": 30,
    "theme": "light",
    "retries": 1,
}
""",
    "settings/merge.py": '''def merge_layers(*layers):
    """Merge low-to-high priority mappings into a new dictionary."""
    merged = {}
    for layer in reversed(layers):
        merged.update(layer)
    return merged
''',
    "settings/loader.py": """from .defaults import DEFAULTS
from .merge import merge_layers


def load_settings(project=None, user=None):
    return merge_layers(DEFAULTS, project or {}, user or {})
""",
    "settings/validation.py": """def validate_settings(settings):
    if settings.get("timeout", 0) <= 0:
        raise ValueError("timeout must be positive")
    return dict(settings)
""",
}

_SETTINGS_TESTS = {
    "test_settings.py": """from settings.loader import load_settings


def test_user_overrides_project_and_project_overrides_defaults():
    project = {"timeout": 60, "theme": "dark"}
    user = {"timeout": 10}

    result = load_settings(project, user)

    assert result == {"timeout": 10, "theme": "dark", "retries": 1}
    assert project == {"timeout": 60, "theme": "dark"}
    assert user == {"timeout": 10}


def test_empty_layers_return_independent_defaults_copy():
    first = load_settings()
    first["theme"] = "changed"
    assert load_settings()["theme"] == "light"
""",
}


_EVENT_BUS_FILES = {
    "events/__init__.py": "",
    "events/bus.py": """class EventBus:
    def __init__(self):
        self._listeners = {}

    def on(self, event, listener):
        self._listeners.setdefault(event, []).append(listener)

    def off(self, event, listener):
        listeners = self._listeners.get(event, [])
        if listener in listeners:
            listeners.remove(listener)

    def emit(self, event, payload):
        for listener in self._listeners.get(event, []):
            listener(payload)
""",
    "events/subscription.py": """class Subscription:
    def __init__(self, bus, event, listener):
        self._bus = bus
        self._event = event
        self._listener = listener

    def close(self):
        self._bus.off(self._event, self._listener)
""",
    "events/registry.py": """def registered_events(bus):
    return tuple(sorted(bus._listeners))
""",
}

_EVENT_BUS_TESTS = {
    "test_event_bus.py": """from events.bus import EventBus


def test_self_unsubscribe_does_not_skip_the_next_listener():
    bus = EventBus()
    calls = []

    def first(payload):
        calls.append(("first", payload))
        bus.off("update", first)

    def second(payload):
        calls.append(("second", payload))

    bus.on("update", first)
    bus.on("update", second)
    bus.emit("update", 1)
    bus.emit("update", 2)

    assert calls == [("first", 1), ("second", 1), ("second", 2)]


def test_listener_added_during_emit_starts_on_the_next_emit():
    bus = EventBus()
    calls = []

    def late(payload):
        calls.append(("late", payload))

    def first(payload):
        calls.append(("first", payload))
        bus.on("update", late)

    def second(payload):
        calls.append(("second", payload))

    bus.on("update", first)
    bus.on("update", second)
    bus.emit("update", 1)

    assert calls == [("first", 1), ("second", 1)]
""",
}


_RETRY_FILES = {
    "net/__init__.py": "",
    "net/errors.py": """class RequestError(Exception):
    pass


class TransientError(RequestError):
    pass


class AuthenticationError(RequestError):
    pass
""",
    "net/retry.py": """from .errors import RequestError


def call_with_retry(operation, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return operation()
        except RequestError:
            if attempt + 1 == max_attempts:
                raise
    raise RuntimeError("unreachable")
""",
    "net/client.py": """from .retry import call_with_retry


class Client:
    def __init__(self, transport):
        self._transport = transport

    def fetch(self):
        return call_with_retry(self._transport.send)
""",
    "net/transport.py": """class ScriptedTransport:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def send(self):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
""",
}

_RETRY_TESTS = {
    "test_retry.py": """import pytest

from net.client import Client
from net.errors import AuthenticationError, TransientError
from net.transport import ScriptedTransport


def test_authentication_failure_is_not_retried():
    transport = ScriptedTransport([
        AuthenticationError("bad token"),
        "must not be reached",
    ])

    with pytest.raises(AuthenticationError, match="bad token"):
        Client(transport).fetch()

    assert transport.calls == 1


def test_transient_failure_is_retried():
    transport = ScriptedTransport([TransientError("busy"), "ok"])
    assert Client(transport).fetch() == "ok"
    assert transport.calls == 2
""",
}


DELEGATION_CALIBRATION_FIXTURES: list[SandboxTask] = [
    SandboxTask(
        id="tenant_cache_scope",
        goal=(
            "A profile request in one tenant can return another tenant's cached profile when user IDs match. "
            "Preserve caching and the public get_profile API, then make the tests pass."
        ),
        files=_TENANT_CACHE_FILES,
        tests=_TENANT_CACHE_TESTS,
        gold_diff="Include tenant_id in ProfileService cache-key construction.",
        gold_regions=["debugging", "security"],
        notes="Cross-file ownership bug: repository identity is tenant+user, but the cache key drops tenant scope.",
    ),
    SandboxTask(
        id="settings_precedence",
        goal=(
            "User settings are ignored when the same key exists in project settings or defaults. "
            "Keep all input mappings immutable and make the tests pass."
        ),
        files=_SETTINGS_FILES,
        tests=_SETTINGS_TESTS,
        gold_diff="Merge DEFAULTS, project, then user in low-to-high priority order.",
        gold_regions=["debugging", "review"],
        notes="The loader supplies the correct layer order; the shared merge helper reverses it.",
    ),
    SandboxTask(
        id="event_bus_snapshot",
        goal=(
            "Listener changes during emit cause inconsistent delivery. Each emit must notify exactly the listeners "
            "that were present when that emit began, while preserving registration order."
        ),
        files=_EVENT_BUS_FILES,
        tests=_EVENT_BUS_TESTS,
        gold_diff="Iterate over a snapshot of the listener list in EventBus.emit.",
        gold_regions=["debugging", "review"],
        notes="Mutating the live listener list during iteration can skip or prematurely include callbacks.",
    ),
    SandboxTask(
        id="retry_error_scope",
        goal=(
            "Authentication failures are being retried and can hide the original failure, while transient request "
            "failures must still retry. Preserve the Client API and make the tests pass."
        ),
        files=_RETRY_FILES,
        tests=_RETRY_TESTS,
        gold_diff="Retry only TransientError rather than every RequestError.",
        gold_regions=["debugging", "review"],
        notes="The exception hierarchy distinguishes retryable transport failures from permanent authentication failures.",
    ),
]


__all__ = ["DELEGATION_CALIBRATION_FIXTURES"]
