"""Opt-in multi-file delegation calibration fixture tests."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from brainregion.sandbox.cli import _resolve_tasks
from brainregion.sandbox.delegation_fixtures import DELEGATION_CALIBRATION_FIXTURES
from brainregion.sandbox.fixtures import SANDBOX_FIXTURES, get_fixture, list_fixture_ids
from brainregion.sandbox.isolation import cleanup_run_dir, make_run_dir, materialize_fixture
from brainregion.sandbox.verify import verify_solution


_GOLD_REPLACEMENTS = {
    "tenant_cache_scope": (
        "app/service.py",
        (
            ("key = self._cache_key(user_id)", "key = self._cache_key(tenant_id, user_id)"),
            ("def _cache_key(user_id):", "def _cache_key(tenant_id, user_id):"),
            ('return f"profile:{user_id}"', 'return f"profile:{tenant_id}:{user_id}"'),
        ),
    ),
    "settings_precedence": (
        "settings/merge.py",
        (("for layer in reversed(layers):", "for layer in layers:"),),
    ),
    "event_bus_snapshot": (
        "events/bus.py",
        (
            (
                "for listener in self._listeners.get(event, []):",
                "for listener in tuple(self._listeners.get(event, [])):",
            ),
        ),
    ),
    "retry_error_scope": (
        "net/retry.py",
        (
            ("from .errors import RequestError", "from .errors import TransientError"),
            ("except RequestError:", "except TransientError:"),
        ),
    ),
    "order_idempotency_rollback": (
        "orders/service.py",
        (
            (
                "from .models import Order",
                "from .models import Order\nfrom .payments import PaymentError",
            ),
            (
                '''    def place_order(self, request_id, sku, quantity):
        self._inventory.reserve(sku, quantity)
        receipt = self._payments.charge(request_id, quantity * 10)
        order = Order(request_id, sku, quantity, receipt)
        self._orders[request_id] = order
        return order''',
                '''    def place_order(self, request_id, sku, quantity):
        existing = self._orders.get(request_id)
        if existing is not None:
            if existing.sku != sku or existing.quantity != quantity:
                raise ValueError("request_id already used with a different payload")
            return existing

        self._inventory.reserve(sku, quantity)
        try:
            receipt = self._payments.charge(request_id, quantity * 10)
        except PaymentError:
            self._inventory.release(sku, quantity)
            raise
        order = Order(request_id, sku, quantity, receipt)
        self._orders[request_id] = order
        return order''',
            ),
        ),
    ),
}


def test_calibration_fixtures_are_discoverable_but_not_in_default_suite():
    calibration_ids = [task.id for task in DELEGATION_CALIBRATION_FIXTURES]
    default_ids = [task.id for task in SANDBOX_FIXTURES]

    assert calibration_ids == [
        "tenant_cache_scope",
        "settings_precedence",
        "event_bus_snapshot",
        "retry_error_scope",
        "order_idempotency_rollback",
    ]
    assert set(calibration_ids).isdisjoint(default_ids)
    assert set(calibration_ids).issubset(list_fixture_ids())
    assert [task.id for task in _resolve_tasks(Namespace(task=None, tasks=None))] == default_ids
    assert get_fixture("tenant_cache_scope").id == "tenant_cache_scope"


def test_calibration_fixtures_require_explicit_selection():
    selected = _resolve_tasks(Namespace(task=None, tasks="tenant_cache_scope,event_bus_snapshot"))

    assert [task.id for task in selected] == ["tenant_cache_scope", "event_bus_snapshot"]


@pytest.mark.parametrize("task", DELEGATION_CALIBRATION_FIXTURES, ids=lambda task: task.id)
def test_calibration_fixture_starts_red_and_minimal_gold_fix_turns_green(task):
    run_dir = make_run_dir(prefix="brainregion-delegation-fixture-test-")
    materialize_fixture(task, Path(run_dir))
    try:
        assert verify_solution(task, run_dir)["tests_green"] is False

        relative_path, replacements = _GOLD_REPLACEMENTS[task.id]
        target = Path(run_dir, relative_path)
        content = target.read_text(encoding="utf-8")
        for old_text, new_text in replacements:
            assert content.count(old_text) == 1
            content = content.replace(old_text, new_text)
        target.write_text(content, encoding="utf-8")

        assert verify_solution(task, run_dir)["tests_green"] is True
    finally:
        cleanup_run_dir(run_dir)
