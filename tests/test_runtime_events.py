from __future__ import annotations

import json

from brainregion.runtime.events import RuntimeEventStore


def test_runtime_event_store_emits_lists_waits_and_persists(tmp_path):
    path = tmp_path / "events.jsonl"
    store = RuntimeEventStore(path=path, max_events=5)

    event = store.emit("region.activation", region_id="memory", payload={"score": 8})

    assert event["sequence"] > 0
    assert event["type"] == "region.activation"
    assert event["region_id"] == "memory"
    assert store.list(after_sequence=0)[0]["payload"] == {"score": 8}
    assert store.wait(after_sequence=0, timeout=0.01)[0]["sequence"] == event["sequence"]

    persisted = json.loads(path.read_text(encoding="utf-8").strip())
    assert persisted["type"] == "region.activation"
    assert persisted["payload"] == {"score": 8}


def test_runtime_event_store_after_sequence_and_ring_limit(tmp_path):
    store = RuntimeEventStore(path=tmp_path / "events.jsonl", max_events=2)

    store.emit("a")
    second = store.emit("b")
    third = store.emit("c")

    assert [e["type"] for e in store.list(after_sequence=0, limit=2)] == ["b", "c"]
    assert store.list(after_sequence=second["sequence"]) == [third]


def test_runtime_event_store_reads_jsonl_from_other_store(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = RuntimeEventStore(path=path)
    reader = RuntimeEventStore(path=path)

    event = writer.emit("model.call_finished", payload={"model": "m"})

    assert reader.list(after_sequence=0)[0]["id"] == event["id"]
    assert reader.wait(after_sequence=0, timeout=0.01)[0]["type"] == "model.call_finished"
