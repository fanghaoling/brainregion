import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "agent-core" / "v1" / "control-message.schema.json"


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _valid_scene_messages() -> list[dict]:
    return [
        {
            "jsonrpc": "2.0",
            "id": "list-1",
            "method": "scene/peers/list",
        },
        {
            "jsonrpc": "2.0",
            "id": "call-1",
            "method": "scene/peer/call",
            "params": {
                "principalId": "local-vr-player",
                "method": "scene/hierarchy",
                "params": {"depth": 2, "includeInactive": False},
                "timeoutMs": 1_000,
            },
        },
        {
            "jsonrpc": "2.0",
            "id": "list-1",
            "result": {
                "peers": [
                    {
                        "principalId": "local-vr-player",
                        "connectionEpoch": 3,
                        "instanceId": "player-process-1",
                        "sessionId": "runtime-session-1",
                        "buildId": "windows-il2cpp-dev",
                        "sceneId": "Sandbox",
                        "sceneRevision": 7,
                        "runtimeStatus": "ready",
                        "runtimeError": None,
                        "supportedCapabilities": ["scene.read", "logs.read"],
                        "grantedCapabilities": ["scene.read"],
                        "connectionState": "connected",
                    }
                ]
            },
        },
        {
            "jsonrpc": "2.0",
            "id": "call-1",
            "result": {
                "principalId": "local-vr-player",
                "method": "scene/hierarchy",
                "result": {"sceneRevision": 7, "nodes": []},
            },
        },
    ]


def test_agent_core_schema_accepts_runtime_scene_peer_control_messages() -> None:
    validator = _validator()

    for message in _valid_scene_messages():
        validator.validate(message)


def test_agent_core_schema_rejects_scene_peer_contract_drift() -> None:
    validator = _validator()
    request = _valid_scene_messages()[1]

    unknown_method = json.loads(json.dumps(request))
    unknown_method["params"]["method"] = "scene/eval"
    assert list(validator.iter_errors(unknown_method))

    invalid_timeout = json.loads(json.dumps(request))
    invalid_timeout["params"]["timeoutMs"] = 300_001
    assert list(validator.iter_errors(invalid_timeout))

    extra_field = json.loads(json.dumps(request))
    extra_field["params"]["retry"] = True
    assert list(validator.iter_errors(extra_field))
