import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schemas" / "scene-rpc" / "v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scene_rpc_schema_and_golden_messages_are_valid() -> None:
    schema = _load(SCHEMA_ROOT / "scene-message.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    fixtures = sorted((SCHEMA_ROOT / "examples").glob("*.json"))
    assert fixtures
    for fixture in fixtures:
        validator.validate(_load(fixture))


def test_scene_rpc_rejects_unpreviewed_operations_and_protocol_drift() -> None:
    schema = _load(SCHEMA_ROOT / "scene-message.schema.json")
    validator = Draft202012Validator(schema)
    preview = _load(SCHEMA_ROOT / "examples" / "preview-request.json")

    preview["params"]["commands"][0]["kind"] = "execute_csharp"
    assert list(validator.iter_errors(preview))

    preview = _load(SCHEMA_ROOT / "examples" / "preview-request.json")
    preview["unexpected"] = True
    assert list(validator.iter_errors(preview))

    registration = _load(SCHEMA_ROOT / "examples" / "runtime-register.json")
    registration["params"]["protocolVersion"] = "brainregion.scene.v2"
    assert list(validator.iter_errors(registration))


def test_scene_rpc_rejects_cross_runtime_contract_drift() -> None:
    schema = _load(SCHEMA_ROOT / "scene-message.schema.json")
    validator = Draft202012Validator(schema)

    preview = _load(SCHEMA_ROOT / "examples" / "preview-request.json")
    preview["params"]["commands"][0]["tempId"] = "tmp:"
    assert list(validator.iter_errors(preview))

    preview = _load(SCHEMA_ROOT / "examples" / "preview-request.json")
    preview["params"]["commands"][0]["localTransform"] = {}
    assert list(validator.iter_errors(preview))

    preview = _load(SCHEMA_ROOT / "examples" / "preview-request.json")
    preview["params"]["commands"][1] = {
        "kind": "set_properties",
        "objectId": "existing-object-01",
        "componentId": "existing-object-01/light",
        "changes": [{"propertyId": "bad/path", "value": 1}],
    }
    assert list(validator.iter_errors(preview))

    registration = _load(SCHEMA_ROOT / "examples" / "runtime-register.json")
    registration["params"]["capabilities"] = []
    assert list(validator.iter_errors(registration))

    preview = _load(SCHEMA_ROOT / "examples" / "preview-request.json")
    preview["deadlineUnixMs"] = 9_007_199_254_740_992
    assert list(validator.iter_errors(preview))


def test_scene_rpc_pairing_challenge_and_proof_are_closed_contracts() -> None:
    schema = _load(SCHEMA_ROOT / "scene-message.schema.json")
    validator = Draft202012Validator(schema)

    challenge = _load(SCHEMA_ROOT / "examples" / "runtime-challenge.json")
    challenge["params"]["nonce"] = "short"
    assert list(validator.iter_errors(challenge))

    challenge = _load(SCHEMA_ROOT / "examples" / "runtime-challenge.json")
    del challenge["params"]["grantedCapabilities"]
    assert list(validator.iter_errors(challenge))

    challenge = _load(SCHEMA_ROOT / "examples" / "runtime-challenge.json")
    challenge["params"]["grantedCapabilities"] = ["scene.read", "scene.read"]
    assert list(validator.iter_errors(challenge))

    challenge = _load(SCHEMA_ROOT / "examples" / "runtime-challenge.json")
    challenge["params"]["grantedCapabilities"] = ["shell.execute"]
    assert list(validator.iter_errors(challenge))

    registration = _load(SCHEMA_ROOT / "examples" / "runtime-register.json")
    registration["params"]["pairingProof"] = "static-replayable-token"
    assert list(validator.iter_errors(registration))
