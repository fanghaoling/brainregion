//! Typed pieces of the transport-neutral Runtime Scene RPC contract.
//!
//! The Unity Player is a peer rather than an MCP server. `brainregiond` will
//! correlate these JSON-RPC requests over a paired session and expose a smaller
//! permissioned surface to desktop/VR clients and, later, as native MCP tools.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCENE_PROTOCOL_VERSION: &str = "brainregion.scene.v1";
pub const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
pub const MAX_SCENE_FRAME_BYTES: usize = 1024 * 1024;
pub const MAX_COMMANDS_PER_TRANSACTION: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcRequest<P> {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    pub deadline_unix_ms: u64,
    pub params: P,
}

impl<P> RpcRequest<P> {
    pub fn new(
        id: impl Into<String>,
        method: impl Into<String>,
        deadline_unix_ms: u64,
        params: P,
    ) -> Self {
        Self {
            jsonrpc: "2.0".to_owned(),
            id: id.into(),
            method: method.into(),
            deadline_unix_ms,
            params,
        }
    }

    pub fn validate_envelope(&self, expected_method: &str) -> Result<(), String> {
        if self.jsonrpc != "2.0" {
            return Err("jsonrpc must be exactly \"2.0\"".to_owned());
        }
        validate_identifier("id", &self.id, 128)?;
        if self.method != expected_method {
            return Err(format!(
                "expected method {expected_method:?}, got {:?}",
                self.method
            ));
        }
        validate_json_safe_integer("deadlineUnixMs", self.deadline_unix_ms)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeRegistrationNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: RuntimeRegistration,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeRegistration {
    pub protocol_version: String,
    pub instance_id: String,
    pub session_id: String,
    pub build_id: String,
    pub unity_version: String,
    pub platform: String,
    pub product: String,
    pub scene_id: String,
    pub scene_revision: u64,
    pub status: RuntimeStatus,
    pub error: Option<String>,
    pub capabilities: Vec<SceneCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_proof: Option<String>,
}

impl RuntimeRegistrationNotification {
    pub fn validate(&self) -> Result<(), String> {
        if self.jsonrpc != "2.0" || self.method != "runtime/register" {
            return Err("invalid runtime/register notification envelope".to_owned());
        }
        if self.params.protocol_version != SCENE_PROTOCOL_VERSION {
            return Err(format!(
                "unsupported Scene RPC protocol {:?}",
                self.params.protocol_version
            ));
        }
        validate_identifier("instanceId", &self.params.instance_id, 128)?;
        validate_identifier("sessionId", &self.params.session_id, 128)?;
        validate_revision(self.params.scene_revision)?;
        validate_bounded_text("buildId", &self.params.build_id, 256)?;
        validate_bounded_text("unityVersion", &self.params.unity_version, 64)?;
        validate_bounded_text("platform", &self.params.platform, 64)?;
        validate_bounded_text("product", &self.params.product, 256)?;
        validate_bounded_text("sceneId", &self.params.scene_id, 256)?;
        if self
            .params
            .error
            .as_ref()
            .is_some_and(|error| error.len() > 4096)
        {
            return Err("runtime error must not exceed 4096 bytes".to_owned());
        }
        if self
            .params
            .pairing_proof
            .as_ref()
            .is_some_and(|proof| proof.len() > 2048)
        {
            return Err("pairingProof must not exceed 2048 bytes".to_owned());
        }
        if self.params.capabilities.is_empty() {
            return Err("runtime must advertise at least one capability".to_owned());
        }
        let unique_capabilities: std::collections::HashSet<_> =
            self.params.capabilities.iter().copied().collect();
        if unique_capabilities.len() != self.params.capabilities.len() {
            return Err("runtime capabilities must be unique".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStatus {
    Ready,
    Degraded,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum SceneCapability {
    #[serde(rename = "scene.read")]
    SceneRead,
    #[serde(rename = "scene.write")]
    SceneWrite,
    #[serde(rename = "scene.spawn")]
    SceneSpawn,
    #[serde(rename = "scene.undo")]
    SceneUndo,
    #[serde(rename = "logs.read")]
    LogsRead,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RpcVector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl RpcVector3 {
    fn validate(self) -> Result<(), String> {
        if self.x.is_finite() && self.y.is_finite() && self.z.is_finite() {
            Ok(())
        } else {
            Err("vector components must be finite".to_owned())
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TransformPatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<RpcVector3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotation_euler: Option<RpcVector3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<RpcVector3>,
}

impl TransformPatch {
    pub fn validate(&self, allow_empty: bool) -> Result<(), String> {
        if !allow_empty
            && self.position.is_none()
            && self.rotation_euler.is_none()
            && self.scale.is_none()
        {
            return Err("transform patch must change at least one field".to_owned());
        }
        for vector in [self.position, self.rotation_euler, self.scale]
            .into_iter()
            .flatten()
        {
            vector.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PropertyChange {
    pub property_id: String,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SceneOperation {
    Spawn {
        #[serde(rename = "tempId")]
        temp_id: String,
        #[serde(rename = "prefabId")]
        prefab_id: String,
        #[serde(rename = "parentId", default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        #[serde(
            rename = "localTransform",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        local_transform: Option<TransformPatch>,
    },
    SetTransform {
        #[serde(rename = "objectId")]
        object_id: String,
        #[serde(rename = "localTransform")]
        local_transform: TransformPatch,
    },
    SetActive {
        #[serde(rename = "objectId")]
        object_id: String,
        active: bool,
    },
    SetProperties {
        #[serde(rename = "objectId")]
        object_id: String,
        #[serde(rename = "componentId")]
        component_id: String,
        changes: Vec<PropertyChange>,
    },
}

impl SceneOperation {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Spawn {
                temp_id,
                prefab_id,
                parent_id,
                local_transform,
            } => {
                let Some(temp_id_suffix) = temp_id.strip_prefix("tmp:") else {
                    return Err("spawn tempId must begin with \"tmp:\"".to_owned());
                };
                if temp_id_suffix.is_empty() {
                    return Err("spawn tempId must contain a value after \"tmp:\"".to_owned());
                }
                validate_identifier("tempId", temp_id, 124)?;
                validate_identifier_segment("tempId suffix", temp_id_suffix, 120)?;
                validate_identifier("prefabId", prefab_id, 128)?;
                if let Some(parent_id) = parent_id {
                    validate_identifier("parentId", parent_id, 160)?;
                }
                if let Some(transform) = local_transform {
                    transform.validate(false)?;
                }
            }
            Self::SetTransform {
                object_id,
                local_transform,
            } => {
                validate_identifier("objectId", object_id, 160)?;
                local_transform.validate(false)?;
            }
            Self::SetActive { object_id, .. } => {
                validate_identifier("objectId", object_id, 160)?;
            }
            Self::SetProperties {
                object_id,
                component_id,
                changes,
            } => {
                validate_identifier("objectId", object_id, 160)?;
                validate_identifier("componentId", component_id, 200)?;
                if changes.is_empty() || changes.len() > 64 {
                    return Err("set_properties changes must contain 1..64 entries".to_owned());
                }
                let mut property_ids = std::collections::HashSet::new();
                for change in changes {
                    validate_identifier_segment("propertyId", &change.property_id, 128)?;
                    validate_property_value(&change.value, false)?;
                    if !property_ids.insert(&change.property_id) {
                        return Err(format!(
                            "duplicate propertyId {:?} in one operation",
                            change.property_id
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ScenePreviewParams {
    pub expected_revision: u64,
    pub client_mutation_id: String,
    pub commands: Vec<SceneOperation>,
}

impl ScenePreviewParams {
    pub fn validate(&self) -> Result<(), String> {
        validate_revision(self.expected_revision)?;
        validate_identifier("clientMutationId", &self.client_mutation_id, 128)?;
        if self.commands.is_empty() || self.commands.len() > MAX_COMMANDS_PER_TRANSACTION {
            return Err(format!(
                "commands must contain 1..{MAX_COMMANDS_PER_TRANSACTION} entries"
            ));
        }
        let mut temp_ids = std::collections::HashSet::new();
        for command in &self.commands {
            command.validate()?;
            if let SceneOperation::Spawn { temp_id, .. } = command
                && !temp_ids.insert(temp_id)
            {
                return Err(format!("duplicate spawn tempId {temp_id:?}"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SceneApplyParams {
    pub preview_token: String,
    pub expected_revision: u64,
    pub client_mutation_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcSuccessResponse {
    pub jsonrpc: String,
    pub id: String,
    pub result: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcErrorResponse {
    pub jsonrpc: String,
    pub id: Option<String>,
    pub error: RpcError,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

fn validate_revision(revision: u64) -> Result<(), String> {
    validate_json_safe_integer("sceneRevision", revision)
}

fn validate_json_safe_integer(name: &str, value: u64) -> Result<(), String> {
    if value <= MAX_JSON_SAFE_INTEGER {
        return Ok(());
    }
    Err(format!(
        "{name} exceeds JSON safe integer {MAX_JSON_SAFE_INTEGER}"
    ))
}

fn validate_bounded_text(name: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum {
        return Err(format!("{name} must contain 1..{maximum} bytes"));
    }
    Ok(())
}

fn validate_identifier(name: &str, value: &str, maximum: usize) -> Result<(), String> {
    validate_bounded_text(name, value, maximum)?;
    if value
        .bytes()
        .any(|byte| !(byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte)))
    {
        return Err(format!("{name} contains an unsupported character"));
    }
    Ok(())
}

fn validate_identifier_segment(name: &str, value: &str, maximum: usize) -> Result<(), String> {
    validate_identifier(name, value, maximum)?;
    if value.bytes().any(|byte| matches!(byte, b':' | b'/')) {
        return Err(format!("{name} must not contain ':' or '/'"));
    }
    Ok(())
}

fn validate_property_value(value: &Value, array_item: bool) -> Result<(), String> {
    match value {
        Value::Null | Value::Bool(_) => Ok(()),
        Value::Number(number) => {
            if number.as_f64().is_some_and(f64::is_finite) {
                Ok(())
            } else {
                Err("property number must be finite".to_owned())
            }
        }
        Value::String(text) => {
            let maximum = if array_item { 4096 } else { 16384 };
            if text.len() <= maximum {
                Ok(())
            } else {
                Err(format!("property string must not exceed {maximum} bytes"))
            }
        }
        Value::Array(items) if !array_item => {
            if items.len() > 64 {
                return Err("property arrays must not exceed 64 items".to_owned());
            }
            for item in items {
                validate_property_value(item, true)?;
            }
            Ok(())
        }
        Value::Array(_) => Err("nested property arrays are not supported".to_owned()),
        Value::Object(object) => {
            if object.len() == 3
                && object.contains_key("x")
                && object.contains_key("y")
                && object.contains_key("z")
            {
                for component in ["x", "y", "z"] {
                    let Some(number) = object[component].as_f64() else {
                        return Err("vector property components must be numbers".to_owned());
                    };
                    if !number.is_finite() {
                        return Err("vector property components must be finite".to_owned());
                    }
                }
                return Ok(());
            }
            if array_item || object.len() != 1 {
                return Err("property object must be a vector, objectRef, or assetRef".to_owned());
            }
            if let Some(Value::String(object_id)) = object.get("objectRef") {
                return validate_identifier("objectRef", object_id, 160);
            }
            if let Some(Value::String(asset_id)) = object.get("assetRef") {
                return validate_bounded_text("assetRef", asset_id, 256);
            }
            Err("property object must be a vector, objectRef, or assetRef".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_fixture_round_trips_and_validates() {
        let fixture = include_str!("../../../schemas/scene-rpc/v1/examples/preview-request.json");
        let request: RpcRequest<ScenePreviewParams> = serde_json::from_str(fixture).unwrap();

        request.validate_envelope("scene/preview").unwrap();
        request.params.validate().unwrap();
        assert_eq!(request.params.commands.len(), 2);

        let rendered = serde_json::to_value(&request).unwrap();
        assert_eq!(rendered["method"], "scene/preview");
        assert_eq!(rendered["params"]["commands"][0]["kind"], "spawn");
    }

    #[test]
    fn rejects_empty_transforms_duplicate_properties_and_unknown_fields() {
        let empty = SceneOperation::SetTransform {
            object_id: "object-1".to_owned(),
            local_transform: TransformPatch::default(),
        };
        assert!(empty.validate().unwrap_err().contains("at least one"));

        let duplicate = SceneOperation::SetProperties {
            object_id: "object-1".to_owned(),
            component_id: "object-1/light".to_owned(),
            changes: vec![
                PropertyChange {
                    property_id: "intensity".to_owned(),
                    value: Value::from(1),
                },
                PropertyChange {
                    property_id: "intensity".to_owned(),
                    value: Value::from(2),
                },
            ],
        };
        assert!(duplicate.validate().unwrap_err().contains("duplicate"));

        let extra = r#"{"jsonrpc":"2.0","id":"1","method":"scene/apply","deadlineUnixMs":1,"params":{"previewToken":"p","expectedRevision":0,"clientMutationId":"m","extra":true}}"#;
        assert!(serde_json::from_str::<RpcRequest<SceneApplyParams>>(extra).is_err());

        let empty_spawn_transform = SceneOperation::Spawn {
            temp_id: "tmp:new".to_owned(),
            prefab_id: "prefab-1".to_owned(),
            parent_id: None,
            local_transform: Some(TransformPatch::default()),
        };
        assert!(empty_spawn_transform.validate().is_err());

        let empty_temp_id = SceneOperation::Spawn {
            temp_id: "tmp:".to_owned(),
            prefab_id: "prefab-1".to_owned(),
            parent_id: None,
            local_transform: None,
        };
        assert!(empty_temp_id.validate().is_err());
    }

    #[test]
    fn rejects_values_and_registration_metadata_outside_the_schema() {
        let too_deep = SceneOperation::SetProperties {
            object_id: "object-1".to_owned(),
            component_id: "object-1/adapter".to_owned(),
            changes: vec![PropertyChange {
                property_id: "items".to_owned(),
                value: serde_json::json!([[1]]),
            }],
        };
        assert!(too_deep.validate().unwrap_err().contains("nested"));

        let fixture = include_str!("../../../schemas/scene-rpc/v1/examples/runtime-register.json");
        let mut registration: RuntimeRegistrationNotification =
            serde_json::from_str(fixture).unwrap();
        registration
            .params
            .capabilities
            .push(SceneCapability::SceneRead);
        assert!(registration.validate().unwrap_err().contains("unique"));
    }

    #[test]
    fn runtime_registration_fixture_is_strict_and_supported() {
        let fixture = include_str!("../../../schemas/scene-rpc/v1/examples/runtime-register.json");
        let registration: RuntimeRegistrationNotification = serde_json::from_str(fixture).unwrap();
        registration.validate().unwrap();
        assert!(
            registration
                .params
                .capabilities
                .contains(&SceneCapability::SceneRead)
        );
    }
}
