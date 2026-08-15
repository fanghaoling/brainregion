//! Headless control plane for BrainRegion clients.
//!
//! The first vertical slice intentionally keeps the existing Python MCP server as
//! the domain/tool worker. This crate owns process lifetime and a small, stable
//! JSON-RPC control protocol that desktop and VR transports can build on.

mod child_transport;
pub mod config;
pub mod error;
pub mod protocol;
pub mod scene_rpc;
pub mod server;
pub mod supervisor;

pub use error::{BrainregiondError, Result};

pub const CONTROL_PROTOCOL_VERSION: &str = "brainregion.control.v1";
pub const CONTROL_SCHEMA_JSON: &str =
    include_str!("../../../schemas/agent-core/v1/control-message.schema.json");
pub const SCENE_SCHEMA_JSON: &str =
    include_str!("../../../schemas/scene-rpc/v1/scene-message.schema.json");
pub const DAEMON_NAME: &str = "brainregiond";
pub const DAEMON_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn embedded_control_schema_is_valid_json() {
        let schema: serde_json::Value = serde_json::from_str(super::CONTROL_SCHEMA_JSON).unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }

    #[test]
    fn embedded_scene_schema_is_valid_json() {
        let schema: serde_json::Value = serde_json::from_str(super::SCENE_SCHEMA_JSON).unwrap();
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
    }
}
