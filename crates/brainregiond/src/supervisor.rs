use std::future::Future;
use std::time::Duration;

use rmcp::model::{
    CallToolRequest, CallToolRequestParams, CallToolResult, ClientInfo, ClientRequest,
    Implementation, ProtocolVersion, ServerResult, Tool,
};
use rmcp::service::{PeerRequestOptions, RunningService};
use rmcp::{RoleClient, ServiceExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::process::Command;

use crate::child_transport::{ChildController, SupervisedChildTransport};
use crate::config::{DaemonConfig, McpProcessConfig};
use crate::error::{BrainregiondError, Result};
use crate::{DAEMON_NAME, DAEMON_VERSION};

type ClientService = RunningService<RoleClient, ClientInfo>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum McpState {
    Starting,
    Ready,
    Degraded,
    Stopped,
}

impl McpState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Degraded => "degraded",
            Self::Stopped => "stopped",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BrainregionPing {
    pub ok: bool,
    pub name: String,
    #[serde(default)]
    pub legacy_name: Option<String>,
    pub version: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct McpMetadata {
    pub protocol_version: String,
    pub server_info: Value,
    pub capabilities: Value,
    pub tool_count: usize,
    pub brainregion: BrainregionPing,
}

/// Owns the official MCP client service and the Python child transport.
pub struct McpSupervisor {
    service: ClientService,
    state: McpState,
    metadata: McpMetadata,
    tools: Vec<Tool>,
    health_timeout: Duration,
    request_timeout: Duration,
    shutdown_incomplete: bool,
    child: ChildController,
}

impl McpSupervisor {
    pub async fn connect(config: &DaemonConfig) -> Result<Self> {
        let (process, child) = spawn_transport(&config.mcp).await?;
        let protocol_version: ProtocolVersion =
            serde_json::from_value(Value::String(config.mcp_protocol_version.clone()))?;
        let mut client_info = ClientInfo::default();
        client_info.protocol_version = protocol_version;
        client_info.client_info = Implementation::new(DAEMON_NAME, DAEMON_VERSION);

        let service_result = with_timeout(
            config.startup_timeout,
            "MCP initialization",
            client_info.serve(process),
        )
        .await;
        let mut service = match service_result {
            Ok(service) => service,
            Err(error) => {
                let cleanup = child.force_reap().await.map_err(BrainregiondError::Io);
                return Err(combine_primary_and_cleanup(error, cleanup));
            }
        };

        let bootstrap = bootstrap_service(&service, config).await;
        let (metadata, tools) = match bootstrap {
            Ok(ready) => ready,
            Err(error) => {
                let cleanup = close_service(&mut service, &child).await;
                return Err(combine_primary_and_cleanup(error, cleanup));
            }
        };

        Ok(Self {
            service,
            state: McpState::Ready,
            metadata,
            tools,
            health_timeout: config.health_timeout,
            request_timeout: config.request_timeout,
            shutdown_incomplete: false,
            child,
        })
    }

    async fn call_tool_result(
        &mut self,
        name: &str,
        arguments: Map<String, Value>,
    ) -> Result<CallToolResult> {
        match call_tool(&self.service, name, arguments, self.request_timeout).await {
            Ok(result) => Ok(result),
            Err(error) => {
                self.state = McpState::Degraded;
                Err(error)
            }
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        if self.shutdown_incomplete {
            return Err(BrainregiondError::Protocol(
                "a previous MCP shutdown timed out; child cleanup is not confirmed".to_owned(),
            ));
        }
        let result = close_service(&mut self.service, &self.child).await;
        match result {
            Ok(()) => {
                self.state = McpState::Stopped;
                Ok(())
            }
            Err(error) => {
                self.state = McpState::Degraded;
                self.shutdown_incomplete = true;
                Err(error)
            }
        }
    }
}

pub(crate) trait McpBackend {
    fn state(&self) -> McpState;
    fn metadata(&self) -> &McpMetadata;
    async fn list_tools_value(&mut self) -> Result<Value>;
    async fn call_tool_value(&mut self, name: &str, arguments: Map<String, Value>)
    -> Result<Value>;
    async fn ping(&mut self) -> Result<BrainregionPing>;
    async fn shutdown(&mut self) -> Result<()>;
}

impl McpBackend for McpSupervisor {
    fn state(&self) -> McpState {
        if self.state != McpState::Stopped && self.service.is_transport_closed() {
            McpState::Degraded
        } else {
            self.state
        }
    }

    fn metadata(&self) -> &McpMetadata {
        &self.metadata
    }

    async fn list_tools_value(&mut self) -> Result<Value> {
        if self.service.is_transport_closed() {
            self.state = McpState::Degraded;
            return Err(BrainregiondError::Protocol(
                "MCP transport is closed; cached tool metadata is stale".to_owned(),
            ));
        }
        Ok(json!({"tools": self.tools}))
    }

    async fn call_tool_value(
        &mut self,
        name: &str,
        arguments: Map<String, Value>,
    ) -> Result<Value> {
        let result = self.call_tool_result(name, arguments).await?;
        Ok(serde_json::to_value(result)?)
    }

    async fn ping(&mut self) -> Result<BrainregionPing> {
        let result = match call_tool(&self.service, "ping", Map::new(), self.health_timeout).await {
            Ok(result) => result,
            Err(error) => {
                self.state = McpState::Degraded;
                return Err(error);
            }
        };
        match parse_ping(&result) {
            Ok(ping) => {
                self.state = McpState::Ready;
                self.metadata.brainregion = ping.clone();
                Ok(ping)
            }
            Err(error) => {
                self.state = McpState::Degraded;
                Err(error)
            }
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        McpSupervisor::shutdown(self).await
    }
}

async fn spawn_transport(
    config: &McpProcessConfig,
) -> Result<(SupervisedChildTransport, ChildController)> {
    let mut command = Command::new(&config.program);
    command.args(&config.args);
    if let Some(cwd) = &config.cwd {
        command.current_dir(cwd);
    }
    SupervisedChildTransport::spawn(command)
        .await
        .map_err(BrainregiondError::Io)
}

async fn bootstrap_service(
    service: &ClientService,
    config: &DaemonConfig,
) -> Result<(McpMetadata, Vec<Tool>)> {
    let (negotiated_protocol, server_info, capabilities) = {
        let peer_info = service.peer().peer_info().ok_or_else(|| {
            BrainregiondError::Protocol("MCP initialize returned no server info".to_owned())
        })?;
        (
            peer_info.protocol_version.to_string(),
            serde_json::to_value(&peer_info.server_info)?,
            serde_json::to_value(&peer_info.capabilities)?,
        )
    };
    if negotiated_protocol != config.mcp_protocol_version {
        return Err(BrainregiondError::Protocol(format!(
            "MCP protocol mismatch: requested {}, negotiated {negotiated_protocol}",
            config.mcp_protocol_version
        )));
    }

    let tools = with_timeout(
        config.startup_timeout,
        "MCP tools/list",
        service.peer().list_all_tools(),
    )
    .await?;
    if !tools.iter().any(|tool| tool.name.as_ref() == "ping") {
        return Err(BrainregiondError::Protocol(
            "BrainRegion MCP does not advertise the required ping tool".to_owned(),
        ));
    }

    let ping_result = call_tool(service, "ping", Map::new(), config.health_timeout).await?;
    let brainregion = parse_ping(&ping_result)?;
    let metadata = McpMetadata {
        protocol_version: negotiated_protocol,
        server_info,
        capabilities,
        tool_count: tools.len(),
        brainregion,
    };
    if metadata
        .capabilities
        .pointer("/tools/listChanged")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err(BrainregiondError::Protocol(
            "MCP tools.listChanged=true is not supported by the v1 cached tool registry".to_owned(),
        ));
    }
    Ok((metadata, tools))
}

async fn close_service(service: &mut ClientService, child: &ChildController) -> Result<()> {
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(7);
    let service_result = match service.close_with_timeout(SHUTDOWN_TIMEOUT).await {
        Ok(Some(_reason)) => Ok(()),
        Ok(None) => Err(BrainregiondError::Timeout {
            operation: "MCP shutdown".to_owned(),
            timeout: SHUTDOWN_TIMEOUT,
        }),
        Err(error) => Err(BrainregiondError::Protocol(format!(
            "MCP shutdown task failed: {error}"
        ))),
    };
    let child_result = child.force_reap().await.map_err(BrainregiondError::Io);
    match (service_result, child_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(child_error)) => Err(BrainregiondError::Protocol(format!(
            "{error}; forced child cleanup also failed: {child_error}"
        ))),
    }
}

fn combine_primary_and_cleanup(
    primary: BrainregiondError,
    cleanup: Result<()>,
) -> BrainregiondError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup_error) => BrainregiondError::Protocol(format!(
            "{primary}; MCP cleanup also failed: {cleanup_error}"
        )),
    }
}

async fn call_tool(
    service: &ClientService,
    name: &str,
    arguments: Map<String, Value>,
    timeout: Duration,
) -> Result<CallToolResult> {
    let params = CallToolRequestParams::new(name.to_owned()).with_arguments(arguments);
    let request = ClientRequest::CallToolRequest(CallToolRequest::new(params));
    let handle = service
        .peer()
        .send_cancellable_request(request, PeerRequestOptions::with_timeout(timeout))
        .await
        .map_err(|error| {
            BrainregiondError::Protocol(format!("MCP tools/call {name} failed to start: {error}"))
        })?;
    let result = handle.await_response().await.map_err(|error| match error {
        rmcp::ServiceError::Timeout { .. } => BrainregiondError::Timeout {
            operation: format!("MCP tools/call {name}"),
            timeout,
        },
        other => BrainregiondError::Protocol(format!("MCP tools/call {name} failed: {other}")),
    })?;
    match result {
        ServerResult::CallToolResult(result) => Ok(result),
        other => Err(BrainregiondError::Protocol(format!(
            "MCP tools/call {name} returned an unsupported response: {other:?}"
        ))),
    }
}

async fn with_timeout<F, T, E>(timeout: Duration, operation: &str, future: F) -> Result<T>
where
    F: Future<Output = std::result::Result<T, E>>,
    E: std::fmt::Display,
{
    match tokio::time::timeout(timeout, future).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(BrainregiondError::Protocol(format!(
            "{operation} failed: {error}"
        ))),
        Err(_) => Err(BrainregiondError::Timeout {
            operation: operation.to_owned(),
            timeout,
        }),
    }
}

pub fn parse_ping(result: &CallToolResult) -> Result<BrainregionPing> {
    if result.is_error == Some(true) {
        return Err(BrainregiondError::Upstream(serde_json::to_value(result)?));
    }

    let payload = if let Some(structured) = &result.structured_content {
        structured.clone()
    } else {
        let text = result
            .content
            .iter()
            .find_map(|block| block.as_text())
            .ok_or_else(|| {
                BrainregiondError::Protocol(
                    "BrainRegion ping returned neither structured content nor text".to_owned(),
                )
            })?;
        serde_json::from_str(&text.text).map_err(|error| {
            BrainregiondError::Protocol(format!("BrainRegion ping text is not valid JSON: {error}"))
        })?
    };
    let ping: BrainregionPing = serde_json::from_value(payload).map_err(|error| {
        BrainregiondError::Protocol(format!("BrainRegion ping payload is invalid: {error}"))
    })?;

    if !ping.ok {
        return Err(BrainregiondError::Protocol(
            "BrainRegion ping reported ok=false".to_owned(),
        ));
    }
    if ping.name != "brainregion" {
        return Err(BrainregiondError::Protocol(format!(
            "unexpected BrainRegion ping name {:?}",
            ping.name
        )));
    }
    if !looks_like_semver(&ping.version) {
        return Err(BrainregiondError::Protocol(format!(
            "BrainRegion ping version is not semantic-version shaped: {:?}",
            ping.version
        )));
    }
    Ok(ping)
}

fn looks_like_semver(version: &str) -> bool {
    let core = version
        .split_once('+')
        .map_or(version, |(core, _)| core)
        .split_once('-')
        .map_or_else(
            || version.split_once('+').map_or(version, |(core, _)| core),
            |(core, _)| core,
        );
    let mut parts = core.split('.');
    let valid_part =
        |part: &str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit());
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some(major), Some(minor), Some(patch), None)
            if valid_part(major) && valid_part(minor) && valid_part(patch)
    )
}

#[cfg(test)]
mod tests {
    use rmcp::model::{CallToolResult, ContentBlock};

    use super::*;

    #[test]
    fn parses_current_fastmcp_text_ping() {
        let result = CallToolResult::success(vec![ContentBlock::text(
            r#"{"ok":true,"name":"brainregion","legacy_name":"brain_region","version":"0.2.0"}"#,
        )]);

        let ping = parse_ping(&result).unwrap();
        assert_eq!(ping.name, "brainregion");
        assert_eq!(ping.version, "0.2.0");
    }

    #[test]
    fn parses_future_structured_ping() {
        let result = CallToolResult::structured(json!({
            "ok": true,
            "name": "brainregion",
            "version": "1.0.0-beta.1"
        }));

        assert_eq!(parse_ping(&result).unwrap().version, "1.0.0-beta.1");
    }

    #[test]
    fn rejects_tool_errors_and_wrong_identity() {
        let tool_error = CallToolResult::error(vec![ContentBlock::text("unavailable")]);
        assert!(matches!(
            parse_ping(&tool_error),
            Err(BrainregiondError::Upstream(_))
        ));

        let wrong_name = CallToolResult::structured(json!({
            "ok": true,
            "name": "other",
            "version": "1.0.0"
        }));
        assert!(
            parse_ping(&wrong_name)
                .unwrap_err()
                .to_string()
                .contains("unexpected")
        );
    }

    #[test]
    fn validates_semver_shape() {
        assert!(looks_like_semver("0.2.0"));
        assert!(looks_like_semver("1.2.3-beta.1+build.9"));
        assert!(!looks_like_semver("1.2"));
        assert!(!looks_like_semver("latest"));
    }
}
