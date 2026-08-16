use std::fmt;
use std::io;
use std::time::Duration;

/// Errors surfaced by the control plane.
#[derive(Debug)]
pub enum BrainregiondError {
    Config(String),
    Io(io::Error),
    Json(serde_json::Error),
    Protocol(String),
    Timeout {
        operation: String,
        timeout: Duration,
    },
    Upstream(serde_json::Value),
}

impl fmt::Display for BrainregiondError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(formatter, "configuration error: {message}"),
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::Protocol(message) => write!(formatter, "protocol error: {message}"),
            Self::Timeout { operation, timeout } => write!(
                formatter,
                "operation {operation:?} timed out after {} ms",
                timeout.as_millis()
            ),
            Self::Upstream(error) => write!(formatter, "upstream MCP error: {error}"),
        }
    }
}

impl std::error::Error for BrainregiondError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for BrainregiondError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for BrainregiondError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub type Result<T> = std::result::Result<T, BrainregiondError>;
