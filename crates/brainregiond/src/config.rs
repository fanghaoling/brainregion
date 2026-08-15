use std::env;
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{BrainregiondError, Result};
use crate::scene_rpc::SceneCapability;

pub const DEFAULT_MCP_PROTOCOL_VERSION: &str = "2025-11-25";
pub const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
pub const DEFAULT_HEALTH_TIMEOUT: Duration = Duration::from_secs(3);
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
pub const DEFAULT_SCENE_PIPE_AUTH_TIMEOUT: Duration = Duration::from_secs(10);
pub const DEFAULT_SCENE_PIPE_MAX_CONNECTIONS: usize = 4;

const MIN_PAIRING_SECRET_BYTES: usize = 32;
const MAX_PAIRING_SECRET_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunMode {
    Serve,
    Probe,
    Schema,
    SceneSchema,
    Help,
    Version,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct McpProcessConfig {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: Option<PathBuf>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct PairingSecret(Vec<u8>);

impl PairingSecret {
    pub fn new(secret: impl AsRef<[u8]>) -> Result<Self> {
        let secret = secret.as_ref();
        if !(MIN_PAIRING_SECRET_BYTES..=MAX_PAIRING_SECRET_BYTES).contains(&secret.len()) {
            return Err(BrainregiondError::Config(format!(
                "scene pairing secret must contain {MIN_PAIRING_SECRET_BYTES}..{MAX_PAIRING_SECRET_BYTES} bytes"
            )));
        }
        Ok(Self(secret.to_vec()))
    }

    pub(crate) fn expose(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Debug for PairingSecret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PairingSecret([REDACTED])")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScenePipeConfig {
    pub name: String,
    pub principal_id: String,
    pub pairing_secret: PairingSecret,
    pub granted_capabilities: Vec<SceneCapability>,
    pub max_connections: usize,
    pub authentication_timeout: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DaemonConfig {
    pub mode: RunMode,
    pub mcp: McpProcessConfig,
    pub startup_timeout: Duration,
    pub health_timeout: Duration,
    pub request_timeout: Duration,
    pub mcp_protocol_version: String,
    pub scene_pipe: Option<ScenePipeConfig>,
}

impl DaemonConfig {
    pub fn discover() -> Result<Self> {
        let current_dir = env::current_dir()?;
        let mut mcp = discover_mcp_process(&current_dir);

        if let Some(program) = env::var_os("BRAINREGIOND_MCP_PROGRAM") {
            if program.is_empty() {
                return Err(BrainregiondError::Config(
                    "BRAINREGIOND_MCP_PROGRAM cannot be empty".to_owned(),
                ));
            }
            mcp.program = PathBuf::from(program);
            mcp.args.clear();
        }

        match env::var("BRAINREGIOND_MCP_ARGS_JSON") {
            Ok(raw_args) => {
                let args: Vec<String> = serde_json::from_str(&raw_args).map_err(|error| {
                    BrainregiondError::Config(format!(
                        "BRAINREGIOND_MCP_ARGS_JSON must be a JSON string array: {error}"
                    ))
                })?;
                mcp.args = args.into_iter().map(OsString::from).collect();
            }
            Err(env::VarError::NotPresent) => {}
            Err(error) => {
                return Err(BrainregiondError::Config(format!(
                    "BRAINREGIOND_MCP_ARGS_JSON is not valid Unicode: {error}"
                )));
            }
        }

        if let Some(cwd) = env::var_os("BRAINREGIOND_MCP_CWD") {
            mcp.cwd = Some(PathBuf::from(cwd));
        }

        let startup_timeout =
            timeout_from_env("BRAINREGIOND_STARTUP_TIMEOUT_MS", DEFAULT_STARTUP_TIMEOUT)?;
        let health_timeout =
            timeout_from_env("BRAINREGIOND_HEALTH_TIMEOUT_MS", DEFAULT_HEALTH_TIMEOUT)?;
        let request_timeout =
            timeout_from_env("BRAINREGIOND_REQUEST_TIMEOUT_MS", DEFAULT_REQUEST_TIMEOUT)?;

        let mcp_protocol_version = match env::var("BRAINREGIOND_MCP_PROTOCOL_VERSION") {
            Ok(value) => value,
            Err(env::VarError::NotPresent) => DEFAULT_MCP_PROTOCOL_VERSION.to_owned(),
            Err(error) => {
                return Err(BrainregiondError::Config(format!(
                    "BRAINREGIOND_MCP_PROTOCOL_VERSION is not valid Unicode: {error}"
                )));
            }
        };
        let scene_pipe = scene_pipe_from_environment()?;

        Ok(Self {
            mode: RunMode::Serve,
            mcp,
            startup_timeout,
            health_timeout,
            request_timeout,
            mcp_protocol_version,
            scene_pipe,
        })
    }

    pub fn apply_args<I, S>(&mut self, args: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: Into<OsString>,
    {
        let mut args = args.into_iter().map(Into::into).peekable();
        if let Some(first) = args.peek() {
            match first.to_string_lossy().as_ref() {
                "serve" => {
                    self.mode = RunMode::Serve;
                    args.next();
                }
                "probe" => {
                    self.mode = RunMode::Probe;
                    args.next();
                }
                "schema" => {
                    self.mode = RunMode::Schema;
                    args.next();
                }
                "scene-schema" => {
                    self.mode = RunMode::SceneSchema;
                    args.next();
                }
                "help" | "--help" | "-h" => {
                    self.mode = RunMode::Help;
                    args.next();
                }
                "version" | "--version" | "-V" => {
                    self.mode = RunMode::Version;
                    args.next();
                }
                _ => {}
            }
        }

        let mut explicit_mcp_args = false;
        while let Some(argument) = args.next() {
            let argument_text = argument.to_string_lossy();
            match argument_text.as_ref() {
                "--help" | "-h" => self.mode = RunMode::Help,
                "--version" | "-V" => self.mode = RunMode::Version,
                "--mcp-program" => {
                    self.mcp.program = PathBuf::from(next_value(&mut args, "--mcp-program")?);
                    if !explicit_mcp_args {
                        self.mcp.args.clear();
                    }
                }
                "--mcp-arg" => {
                    if !explicit_mcp_args {
                        self.mcp.args.clear();
                        explicit_mcp_args = true;
                    }
                    self.mcp.args.push(next_value(&mut args, "--mcp-arg")?);
                }
                "--mcp-cwd" => {
                    self.mcp.cwd = Some(PathBuf::from(next_value(&mut args, "--mcp-cwd")?));
                }
                "--request-timeout-ms" => {
                    let value = next_value(&mut args, "--request-timeout-ms")?;
                    self.request_timeout = parse_timeout(&value.to_string_lossy())?;
                }
                "--startup-timeout-ms" => {
                    let value = next_value(&mut args, "--startup-timeout-ms")?;
                    self.startup_timeout = parse_timeout(&value.to_string_lossy())?;
                }
                "--health-timeout-ms" => {
                    let value = next_value(&mut args, "--health-timeout-ms")?;
                    self.health_timeout = parse_timeout(&value.to_string_lossy())?;
                }
                "--mcp-protocol-version" => {
                    let value = next_value(&mut args, "--mcp-protocol-version")?;
                    self.mcp_protocol_version = value.to_string_lossy().into_owned();
                }
                unknown => {
                    return Err(BrainregiondError::Config(format!(
                        "unknown argument {unknown:?}; run brainregiond --help"
                    )));
                }
            }
        }

        if self.mcp.program.as_os_str().is_empty() {
            return Err(BrainregiondError::Config(
                "MCP program cannot be empty".to_owned(),
            ));
        }
        if self.mcp_protocol_version.trim().is_empty() {
            return Err(BrainregiondError::Config(
                "MCP protocol version cannot be empty".to_owned(),
            ));
        }
        Ok(())
    }
}

fn next_value<I>(args: &mut I, option: &str) -> Result<OsString>
where
    I: Iterator<Item = OsString>,
{
    args.next()
        .ok_or_else(|| BrainregiondError::Config(format!("{option} requires a following value")))
}

fn parse_timeout(value: &str) -> Result<Duration> {
    let milliseconds = value.parse::<u64>().map_err(|_| {
        BrainregiondError::Config(format!("request timeout must be an integer, got {value:?}"))
    })?;
    if milliseconds == 0 {
        return Err(BrainregiondError::Config(
            "request timeout must be greater than zero".to_owned(),
        ));
    }
    Ok(Duration::from_millis(milliseconds))
}

fn timeout_from_env(name: &str, default: Duration) -> Result<Duration> {
    match env::var(name) {
        Ok(value) => parse_timeout(&value),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(BrainregiondError::Config(format!(
            "{name} is not valid Unicode: {error}"
        ))),
    }
}

fn scene_pipe_from_environment() -> Result<Option<ScenePipeConfig>> {
    let name = match env::var("BRAINREGIOND_SCENE_PIPE_NAME") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(None),
        Err(error) => {
            return Err(BrainregiondError::Config(format!(
                "BRAINREGIOND_SCENE_PIPE_NAME is not valid Unicode: {error}"
            )));
        }
    };
    validate_pipe_name(&name)?;

    let secret = required_environment_value("BRAINREGIOND_SCENE_PAIRING_SECRET")?;
    let principal_id = match env::var("BRAINREGIOND_SCENE_PRINCIPAL_ID") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => "unity-local".to_owned(),
        Err(error) => {
            return Err(BrainregiondError::Config(format!(
                "BRAINREGIOND_SCENE_PRINCIPAL_ID is not valid Unicode: {error}"
            )));
        }
    };
    validate_scene_identifier("scene principal id", &principal_id, 128)?;

    let granted_capabilities = match env::var("BRAINREGIOND_SCENE_CAPABILITIES_JSON") {
        Ok(value) => parse_scene_capabilities(&value)?,
        Err(env::VarError::NotPresent) => vec![SceneCapability::SceneRead],
        Err(error) => {
            return Err(BrainregiondError::Config(format!(
                "BRAINREGIOND_SCENE_CAPABILITIES_JSON is not valid Unicode: {error}"
            )));
        }
    };
    let max_connections = usize_from_env(
        "BRAINREGIOND_SCENE_MAX_CONNECTIONS",
        DEFAULT_SCENE_PIPE_MAX_CONNECTIONS,
        1,
        32,
    )?;
    let authentication_timeout = timeout_from_env(
        "BRAINREGIOND_SCENE_AUTH_TIMEOUT_MS",
        DEFAULT_SCENE_PIPE_AUTH_TIMEOUT,
    )?;

    Ok(Some(ScenePipeConfig {
        name,
        principal_id,
        pairing_secret: PairingSecret::new(secret.as_bytes())?,
        granted_capabilities,
        max_connections,
        authentication_timeout,
    }))
}

fn required_environment_value(name: &str) -> Result<String> {
    match env::var(name) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) | Err(env::VarError::NotPresent) => Err(BrainregiondError::Config(format!(
            "{name} is required when BRAINREGIOND_SCENE_PIPE_NAME is set"
        ))),
        Err(error) => Err(BrainregiondError::Config(format!(
            "{name} is not valid Unicode: {error}"
        ))),
    }
}

fn parse_scene_capabilities(value: &str) -> Result<Vec<SceneCapability>> {
    let mut capabilities: Vec<SceneCapability> = serde_json::from_str(value).map_err(|error| {
        BrainregiondError::Config(format!(
            "BRAINREGIOND_SCENE_CAPABILITIES_JSON must be a JSON capability array: {error}"
        ))
    })?;
    capabilities.sort_by_key(|capability| match capability {
        SceneCapability::SceneRead => 0,
        SceneCapability::SceneWrite => 1,
        SceneCapability::SceneSpawn => 2,
        SceneCapability::SceneUndo => 3,
        SceneCapability::LogsRead => 4,
    });
    capabilities.dedup();
    Ok(capabilities)
}

fn usize_from_env(name: &str, default: usize, minimum: usize, maximum: usize) -> Result<usize> {
    let raw = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(default),
        Err(error) => {
            return Err(BrainregiondError::Config(format!(
                "{name} is not valid Unicode: {error}"
            )));
        }
    };
    let value = raw.parse::<usize>().map_err(|_| {
        BrainregiondError::Config(format!("{name} must be an integer, got {raw:?}"))
    })?;
    if !(minimum..=maximum).contains(&value) {
        return Err(BrainregiondError::Config(format!(
            "{name} must be within {minimum}..{maximum}"
        )));
    }
    Ok(value)
}

fn validate_pipe_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 128
        || name
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || b"._-".contains(&byte)))
    {
        return Err(BrainregiondError::Config(
            "scene pipe name must contain 1..128 ASCII letters, digits, '.', '_' or '-'".to_owned(),
        ));
    }
    Ok(())
}

fn validate_scene_identifier(name: &str, value: &str, maximum: usize) -> Result<()> {
    if value.is_empty()
        || value.len() > maximum
        || value
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || b"._:/-".contains(&byte)))
    {
        return Err(BrainregiondError::Config(format!(
            "{name} must contain 1..{maximum} supported ASCII characters"
        )));
    }
    Ok(())
}

fn discover_mcp_process(current_dir: &Path) -> McpProcessConfig {
    for directory in current_dir.ancestors() {
        let windows_python = directory.join(".venv").join("Scripts").join("python.exe");
        if windows_python.is_file() {
            return McpProcessConfig {
                program: windows_python,
                args: vec![OsString::from("-m"), OsString::from("brainregion.server")],
                cwd: Some(directory.to_path_buf()),
            };
        }

        let unix_python = directory.join(".venv").join("bin").join("python");
        if unix_python.is_file() {
            return McpProcessConfig {
                program: unix_python,
                args: vec![OsString::from("-m"), OsString::from("brainregion.server")],
                cwd: Some(directory.to_path_buf()),
            };
        }
    }

    McpProcessConfig {
        program: PathBuf::from("brain-region-mcp"),
        args: Vec::new(),
        cwd: Some(current_dir.to_path_buf()),
    }
}

pub fn usage() -> &'static str {
    "brainregiond [serve|probe|schema|scene-schema] [options]\n\
\n\
Commands:\n\
  serve                       Start the JSONL control protocol on stdin/stdout (default)\n\
  probe                       Initialize the MCP child, call ping, print JSON, and exit\n\
  schema                      Print the embedded control-protocol JSON Schema and exit\n\
  scene-schema                Print the embedded Runtime Scene RPC JSON Schema and exit\n\
\n\
Options:\n\
  --mcp-program PATH          MCP executable (default: local .venv Python or brain-region-mcp)\n\
  --mcp-arg VALUE             MCP argument; repeat to provide multiple arguments\n\
  --mcp-cwd PATH              Working directory for the MCP child\n\
  --startup-timeout-ms N      Initialize/list timeout in milliseconds (default: 20000)\n\
  --health-timeout-ms N       Application ping timeout in milliseconds (default: 3000)\n\
  --request-timeout-ms N      Tool-call timeout in milliseconds (default: 300000)\n\
  --mcp-protocol-version V    MCP initialize protocol version (default: 2025-11-25)\n\
  -h, --help                  Show this help\n\
  -V, --version               Show the daemon version\n\
\n\
Equivalent environment variables use the BRAINREGIOND_ prefix; MCP arguments are\n\
provided as a JSON string array in BRAINREGIOND_MCP_ARGS_JSON. The Windows Runtime\n\
pipe stays disabled unless BRAINREGIOND_SCENE_PIPE_NAME and its pairing secret are set."
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DaemonConfig {
        DaemonConfig {
            mode: RunMode::Serve,
            mcp: McpProcessConfig {
                program: PathBuf::from("brain-region-mcp"),
                args: Vec::new(),
                cwd: None,
            },
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            health_timeout: DEFAULT_HEALTH_TIMEOUT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            mcp_protocol_version: DEFAULT_MCP_PROTOCOL_VERSION.to_owned(),
            scene_pipe: None,
        }
    }

    #[test]
    fn parses_probe_and_explicit_child_command() {
        let mut config = test_config();
        config
            .apply_args([
                "probe",
                "--mcp-program",
                "python",
                "--mcp-arg",
                "-m",
                "--mcp-arg",
                "brainregion.server",
                "--request-timeout-ms",
                "2500",
            ])
            .unwrap();

        assert_eq!(config.mode, RunMode::Probe);
        assert_eq!(config.mcp.program, PathBuf::from("python"));
        assert_eq!(
            config.mcp.args,
            vec![OsString::from("-m"), OsString::from("brainregion.server")]
        );
        assert_eq!(config.request_timeout, Duration::from_millis(2500));
    }

    #[test]
    fn rejects_zero_timeout() {
        let mut config = test_config();
        let error = config
            .apply_args(["--request-timeout-ms", "0"])
            .unwrap_err();
        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn redacts_and_validates_pairing_secrets() {
        let secret = PairingSecret::new("x".repeat(32)).unwrap();
        assert_eq!(format!("{secret:?}"), "PairingSecret([REDACTED])");
        assert!(PairingSecret::new("too-short").is_err());
    }

    #[test]
    fn parses_capabilities_and_rejects_unsafe_pipe_names() {
        let capabilities =
            parse_scene_capabilities(r#"["logs.read","scene.read","scene.read"]"#).unwrap();
        assert_eq!(
            capabilities,
            vec![SceneCapability::SceneRead, SceneCapability::LogsRead]
        );
        assert!(validate_pipe_name("brainregion.scene.local").is_ok());
        assert!(validate_pipe_name(r"..\\remote\pipe").is_err());
    }
}
