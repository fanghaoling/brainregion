use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{BrainregiondError, Result};

pub const DEFAULT_MCP_PROTOCOL_VERSION: &str = "2025-11-25";
pub const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
pub const DEFAULT_HEALTH_TIMEOUT: Duration = Duration::from_secs(3);
pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DaemonConfig {
    pub mode: RunMode,
    pub mcp: McpProcessConfig,
    pub startup_timeout: Duration,
    pub health_timeout: Duration,
    pub request_timeout: Duration,
    pub mcp_protocol_version: String,
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

        Ok(Self {
            mode: RunMode::Serve,
            mcp,
            startup_timeout,
            health_timeout,
            request_timeout,
            mcp_protocol_version,
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
provided as a JSON string array in BRAINREGIOND_MCP_ARGS_JSON."
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
}
