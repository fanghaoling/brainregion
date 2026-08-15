use std::io::{self, BufRead};

use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt};

pub const MAX_CONTROL_FRAME_BYTES: usize = 1024 * 1024;
pub const MAX_CONTROL_OUTPUT_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub id: Value,
    pub method: String,
    pub params: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RpcFault {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestParseError {
    pub id: Value,
    pub fault: RpcFault,
}

impl RequestParseError {
    fn new(id: Value, fault: RpcFault) -> Self {
        Self { id, fault }
    }
}

impl RpcFault {
    pub fn parse(message: impl Into<String>) -> Self {
        Self {
            code: -32700,
            message: message.into(),
            data: None,
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: message.into(),
            data: None,
        }
    }

    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
            data: None,
        }
    }

    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
            data: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: -32000,
            message: message.into(),
            data: None,
        }
    }
}

pub fn parse_request(line: &str) -> std::result::Result<Request, RequestParseError> {
    let value: Value = serde_json::from_str(line).map_err(|error| {
        RequestParseError::new(
            Value::Null,
            RpcFault::parse(format!("invalid JSON: {error}")),
        )
    })?;
    let object = value.as_object().ok_or_else(|| {
        RequestParseError::new(
            Value::Null,
            RpcFault::invalid_request("request must be a JSON object"),
        )
    })?;
    let response_id = object
        .get("id")
        .filter(|id| is_valid_id(id))
        .cloned()
        .unwrap_or(Value::Null);

    if let Some(field) = object
        .keys()
        .find(|field| !matches!(field.as_str(), "jsonrpc" | "id" | "method" | "params"))
    {
        return Err(RequestParseError::new(
            response_id,
            RpcFault::invalid_request(format!("unexpected request field {field:?}")),
        ));
    }

    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(RequestParseError::new(
            response_id,
            RpcFault::invalid_request("jsonrpc must be exactly \"2.0\""),
        ));
    }
    let id = object.get("id").cloned().ok_or_else(|| {
        RequestParseError::new(
            Value::Null,
            RpcFault::invalid_request("control requests require an id"),
        )
    })?;
    if !is_valid_id(&id) {
        return Err(RequestParseError::new(
            Value::Null,
            RpcFault::invalid_request("id must be a string, integer, or null"),
        ));
    }
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .filter(|method| !method.is_empty())
        .ok_or_else(|| {
            RequestParseError::new(
                id.clone(),
                RpcFault::invalid_request("method must be a non-empty string"),
            )
        })?
        .to_owned();
    let params = object.get("params").cloned().unwrap_or_else(|| json!({}));
    if !params.is_object() {
        return Err(RequestParseError::new(
            id,
            RpcFault::invalid_params("params must be an object"),
        ));
    }

    Ok(Request { id, method, params })
}

fn is_valid_id(id: &Value) -> bool {
    match id {
        Value::Null | Value::String(_) => true,
        Value::Number(number) => number.is_i64() || number.is_u64(),
        _ => false,
    }
}

pub fn success(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

pub fn failure(id: Value, fault: RpcFault) -> Value {
    let mut error = json!({"code": fault.code, "message": fault.message});
    if let Some(data) = fault.data {
        error["data"] = data;
    }
    json!({"jsonrpc": "2.0", "id": id, "error": error})
}

pub fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "method": method, "params": params})
}

/// Read one UTF-8 JSONL frame without allowing an unbounded allocation.
pub fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    max_bytes: usize,
) -> io::Result<Option<String>> {
    let mut bytes = Vec::new();

    loop {
        let (take_len, found_newline, available_len) = {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                if bytes.is_empty() {
                    return Ok(None);
                }
                break;
            }
            let newline = available.iter().position(|byte| *byte == b'\n');
            (
                newline.map_or(available.len(), |position| position + 1),
                newline.is_some(),
                available.len(),
            )
        };

        if bytes.len().saturating_add(take_len) > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSONL frame exceeds {max_bytes} bytes"),
            ));
        }

        {
            let available = reader.fill_buf()?;
            bytes.extend_from_slice(&available[..take_len]);
        }
        reader.consume(take_len);

        if found_newline {
            break;
        }
        debug_assert_eq!(take_len, available_len);
    }

    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

/// Async counterpart used by the daemon's stdin transport so termination
/// signals can interrupt an idle client while preserving the same size bound.
pub async fn read_bounded_line_async<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max_bytes: usize,
) -> io::Result<Option<String>> {
    let mut bytes = Vec::new();

    loop {
        let (take_len, found_newline, available_len) = {
            let available = reader.fill_buf().await?;
            if available.is_empty() {
                if bytes.is_empty() {
                    return Ok(None);
                }
                break;
            }
            let newline = available.iter().position(|byte| *byte == b'\n');
            (
                newline.map_or(available.len(), |position| position + 1),
                newline.is_some(),
                available.len(),
            )
        };

        if bytes.len().saturating_add(take_len) > max_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("JSONL frame exceeds {max_bytes} bytes"),
            ));
        }

        {
            let available = reader.fill_buf().await?;
            bytes.extend_from_slice(&available[..take_len]);
        }
        reader.consume(take_len);

        if found_newline {
            break;
        }
        debug_assert_eq!(take_len, available_len);
    }

    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn parses_a_control_request() {
        let request =
            parse_request(r#"{"jsonrpc":"2.0","id":"health-1","method":"daemon/health"}"#).unwrap();

        assert_eq!(request.id, json!("health-1"));
        assert_eq!(request.method, "daemon/health");
        assert_eq!(request.params, json!({}));
    }

    #[test]
    fn rejects_notifications_on_the_control_input() {
        let error = parse_request(r#"{"jsonrpc":"2.0","method":"daemon/health"}"#).unwrap_err();
        assert_eq!(error.fault.code, -32600);
        assert_eq!(error.id, Value::Null);
    }

    #[test]
    fn rejects_non_integer_ids_extra_fields_and_non_object_params() {
        for input in [
            r#"{"jsonrpc":"2.0","id":1.5,"method":"daemon/health"}"#,
            r#"{"jsonrpc":"2.0","id":1,"method":"daemon/health","extra":true}"#,
            r#"{"jsonrpc":"2.0","id":1,"method":"daemon/health","params":[]}"#,
        ] {
            assert!(
                parse_request(input).is_err(),
                "unexpectedly accepted {input}"
            );
        }
    }

    #[test]
    fn preserves_valid_ids_for_correlated_request_errors() {
        let error = parse_request(
            r#"{"jsonrpc":"2.0","id":"bad-params-1","method":"daemon/health","params":[]}"#,
        )
        .unwrap_err();

        assert_eq!(error.id, "bad-params-1");
        assert_eq!(error.fault.code, -32602);
    }

    #[test]
    fn bounded_reader_handles_crlf_and_eof() {
        let mut input = Cursor::new(b"one\r\ntwo".to_vec());
        assert_eq!(
            read_bounded_line(&mut input, 16).unwrap().as_deref(),
            Some("one")
        );
        assert_eq!(
            read_bounded_line(&mut input, 16).unwrap().as_deref(),
            Some("two")
        );
        assert_eq!(read_bounded_line(&mut input, 16).unwrap(), None);
    }

    #[test]
    fn bounded_reader_rejects_large_frames() {
        let mut input = Cursor::new(b"12345\n".to_vec());
        let error = read_bounded_line(&mut input, 4).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn async_bounded_reader_matches_sync_contract() {
        let mut input = Cursor::new(b"one\r\ntwo".to_vec());
        assert_eq!(
            read_bounded_line_async(&mut input, 16)
                .await
                .unwrap()
                .as_deref(),
            Some("one")
        );
        assert_eq!(
            read_bounded_line_async(&mut input, 16)
                .await
                .unwrap()
                .as_deref(),
            Some("two")
        );
        assert_eq!(read_bounded_line_async(&mut input, 16).await.unwrap(), None);
    }
}
