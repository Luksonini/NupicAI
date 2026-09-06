use reqwest::{header, multipart, Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ApiError {
    pub status: Option<u16>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoginResult {
    pub user: Value,
    pub token: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptResult {
    pub transcript: String,
    pub detected_language: String,
}

#[derive(Debug, Deserialize)]
struct JobResponse {
    status: String,
    #[serde(default)]
    message: String,
    result: Option<Value>,
    error: Option<String>,
}

fn client() -> Result<Client, ApiError> {
    Client::builder()
        .user_agent("NupicAI-Flow-Tauri/0.1")
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(network_error)
}

fn endpoint(server_url: &str, path: &str) -> String {
    format!("{}{}", server_url.trim().trim_end_matches('/'), path)
}

fn cookie(token: &str) -> String {
    format!("nupicai_session={token}")
}

fn network_error(error: reqwest::Error) -> ApiError {
    ApiError {
        status: error.status().map(|status| status.as_u16()),
        message: format!("Nie można połączyć się z serwerem NupicAI: {error}"),
    }
}

async fn checked(response: Response) -> Result<Response, ApiError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status = response.status().as_u16();
    let fallback = response
        .status()
        .canonical_reason()
        .unwrap_or("Błąd serwera")
        .to_string();
    let body = response.json::<Value>().await.unwrap_or(Value::Null);
    let message = body
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or(&fallback);
    Err(ApiError {
        status: Some(status),
        message: format!("NupicAI HTTP {status}: {message}"),
    })
}

pub async fn login(server_url: &str, email: &str, password: &str) -> Result<LoginResult, ApiError> {
    let response = checked(
        client()?
            .post(endpoint(server_url, "/auth/login"))
            .json(&json!({"email": email, "password": password}))
            .send()
            .await
            .map_err(network_error)?,
    )
    .await?;
    let token = response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .find_map(|value| {
            value.split(';').find_map(|part| {
                part.trim()
                    .strip_prefix("nupicai_session=")
                    .map(str::to_owned)
            })
        })
        .ok_or_else(|| ApiError {
            status: None,
            message: "Serwer nie zwrócił sesji logowania".into(),
        })?;
    let body = response.json::<Value>().await.map_err(network_error)?;
    Ok(LoginResult {
        user: body.get("user").cloned().unwrap_or(Value::Null),
        token,
    })
}

pub async fn current_user(server_url: &str, token: &str) -> Result<Value, ApiError> {
    let response = checked(
        client()?
            .get(endpoint(server_url, "/auth/me"))
            .header(header::COOKIE, cookie(token))
            .send()
            .await
            .map_err(network_error)?,
    )
    .await?;
    let body = response.json::<Value>().await.map_err(network_error)?;
    Ok(body.get("user").cloned().unwrap_or(Value::Null))
}

pub async fn logout(server_url: &str, token: &str) {
    if let Ok(client) = client() {
        let _ = client
            .post(endpoint(server_url, "/auth/logout"))
            .header(header::COOKIE, cookie(token))
            .send()
            .await;
    }
}

pub async fn transcribe(
    server_url: &str,
    token: &str,
    wav: Vec<u8>,
) -> Result<TranscriptResult, ApiError> {
    let part = multipart::Part::bytes(wav)
        .file_name("dictation.wav")
        .mime_str("audio/wav")
        .map_err(|error| ApiError {
            status: None,
            message: error.to_string(),
        })?;
    let response = checked(
        client()?
            .post(endpoint(server_url, "/dictation/transcribe"))
            .header(header::COOKIE, cookie(token))
            .multipart(multipart::Form::new().part("file", part))
            .send()
            .await
            .map_err(network_error)?,
    )
    .await?;
    let body = response.json::<Value>().await.map_err(network_error)?;
    let job_id = body
        .get("job_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError {
            status: None,
            message: "Serwer nie zwrócił identyfikatora transkrypcji".into(),
        })?;
    let started = Instant::now();
    loop {
        if started.elapsed() > Duration::from_secs(120) {
            return Err(ApiError {
                status: None,
                message: "Przekroczono czas transkrypcji".into(),
            });
        }
        tokio::time::sleep(Duration::from_millis(180)).await;
        let response = checked(
            client()?
                .get(endpoint(server_url, &format!("/jobs/{job_id}")))
                .header(header::COOKIE, cookie(token))
                .send()
                .await
                .map_err(network_error)?,
        )
        .await?;
        let job = response
            .json::<JobResponse>()
            .await
            .map_err(network_error)?;
        match job.status.as_str() {
            "done" => {
                let result = job.result.unwrap_or(Value::Null);
                return Ok(TranscriptResult {
                    transcript: result
                        .get("transcript")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                    detected_language: result
                        .get("detected_language")
                        .and_then(Value::as_str)
                        .unwrap_or("auto")
                        .to_string(),
                });
            }
            "error" => {
                return Err(ApiError {
                    status: None,
                    message: job
                        .error
                        .unwrap_or_else(|| "Transkrypcja nie powiodła się".into()),
                });
            }
            _ => {
                let _ = job.message;
            }
        }
    }
}

pub async fn polish(
    server_url: &str,
    token: &str,
    text: &str,
    language: &str,
) -> Result<String, ApiError> {
    let response = checked(
        client()?
            .post(endpoint(server_url, "/dictation/polish"))
            .header(header::COOKIE, cookie(token))
            .json(&json!({"text": text, "language": language}))
            .send()
            .await
            .map_err(network_error)?,
    )
    .await?;
    let body = response.json::<Value>().await.map_err(network_error)?;
    Ok(body
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or(text)
        .trim()
        .to_string())
}
