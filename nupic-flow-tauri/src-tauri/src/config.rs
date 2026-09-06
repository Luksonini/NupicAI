use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

const SERVICE_NAME: &str = "ai.nupic.flow";
const SESSION_ACCOUNT: &str = "nupicai-session";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub server_url: String,
    pub email: String,
    pub input_source: String,
    pub input_device: String,
    pub polish: bool,
    pub auto_paste: bool,
    pub shortcut: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            server_url: "http://127.0.0.1:8765".into(),
            email: String::new(),
            input_source: "microphone".into(),
            input_device: String::new(),
            polish: false,
            auto_paste: true,
            shortcut: "Ctrl+Alt+Space".into(),
        }
    }
}

fn config_path() -> Result<PathBuf, String> {
    ProjectDirs::from("ai", "NupicAI", "Flow")
        .map(|dirs| dirs.config_dir().join("config.json"))
        .ok_or_else(|| "Nie można odnaleźć katalogu konfiguracji użytkownika".into())
}

fn legacy_config_path() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|dirs| dirs.config_dir().join("nupicai-flow/config.json"))
}

pub fn load() -> AppSettings {
    config_path()
        .ok()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .or_else(|| {
            legacy_config_path()
                .and_then(|path| fs::read_to_string(path).ok())
                .and_then(|raw| serde_json::from_str(&raw).ok())
        })
        .unwrap_or_default()
}

pub fn save(settings: &AppSettings) -> Result<(), String> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let body = serde_json::to_vec_pretty(settings).map_err(|error| error.to_string())?;
    fs::write(&path, body).map_err(|error| error.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE_NAME, SESSION_ACCOUNT).map_err(|error| error.to_string())
}

pub fn load_session() -> Option<String> {
    let secure = entry()
        .ok()
        .and_then(|entry| entry.get_password().ok())
        .filter(|token| !token.is_empty());
    if secure.is_some() {
        return secure;
    }
    let legacy = legacy_config_path()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("session_token")
                .and_then(|token| token.as_str())
                .map(str::to_owned)
        })
        .filter(|token| !token.is_empty());
    if let Some(token) = legacy.as_deref() {
        let _ = save_session(token);
    }
    legacy
}

pub fn save_session(token: &str) -> Result<(), String> {
    entry()?
        .set_password(token)
        .map_err(|error| error.to_string())
}

pub fn clear_session() {
    if let Ok(entry) = entry() {
        let _ = entry.delete_credential();
    }
}
