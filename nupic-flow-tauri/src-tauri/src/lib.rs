mod api;
mod audio;
mod config;

use audio::{AudioRecorder, RecordingStatus};
use config::AppSettings;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use serde::Serialize;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_clipboard_manager::ClipboardExt;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

struct AppState {
    settings: Mutex<AppSettings>,
    recorder: Mutex<AudioRecorder>,
    processing: AtomicBool,
}

#[derive(Debug, Serialize)]
struct Bootstrap {
    settings: AppSettings,
    devices: Vec<String>,
    authenticated: bool,
    user: Option<Value>,
    auth_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FlowEvent {
    phase: String,
    message: String,
    transcript: Option<String>,
}

fn state_error() -> String {
    "Wewnętrzny stan aplikacji jest chwilowo niedostępny".into()
}

#[tauri::command]
async fn bootstrap(state: State<'_, AppState>) -> Result<Bootstrap, String> {
    let settings = state.settings.lock().map_err(|_| state_error())?.clone();
    let devices = AudioRecorder::devices().unwrap_or_default();
    let Some(token) = config::load_session() else {
        return Ok(Bootstrap {
            settings,
            devices,
            authenticated: false,
            user: None,
            auth_error: None,
        });
    };
    match api::current_user(&settings.server_url, &token).await {
        Ok(user) => Ok(Bootstrap {
            settings,
            devices,
            authenticated: true,
            user: Some(user),
            auth_error: None,
        }),
        Err(error) if matches!(error.status, Some(401 | 403)) => {
            config::clear_session();
            Ok(Bootstrap {
                settings,
                devices,
                authenticated: false,
                user: None,
                auth_error: Some("Sesja wygasła".into()),
            })
        }
        Err(error) => Ok(Bootstrap {
            settings,
            devices,
            authenticated: true,
            user: None,
            auth_error: Some(error.to_string()),
        }),
    }
}

#[tauri::command]
async fn login(
    state: State<'_, AppState>,
    server_url: String,
    email: String,
    password: String,
) -> Result<Value, String> {
    if email.trim().is_empty() || password.is_empty() {
        return Err("Podaj e-mail i hasło".into());
    }
    let result = api::login(&server_url, &email, &password)
        .await
        .map_err(|error| error.to_string())?;
    config::save_session(&result.token)?;
    let mut settings = state.settings.lock().map_err(|_| state_error())?;
    settings.server_url = server_url.trim().trim_end_matches('/').to_string();
    settings.email = email.trim().to_string();
    config::save(&settings)?;
    Ok(result.user)
}

#[tauri::command]
async fn logout(state: State<'_, AppState>) -> Result<(), String> {
    let settings = state.settings.lock().map_err(|_| state_error())?.clone();
    if let Some(token) = config::load_session() {
        api::logout(&settings.server_url, &token).await;
    }
    config::clear_session();
    Ok(())
}

#[tauri::command]
fn save_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    settings: AppSettings,
) -> Result<(), String> {
    register_shortcut(&app, &settings.shortcut)?;
    config::save(&settings)?;
    *state.settings.lock().map_err(|_| state_error())? = settings;
    Ok(())
}

#[tauri::command]
fn list_input_devices() -> Result<Vec<String>, String> {
    AudioRecorder::devices()
}

#[tauri::command]
fn copy_text(app: AppHandle, text: String) -> Result<(), String> {
    app.clipboard()
        .write_text(text)
        .map_err(|error| error.to_string())
}

fn start_recording_inner(state: &AppState) -> Result<(), String> {
    if state.processing.load(Ordering::Relaxed) {
        return Err("Poczekaj na zakończenie poprzedniej transkrypcji".into());
    }
    let settings = state.settings.lock().map_err(|_| state_error())?.clone();
    state
        .recorder
        .lock()
        .map_err(|_| state_error())?
        .start(&settings.input_source, &settings.input_device)
}

#[tauri::command]
fn start_recording(state: State<'_, AppState>) -> Result<(), String> {
    start_recording_inner(&state)
}

#[tauri::command]
fn recording_status(state: State<'_, AppState>) -> Result<RecordingStatus, String> {
    Ok(state.recorder.lock().map_err(|_| state_error())?.status())
}

async fn stop_and_transcribe_inner(app: AppHandle) -> Result<api::TranscriptResult, String> {
    let state = app.state::<AppState>();
    if state.processing.swap(true, Ordering::SeqCst) {
        return Err("Transkrypcja jest już przetwarzana".into());
    }
    let result = async {
        let wav = state
            .recorder
            .lock()
            .map_err(|_| state_error())?
            .stop_wav()?;
        let settings = state.settings.lock().map_err(|_| state_error())?.clone();
        let token =
            config::load_session().ok_or_else(|| "Zaloguj się w ustawieniach".to_string())?;
        emit_phase(&app, "processing", "Transkrybuję…", None);
        let mut transcript = api::transcribe(&settings.server_url, &token, wav)
            .await
            .map_err(|error| error.to_string())?;
        if transcript.transcript.is_empty() {
            return Err("Nie rozpoznano żadnego tekstu. Sprawdź poziom mikrofonu.".into());
        }
        if settings.polish {
            emit_phase(&app, "processing", "Poprawiam tekst…", None);
            if let Ok(polished) = api::polish(
                &settings.server_url,
                &token,
                &transcript.transcript,
                &transcript.detected_language,
            )
            .await
            {
                transcript.transcript = polished;
            }
        }
        app.clipboard()
            .write_text(transcript.transcript.clone())
            .map_err(|error| error.to_string())?;
        if settings.auto_paste && !main_window_focused(&app) {
            let _ = paste_clipboard();
        }
        Ok(transcript)
    }
    .await;
    state.processing.store(false, Ordering::SeqCst);
    result
}

#[tauri::command]
async fn stop_and_transcribe(app: AppHandle) -> Result<api::TranscriptResult, String> {
    let result = stop_and_transcribe_inner(app.clone()).await;
    match &result {
        Ok(transcript) => emit_phase(
            &app,
            "done",
            "Tekst gotowy",
            Some(transcript.transcript.clone()),
        ),
        Err(error) => emit_phase(&app, "error", error, None),
    }
    result
}

#[tauri::command]
fn cancel_recording(state: State<'_, AppState>) -> Result<(), String> {
    let mut recorder = state.recorder.lock().map_err(|_| state_error())?;
    if recorder.status().recording {
        let _ = recorder.stop_wav();
    }
    Ok(())
}

fn emit_phase(app: &AppHandle, phase: &str, message: &str, transcript: Option<String>) {
    let _ = app.emit(
        "flow-state",
        FlowEvent {
            phase: phase.into(),
            message: message.into(),
            transcript,
        },
    );
}

fn main_window_focused(app: &AppHandle) -> bool {
    app.get_webview_window("main")
        .and_then(|window| window.is_focused().ok())
        .unwrap_or(false)
}

fn paste_clipboard() -> Result<(), String> {
    std::thread::sleep(Duration::from_millis(90));
    let mut enigo = Enigo::new(&EnigoSettings::default()).map_err(|error| error.to_string())?;
    #[cfg(target_os = "macos")]
    let modifier = Key::Meta;
    #[cfg(not(target_os = "macos"))]
    let modifier = Key::Control;
    enigo
        .key(modifier, Direction::Press)
        .map_err(|error| error.to_string())?;
    let paste_result = enigo
        .key(Key::Unicode('v'), Direction::Click)
        .map_err(|error| error.to_string());
    let _ = enigo.key(modifier, Direction::Release);
    paste_result
}

fn register_shortcut(app: &AppHandle, shortcut: &str) -> Result<(), String> {
    app.global_shortcut()
        .unregister_all()
        .map_err(|error| error.to_string())?;
    app.global_shortcut()
        .register(shortcut)
        .map_err(|error| format!("Nieprawidłowy skrót: {error}"))
}

fn show_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn install_tray(app: &tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "Otwórz NupicAI Flow", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Zakończ", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;
    let mut builder = TrayIconBuilder::with_id("nupic-flow")
        .tooltip("NupicAI Flow")
        .menu(&menu)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_window(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_window(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            show_window(app)
        }))
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _, event| match event.state() {
                    ShortcutState::Pressed => {
                        let state = app.state::<AppState>();
                        match start_recording_inner(&state) {
                            Ok(()) => emit_phase(app, "recording", "Mów teraz", None),
                            Err(error) => emit_phase(app, "error", &error, None),
                        }
                    }
                    ShortcutState::Released => {
                        let handle = app.clone();
                        tauri::async_runtime::spawn(async move {
                            let result = stop_and_transcribe_inner(handle.clone()).await;
                            match result {
                                Ok(transcript) => emit_phase(
                                    &handle,
                                    "done",
                                    "Tekst gotowy",
                                    Some(transcript.transcript),
                                ),
                                Err(error) => emit_phase(&handle, "error", &error, None),
                            }
                        });
                    }
                })
                .build(),
        )
        .manage(AppState {
            settings: Mutex::new(config::load()),
            recorder: Mutex::new(AudioRecorder::default()),
            processing: AtomicBool::new(false),
        })
        .setup(|app| {
            install_tray(app)?;
            let shortcut = app
                .state::<AppState>()
                .settings
                .lock()
                .map_err(|_| std::io::Error::other(state_error()))?
                .shortcut
                .clone();
            register_shortcut(app.handle(), &shortcut).map_err(std::io::Error::other)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            bootstrap,
            login,
            logout,
            save_settings,
            list_input_devices,
            copy_text,
            start_recording,
            recording_status,
            stop_and_transcribe,
            cancel_recording,
        ])
        .run(tauri::generate_context!())
        .expect("NupicAI Flow failed to start");
}
