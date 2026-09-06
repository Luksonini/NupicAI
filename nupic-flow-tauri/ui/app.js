const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const elements = {
  record: document.querySelector('#record'),
  status: document.querySelector('#status'),
  wave: document.querySelector('#wave'),
  result: document.querySelector('#result'),
  transcript: document.querySelector('#transcript'),
  copy: document.querySelector('#copy'),
  settings: document.querySelector('#settings'),
  scrim: document.querySelector('#scrim'),
  settingsOpen: document.querySelector('#settings-open'),
  settingsClose: document.querySelector('#settings-close'),
  loginForm: document.querySelector('#login-form'),
  authenticated: document.querySelector('#account-authenticated'),
  accountName: document.querySelector('#account-name'),
  logout: document.querySelector('#logout'),
  preferences: document.querySelector('#preferences-form'),
  serverUrl: document.querySelector('#server-url'),
  email: document.querySelector('#email'),
  password: document.querySelector('#password'),
  inputDevice: document.querySelector('#input-device'),
  inputSources: document.querySelectorAll('input[name="input-source"]'),
  activationModes: document.querySelectorAll('input[name="activation-mode"]'),
  microphoneDeviceRow: document.querySelector('#microphone-device-row'),
  hotkey: document.querySelector('#hotkey'),
  hotkeyCapture: document.querySelector('#hotkey-capture'),
  autoPaste: document.querySelector('#auto-paste'),
  polish: document.querySelector('#polish'),
  settingsMessage: document.querySelector('#settings-message'),
};

let recording = false;
let processing = false;
let authenticated = false;
let transcript = '';
let continuous = false;
let activationMode = 'hold';
let capturingHotkey = false;
let previousHotkey = 'Ctrl+Alt+Space';

function setStatus(message, phase = 'idle') {
  elements.status.textContent = message;
  elements.status.classList.toggle('error', phase === 'error');
  document.body.classList.toggle('recording', phase === 'recording');
  elements.wave.classList.toggle('active', phase === 'recording');
  elements.wave.classList.toggle('listening', phase === 'listening');
  elements.wave.classList.toggle('processing', phase === 'processing');
}

function renderShortcut(shortcut) {
  const parts = String(shortcut || '').split('+').map(part => part.trim()).filter(Boolean);
  elements.shortcut.replaceChildren();
  parts.forEach((part, index) => {
    if (index) elements.shortcut.append(document.createTextNode('+'));
    const key = document.createElement('kbd');
    key.textContent = part;
    elements.shortcut.append(key);
  });
}

function beginHotkeyCapture() {
  if (capturingHotkey) return;
  capturingHotkey = true;
  previousHotkey = elements.hotkey.value;
  elements.hotkey.value = 'Naciśnij kombinację…';
  elements.hotkeyCapture.classList.add('active');
  elements.hotkeyCapture.setAttribute('aria-label', 'Oczekiwanie na skrót');
}

function finishHotkeyCapture(value = previousHotkey) {
  capturingHotkey = false;
  elements.hotkey.value = value;
  elements.hotkeyCapture.classList.remove('active');
  elements.hotkeyCapture.setAttribute('aria-label', 'Ustaw skrót');
}

function shortcutFromEvent(event) {
  const keyNames = {
    ' ': 'Space',
    ArrowUp: 'ArrowUp',
    ArrowDown: 'ArrowDown',
    ArrowLeft: 'ArrowLeft',
    ArrowRight: 'ArrowRight',
    Escape: 'Escape',
  };
  const modifiers = [];
  if (event.ctrlKey) modifiers.push('Ctrl');
  if (event.altKey) modifiers.push('Alt');
  if (event.shiftKey) modifiers.push('Shift');
  if (event.metaKey) modifiers.push('Super');
  if (['Control', 'Alt', 'Shift', 'Meta'].includes(event.key)) return null;
  let key = keyNames[event.key] || event.key;
  if (/^[a-z]$/i.test(key)) key = key.toUpperCase();
  if (!/^(?:[A-Z0-9]|F(?:[1-9]|1[0-9]|2[0-4])|Space|Enter|Tab|Backspace|Delete|Insert|Home|End|PageUp|PageDown|ArrowUp|ArrowDown|ArrowLeft|ArrowRight|Escape)$/.test(key)) return null;
  return [...modifiers, key].join('+');
}

function setAuthenticated(user) {
  authenticated = true;
  const identity = user?.display_name || user?.email || elements.email.value || 'NupicAI';
  elements.accountName.textContent = identity;
  elements.authenticated.hidden = false;
  elements.loginForm.hidden = true;
  setStatus('Gotowy do dyktowania');
}

function setLoggedOut() {
  authenticated = false;
  elements.authenticated.hidden = true;
  elements.loginForm.hidden = false;
  setStatus('Zaloguj się w ustawieniach');
}

function settingsMessage(message, error = false) {
  elements.settingsMessage.textContent = message;
  elements.settingsMessage.classList.toggle('error', error);
}

function openSettings() {
  elements.settings.classList.add('open');
  elements.settings.setAttribute('aria-hidden', 'false');
  elements.scrim.hidden = false;
}

function closeSettings() {
  elements.settings.classList.remove('open');
  elements.settings.setAttribute('aria-hidden', 'true');
  elements.scrim.hidden = true;
}

function renderDevices(devices, selected = '') {
  elements.inputDevice.replaceChildren(new Option('Domyślny mikrofon', ''));
  for (const device of devices) elements.inputDevice.add(new Option(device, device));
  elements.inputDevice.value = selected;
}

function syncInputSource() {
  const systemAudio = [...elements.inputSources].some(input => input.checked && input.value === 'system');
  elements.inputDevice.disabled = systemAudio;
  elements.microphoneDeviceRow.classList.toggle('disabled', systemAudio);
}

async function load() {
  try {
    const data = await invoke('bootstrap');
    const settings = data.settings;
    elements.serverUrl.value = settings.server_url;
    elements.email.value = settings.email;
    elements.hotkey.value = settings.shortcut;
    renderShortcut(settings.shortcut);
    activationMode = settings.activation_mode || 'hold';
    const activation = [...elements.activationModes].find(input => input.value === activationMode);
    if (activation) activation.checked = true;
    const source = [...elements.inputSources].find(input => input.value === settings.input_source);
    if (source) source.checked = true;
    syncInputSource();
    elements.autoPaste.checked = settings.auto_paste;
    elements.polish.checked = settings.polish;
    renderDevices(data.devices, settings.input_device);
    if (data.authenticated) setAuthenticated(data.user);
    else setLoggedOut();
    if (data.auth_error) settingsMessage(data.auth_error, !data.authenticated);
  } catch (error) {
    setStatus(String(error), 'error');
    openSettings();
  }
}

async function beginRecording() {
  if (processing || recording) return;
  if (!authenticated) {
    openSettings();
    settingsMessage('Zaloguj się, aby rozpocząć dyktowanie.', true);
    return;
  }
  try {
    if (activationMode === 'continuous') {
      await invoke('start_continuous');
      continuous = true;
      recording = true;
      setStatus('Nasłuchuję', 'listening');
      return;
    }
    await invoke('start_recording');
    recording = true;
    elements.record.setAttribute('aria-label', 'Zatrzymaj nagrywanie');
    setStatus('Mów teraz', 'recording');
  } catch (error) {
    setStatus(String(error), 'error');
  }
}

async function finishRecording() {
  if (!recording || (processing && !continuous)) return;
  if (continuous) {
    continuous = false;
    await invoke('stop_continuous');
    return;
  }
  recording = false;
  processing = true;
  elements.record.setAttribute('aria-label', 'Rozpocznij nagrywanie');
  setStatus('Transkrybuję…', 'processing');
  try {
    const result = await invoke('stop_and_transcribe');
    showTranscript(result.transcript);
    setStatus('Tekst gotowy');
  } catch (error) {
    setStatus(String(error), 'error');
  } finally {
    processing = false;
  }
}

function showTranscript(text, append = false) {
  transcript = append && transcript ? `${transcript} ${text || ''}`.trim() : (text || '');
  elements.transcript.textContent = transcript;
  elements.result.hidden = !transcript;
}

elements.record.addEventListener('click', () => recording ? finishRecording() : beginRecording());
elements.hotkey.addEventListener('click', beginHotkeyCapture);
elements.hotkeyCapture.addEventListener('click', beginHotkeyCapture);
window.addEventListener('keydown', event => {
  if (!capturingHotkey) return;
  event.preventDefault();
  event.stopPropagation();
  if (event.key === 'Escape') {
    finishHotkeyCapture();
    return;
  }
  const shortcut = shortcutFromEvent(event);
  if (shortcut) finishHotkeyCapture(shortcut);
}, true);
elements.settingsOpen.addEventListener('click', openSettings);
elements.settingsClose.addEventListener('click', closeSettings);
elements.scrim.addEventListener('click', closeSettings);
elements.inputSources.forEach(input => input.addEventListener('change', syncInputSource));

elements.copy.addEventListener('click', async () => {
  if (transcript) await invoke('copy_text', { text: transcript });
});

elements.loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  settingsMessage('Logowanie…');
  try {
    const user = await invoke('login', {
      serverUrl: elements.serverUrl.value.trim(),
      email: elements.email.value.trim(),
      password: elements.password.value,
    });
    elements.password.value = '';
    setAuthenticated(user);
    settingsMessage('Sesja została bezpiecznie zapisana.');
  } catch (error) {
    settingsMessage(String(error), true);
  }
});

elements.logout.addEventListener('click', async () => {
  await invoke('logout');
  setLoggedOut();
  settingsMessage('Wylogowano.');
});

elements.preferences.addEventListener('submit', async (event) => {
  event.preventDefault();
  try {
    await invoke('save_settings', {
      settings: {
        server_url: elements.serverUrl.value.trim(),
        email: elements.email.value.trim(),
        input_source: [...elements.inputSources].find(input => input.checked)?.value || 'microphone',
        input_device: elements.inputDevice.value,
        polish: elements.polish.checked,
        auto_paste: elements.autoPaste.checked,
        shortcut: elements.hotkey.value.trim(),
        activation_mode: [...elements.activationModes].find(input => input.checked)?.value || 'hold',
      },
    });
    activationMode = [...elements.activationModes].find(input => input.checked)?.value || 'hold';
    renderShortcut(elements.hotkey.value.trim());
    settingsMessage('Ustawienia zapisane.');
  } catch (error) {
    settingsMessage(String(error), true);
  }
});

listen('flow-state', ({ payload }) => {
  if (payload.phase === 'recording') {
    recording = true;
    setStatus(payload.message, 'recording');
  } else if (payload.phase === 'listening') {
    continuous = true;
    recording = true;
    processing = false;
    setStatus(payload.message, 'listening');
  } else if (payload.phase === 'processing') {
    recording = continuous;
    processing = true;
    setStatus(payload.message, 'processing');
  } else if (payload.phase === 'phrase') {
    recording = true;
    processing = false;
    showTranscript(payload.transcript, true);
    setStatus('Nasłuchuję', 'listening');
  } else if (payload.phase === 'done') {
    recording = false;
    processing = false;
    showTranscript(payload.transcript);
    setStatus(payload.message);
  } else if (payload.phase === 'error') {
    processing = false;
    setStatus(payload.message, 'error');
  } else if (payload.phase === 'idle') {
    continuous = false;
    recording = false;
    processing = false;
    setStatus(payload.message);
  }
});

setInterval(async () => {
  try {
    const state = await invoke('recording_status');
    if (!state.recording) return;
    const level = Math.max(.06, Math.min(1, state.level));
    [...elements.wave.children].forEach((bar, index, bars) => {
      const center = 1 - Math.abs(index - (bars.length - 1) / 2) / (bars.length / 2);
      const flutter = .76 + Math.sin(Date.now() / 95 + index * 1.7) * .24;
      bar.style.height = `${5 + level * center * flutter * 38}px`;
    });
  } catch (_) {}
}, 80);

load();
