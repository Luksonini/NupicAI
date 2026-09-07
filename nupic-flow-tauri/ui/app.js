const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const copy = {
  pl: {
    settings: 'Ustawienia', account: 'Konto', logout: 'Wyloguj', server: 'Serwer', password: 'Hasło', login: 'Zaloguj',
    audioSource: 'Źródło nagrania', microphone: 'Mikrofon', systemAudio: 'Dźwięk systemu', defaultMicrophone: 'Domyślny mikrofon',
    shortcuts: 'Skróty globalne', hold: 'Przytrzymaj', toggle: 'Przełącz', autoVad: 'Auto VAD',
    autoPaste: 'Automatyczne wklejanie', autoPasteHelp: 'Wstaw tekst do aktywnego pola', aiCorrection: 'Korekta przez AI',
    aiCorrectionHelp: 'Dokładniej, ale trochę wolniej', language: 'Język aplikacji', languageAuto: 'Automatyczny',
    saveSettings: 'Zapisz ustawienia', saving: 'Zapisywanie…', saved: 'Zapisano', ready: 'Gotowy do dyktowania',
    loginRequired: 'Zaloguj się w ustawieniach', loginPrompt: 'Zaloguj się, aby rozpocząć dyktowanie.', loggingIn: 'Logowanie…',
    sessionSaved: 'Sesja została bezpiecznie zapisana.', loggingOut: 'Wylogowywanie…', loggedOut: 'Wylogowano.',
    speak: 'Mów teraz', listening: 'Nasłuchuję', transcribing: 'Transkrybuję…', readyText: 'Tekst gotowy',
    setShortcut: 'Ustaw skrót', pressShortcut: 'Naciśnij kombinację…', sourceChanged: 'Źródło: {source}', sourceSwitch: 'Źródło: {source}. Kliknij, aby przełączyć.',
    signalWaiting: 'Poziom pojawi się podczas nagrywania', signalSilent: 'Brak sygnału', signalLow: 'Słaby sygnał', signalGood: 'Sygnał prawidłowy',
    captured: '{seconds} s · poziom {level}%', settingsError: 'Nie udało się zapisać ustawień', copyTitle: 'Kopiuj', close: 'Zamknij',
    testMicrophone: 'Testuj mikrofon', stopTest: 'Zatrzymaj test', testingMicrophone: 'Powiedz kilka słów…',
    micNoSamples: 'Mikrofon nie przekazał żadnych próbek.', micNoSignal: 'Próbki docierają, ale zawierają ciszę.',
    micWeak: 'Mikrofon działa, ale sygnał jest bardzo słaby.', micWorks: 'Mikrofon działa prawidłowo.',
    micResult: 'szczyt {peak}% · RMS {rms}%',
    liveStart: 'Uruchom transkrypcję na żywo', liveStop: 'Zatrzymaj transkrypcję na żywo',
  },
  en: {
    settings: 'Settings', account: 'Account', logout: 'Sign out', server: 'Server', password: 'Password', login: 'Sign in',
    audioSource: 'Recording source', microphone: 'Microphone', systemAudio: 'System audio', defaultMicrophone: 'Default microphone',
    shortcuts: 'Global shortcuts', hold: 'Hold to talk', toggle: 'Toggle recording', autoVad: 'Auto VAD',
    autoPaste: 'Paste automatically', autoPasteHelp: 'Insert text into the active field', aiCorrection: 'AI correction',
    aiCorrectionHelp: 'More accurate, but slightly slower', language: 'App language', languageAuto: 'Automatic',
    saveSettings: 'Save settings', saving: 'Saving…', saved: 'Saved', ready: 'Ready for dictation',
    loginRequired: 'Sign in from Settings', loginPrompt: 'Sign in to start dictating.', loggingIn: 'Signing in…',
    sessionSaved: 'Your session was saved securely.', loggingOut: 'Signing out…', loggedOut: 'Signed out.',
    speak: 'Speak now', listening: 'Listening', transcribing: 'Transcribing…', readyText: 'Text ready',
    setShortcut: 'Set shortcut', pressShortcut: 'Press a key combination…', sourceChanged: 'Source: {source}', sourceSwitch: 'Source: {source}. Click to switch.',
    signalWaiting: 'The level will appear while recording', signalSilent: 'No signal', signalLow: 'Low signal', signalGood: 'Signal is good',
    captured: '{seconds} s · level {level}%', settingsError: 'Could not save settings', copyTitle: 'Copy', close: 'Close',
    testMicrophone: 'Test microphone', stopTest: 'Stop test', testingMicrophone: 'Say a few words…',
    micNoSamples: 'The microphone did not provide any samples.', micNoSignal: 'Samples arrived, but they contain silence.',
    micWeak: 'The microphone works, but its signal is very low.', micWorks: 'The microphone works correctly.',
    micResult: 'peak {peak}% · RMS {rms}%',
    liveStart: 'Start live transcription', liveStop: 'Stop live transcription',
  },
};

const elements = {
  record: document.querySelector('#record'), status: document.querySelector('#status'), shortcut: document.querySelector('#shortcut'),
  wave: document.querySelector('#wave'), result: document.querySelector('#result'), transcript: document.querySelector('#transcript'),
  copy: document.querySelector('#copy'), settings: document.querySelector('#settings'), scrim: document.querySelector('#scrim'),
  settingsOpen: document.querySelector('#settings-open'), settingsClose: document.querySelector('#settings-close'),
  liveVad: document.querySelector('#live-vad'),
  quickSource: document.querySelector('#quick-source'), quickSourceLabel: document.querySelector('#quick-source-label'),
  loginForm: document.querySelector('#login-form'), authenticated: document.querySelector('#account-authenticated'),
  accountName: document.querySelector('#account-name'), logout: document.querySelector('#logout'), preferences: document.querySelector('#preferences-form'),
  saveSettings: document.querySelector('#save-settings'), serverUrl: document.querySelector('#server-url'), email: document.querySelector('#email'),
  password: document.querySelector('#password'), inputDevice: document.querySelector('#input-device'),
  testMicrophone: document.querySelector('#test-microphone'),
  inputSources: document.querySelectorAll('input[name="input-source"]'), microphoneDeviceRow: document.querySelector('#microphone-device-row'),
  hotkeys: document.querySelectorAll('[data-hotkey]'), hotkeyCaptures: document.querySelectorAll('[data-hotkey-capture]'),
  autoPaste: document.querySelector('#auto-paste'), polish: document.querySelector('#polish'), language: document.querySelector('#language'),
  settingsMessage: document.querySelector('#settings-message'), audioDiagnostic: document.querySelector('#audio-diagnostic'),
};

let recording = false;
let processing = false;
let authenticated = false;
let transcript = '';
let continuous = false;
let inputSource = 'microphone';
let languagePreference = 'auto';
let locale = 'pl';
let capturingHotkey = null;
let previousHotkey = '';
let testingMicrophone = false;
let microphoneTestTimer = null;

function resolveLocale(preference = languagePreference) {
  if (preference === 'pl' || preference === 'en') return preference;
  return String(navigator.language || '').toLowerCase().startsWith('pl') ? 'pl' : 'en';
}

function t(key, values = {}) {
  let value = copy[locale]?.[key] || copy.pl[key] || key;
  for (const [name, replacement] of Object.entries(values)) value = value.replace(`{${name}}`, replacement);
  return value;
}

function applyLanguage(preference = languagePreference) {
  languagePreference = preference || 'auto';
  locale = resolveLocale(languagePreference);
  document.documentElement.lang = locale;
  document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
  elements.settingsOpen.title = elements.settingsOpen.ariaLabel = t('settings');
  elements.settingsClose.title = elements.settingsClose.ariaLabel = t('close');
  elements.copy.title = elements.copy.ariaLabel = t('copyTitle');
  elements.liveVad.title = elements.liveVad.ariaLabel = t(continuous ? 'liveStop' : 'liveStart');
  elements.hotkeyCaptures.forEach(button => { button.title = button.ariaLabel = t('setShortcut'); });
  renderDevices([...elements.inputDevice.options].slice(1).map(option => option.value), elements.inputDevice.value);
  syncInputSource();
}

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

function hotkeyInput(mode) { return document.querySelector(`[data-hotkey="${mode}"]`); }

function beginHotkeyCapture(mode) {
  if (capturingHotkey) finishHotkeyCapture(capturingHotkey);
  capturingHotkey = mode;
  const input = hotkeyInput(mode);
  previousHotkey = input.value;
  input.value = t('pressShortcut');
  document.querySelector(`[data-hotkey-capture="${mode}"]`)?.classList.add('active');
}

function finishHotkeyCapture(mode, value = previousHotkey) {
  hotkeyInput(mode).value = value;
  document.querySelector(`[data-hotkey-capture="${mode}"]`)?.classList.remove('active');
  capturingHotkey = null;
}

function shortcutFromEvent(event) {
  const keyNames = { ' ': 'Space', ArrowUp: 'ArrowUp', ArrowDown: 'ArrowDown', ArrowLeft: 'ArrowLeft', ArrowRight: 'ArrowRight', Escape: 'Escape' };
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
  elements.accountName.textContent = user?.display_name || user?.email || elements.email.value || 'NupicAI';
  elements.authenticated.hidden = false;
  elements.loginForm.hidden = true;
  setStatus(t('ready'));
}

function setLoggedOut() {
  authenticated = false;
  elements.authenticated.hidden = true;
  elements.loginForm.hidden = false;
  setStatus(t('loginRequired'));
}

function settingsMessage(message, error = false, success = false) {
  elements.settingsMessage.textContent = message;
  elements.settingsMessage.classList.toggle('error', error);
  elements.settingsMessage.classList.toggle('success', success);
}

function openSettings() { elements.settings.classList.add('open'); elements.settings.setAttribute('aria-hidden', 'false'); elements.scrim.hidden = false; }
function closeSettings() { elements.settings.classList.remove('open'); elements.settings.setAttribute('aria-hidden', 'true'); elements.scrim.hidden = true; }

function renderDevices(devices, selected = '') {
  elements.inputDevice.replaceChildren(new Option(t('defaultMicrophone'), ''));
  for (const device of devices) elements.inputDevice.add(new Option(device, device));
  elements.inputDevice.value = selected;
}

function selectedSource() { return [...elements.inputSources].find(input => input.checked)?.value || 'microphone'; }

function syncInputSource() {
  inputSource = selectedSource();
  const systemAudio = inputSource === 'system';
  elements.inputDevice.disabled = systemAudio;
  elements.microphoneDeviceRow.classList.toggle('disabled', systemAudio);
  elements.quickSource.classList.toggle('system', systemAudio);
  elements.quickSourceLabel.textContent = t(systemAudio ? 'systemAudio' : 'microphone');
  elements.quickSource.title = elements.quickSource.ariaLabel = t('sourceSwitch', { source: elements.quickSourceLabel.textContent });
}

async function load() {
  closeSettings();
  try {
    const data = await invoke('bootstrap');
    const settings = data.settings;
    languagePreference = settings.language || 'auto';
    elements.language.value = languagePreference;
    applyLanguage(languagePreference);
    elements.serverUrl.value = settings.server_url;
    elements.email.value = settings.email;
    hotkeyInput('hold').value = settings.shortcut_hold || 'Ctrl+Alt+Space';
    hotkeyInput('toggle').value = settings.shortcut_toggle || 'Ctrl+Alt+D';
    hotkeyInput('continuous').value = settings.shortcut_continuous || 'Ctrl+Alt+V';
    renderShortcut(hotkeyInput('hold').value);
    const source = [...elements.inputSources].find(input => input.value === settings.input_source);
    if (source) source.checked = true;
    syncInputSource();
    elements.autoPaste.checked = settings.auto_paste;
    elements.polish.checked = settings.polish;
    renderDevices(data.devices, settings.input_device);
    if (data.authenticated) setAuthenticated(data.user);
    else setLoggedOut();
    if (data.auth_error) settingsMessage(data.auth_error, !data.authenticated);
    closeSettings();
  } catch (error) {
    setStatus(String(error), 'error');
    openSettings();
  }
}

async function beginRecording(mode = 'toggle') {
  if (processing || recording) return;
  if (!authenticated) { openSettings(); settingsMessage(t('loginPrompt'), true); return; }
  try {
    if (mode === 'continuous') {
      await invoke('start_continuous');
      continuous = recording = true;
      setStatus(t('listening'), 'listening');
      return;
    }
    await invoke('start_recording');
    recording = true;
    setStatus(t('speak'), 'recording');
  } catch (error) { setStatus(String(error), 'error'); }
}

async function finishRecording() {
  if (!recording || (processing && !continuous)) return;
  if (continuous) { continuous = false; await invoke('stop_continuous'); return; }
  recording = false;
  processing = true;
  setStatus(t('transcribing'), 'processing');
  try {
    const result = await invoke('stop_and_transcribe');
    showTranscript(result.transcript);
    setStatus(t('readyText'));
  } catch (error) { setStatus(String(error), 'error'); }
  finally { processing = false; }
}

function renderMicrophoneTest(result) {
  const peak = Math.round(Math.min(1, result.peak || 0) * 100);
  const rms = Math.round(Math.min(1, result.rms || 0) * 1000) / 10;
  let key = 'micWorks';
  let error = false;
  if (!result.sample_count) { key = 'micNoSamples'; error = true; }
  else if ((result.peak || 0) < 0.0005) { key = 'micNoSignal'; error = true; }
  else if ((result.rms || 0) < 0.003) { key = 'micWeak'; error = true; }
  settingsMessage(`${t(key)} ${t('micResult', { peak, rms })}`, error, !error);
}

async function stopMicrophoneTest() {
  if (!testingMicrophone) return;
  testingMicrophone = false;
  clearTimeout(microphoneTestTimer);
  elements.testMicrophone.disabled = true;
  try {
    renderMicrophoneTest(await invoke('stop_microphone_test'));
  } catch (error) { settingsMessage(String(error), true); }
  finally {
    elements.testMicrophone.disabled = false;
    elements.testMicrophone.querySelector('span').textContent = t('testMicrophone');
  }
}

function normalizedWord(word) {
  return String(word || '').toLocaleLowerCase(locale).replace(/[^\p{L}\p{N}]+/gu, '');
}

function mergeTranscript(current, incoming) {
  const left = String(current || '').trim();
  const right = String(incoming || '').trim();
  if (!left) return right;
  if (!right) return left;
  const leftWords = left.split(/\s+/);
  const rightWords = right.split(/\s+/);
  const maxOverlap = Math.min(12, leftWords.length, rightWords.length);
  let overlap = 0;
  for (let size = maxOverlap; size >= 1; size -= 1) {
    const suffix = leftWords.slice(-size).map(normalizedWord);
    const prefix = rightWords.slice(0, size).map(normalizedWord);
    if (suffix.every((word, index) => word && word === prefix[index])) {
      overlap = size;
      break;
    }
  }
  return `${left} ${rightWords.slice(overlap).join(' ')}`.trim();
}

function showTranscript(text, append = false) {
  transcript = append ? mergeTranscript(transcript, text) : (text || '');
  elements.transcript.textContent = transcript;
  elements.result.hidden = !transcript;
}

elements.record.addEventListener('click', () => recording ? finishRecording() : beginRecording('toggle'));
elements.liveVad.addEventListener('click', () => continuous ? finishRecording() : beginRecording('continuous'));
elements.hotkeys.forEach(input => input.addEventListener('click', () => beginHotkeyCapture(input.dataset.hotkey)));
elements.hotkeyCaptures.forEach(button => button.addEventListener('click', () => beginHotkeyCapture(button.dataset.hotkeyCapture)));
window.addEventListener('keydown', event => {
  if (!capturingHotkey) return;
  event.preventDefault(); event.stopPropagation();
  const mode = capturingHotkey;
  if (event.key === 'Escape') { finishHotkeyCapture(mode); return; }
  const shortcut = shortcutFromEvent(event);
  if (shortcut) finishHotkeyCapture(mode, shortcut);
}, true);

elements.settingsOpen.addEventListener('click', openSettings);
elements.settingsClose.addEventListener('click', closeSettings);
elements.scrim.addEventListener('click', closeSettings);
elements.inputSources.forEach(input => input.addEventListener('change', syncInputSource));
elements.language.addEventListener('change', () => applyLanguage(elements.language.value));
elements.testMicrophone.addEventListener('click', async () => {
  if (testingMicrophone) { await stopMicrophoneTest(); return; }
  if (recording || processing) return;
  try {
    await invoke('start_microphone_test', { inputDevice: elements.inputDevice.value });
    testingMicrophone = true;
    elements.testMicrophone.querySelector('span').textContent = t('stopTest');
    settingsMessage(t('testingMicrophone'));
    microphoneTestTimer = setTimeout(stopMicrophoneTest, 5000);
  } catch (error) { settingsMessage(String(error), true); }
});

elements.quickSource.addEventListener('click', async () => {
  if (recording || processing) return;
  const next = inputSource === 'microphone' ? 'system' : 'microphone';
  const radio = [...elements.inputSources].find(input => input.value === next);
  if (radio) radio.checked = true;
  syncInputSource();
  try {
    await invoke('set_input_source', { inputSource: next });
    settingsMessage(t('sourceChanged', { source: elements.quickSourceLabel.textContent }), false, true);
  } catch (error) { settingsMessage(String(error), true); }
});

elements.copy.addEventListener('click', async () => { if (transcript) await invoke('copy_text', { text: transcript }); });

elements.loginForm.addEventListener('submit', async event => {
  event.preventDefault();
  const button = elements.loginForm.querySelector('button[type="submit"]');
  button.disabled = true;
  settingsMessage(t('loggingIn'));
  try {
    const user = await invoke('login', { serverUrl: elements.serverUrl.value.trim(), email: elements.email.value.trim(), password: elements.password.value });
    elements.password.value = '';
    setAuthenticated(user);
    settingsMessage(t('sessionSaved'), false, true);
  } catch (error) { settingsMessage(String(error), true); }
  finally { button.disabled = false; }
});

elements.logout.addEventListener('click', async () => {
  elements.logout.disabled = true;
  settingsMessage(t('loggingOut'));
  try { await invoke('logout'); setLoggedOut(); settingsMessage(t('loggedOut'), false, true); }
  catch (error) { settingsMessage(String(error), true); }
  finally { elements.logout.disabled = false; }
});

elements.preferences.addEventListener('submit', async event => {
  event.preventDefault();
  elements.saveSettings.disabled = true;
  elements.saveSettings.classList.remove('saved');
  elements.saveSettings.textContent = t('saving');
  settingsMessage(t('saving'));
  try {
    await invoke('save_settings', { settings: {
      server_url: elements.serverUrl.value.trim(), email: elements.email.value.trim(), input_source: selectedSource(),
      input_device: elements.inputDevice.value, polish: elements.polish.checked, auto_paste: elements.autoPaste.checked,
      shortcut_hold: hotkeyInput('hold').value.trim(), shortcut_toggle: hotkeyInput('toggle').value.trim(),
      shortcut_continuous: hotkeyInput('continuous').value.trim(), language: elements.language.value,
    }});
    applyLanguage(elements.language.value);
    renderShortcut(hotkeyInput('hold').value.trim());
    elements.saveSettings.classList.add('saved');
    elements.saveSettings.textContent = t('saved');
    settingsMessage(t('saved'), false, true);
    setTimeout(() => { elements.saveSettings.classList.remove('saved'); elements.saveSettings.textContent = t('saveSettings'); }, 1400);
  } catch (error) {
    elements.saveSettings.textContent = t('saveSettings');
    settingsMessage(String(error) || t('settingsError'), true);
  } finally { elements.saveSettings.disabled = false; }
});

listen('flow-state', ({ payload }) => {
  if (payload.phase === 'recording') { recording = true; setStatus(t('speak'), 'recording'); }
  else if (payload.phase === 'listening') { continuous = recording = true; processing = false; elements.liveVad.classList.add('active'); elements.liveVad.title = elements.liveVad.ariaLabel = t('liveStop'); setStatus(t('listening'), 'listening'); }
  else if (payload.phase === 'processing') { recording = continuous; processing = true; setStatus(t('transcribing'), 'processing'); }
  else if (payload.phase === 'phrase') { recording = true; processing = false; showTranscript(payload.transcript, true); setStatus(t('listening'), 'listening'); }
  else if (payload.phase === 'done') { recording = processing = false; showTranscript(payload.transcript); setStatus(t('readyText')); }
  else if (payload.phase === 'error') { processing = false; setStatus(payload.message, 'error'); }
  else if (payload.phase === 'idle') { continuous = recording = processing = false; elements.liveVad.classList.remove('active'); elements.liveVad.title = elements.liveVad.ariaLabel = t('liveStart'); setStatus(t('ready')); }
});

setInterval(async () => {
  try {
    const state = await invoke('recording_status');
    elements.quickSource.disabled = state.recording || processing;
    if (!state.recording) {
      elements.audioDiagnostic.className = 'audio-diagnostic';
      elements.audioDiagnostic.querySelector('span').textContent = t('signalWaiting');
      elements.audioDiagnostic.querySelector('strong').textContent = '';
      return;
    }
    const level = Math.max(0, Math.min(1, state.level));
    const percent = Math.round(level * 100);
    const signalKey = percent < 2 ? 'signalSilent' : percent < 9 ? 'signalLow' : 'signalGood';
    elements.audioDiagnostic.className = `audio-diagnostic ${percent < 2 ? 'silent' : percent < 9 ? 'low' : 'good'}`;
    elements.audioDiagnostic.querySelector('span').textContent = t(signalKey);
    elements.audioDiagnostic.querySelector('strong').textContent = t('captured', { seconds: (state.elapsed_ms / 1000).toFixed(1), level: percent });
    const visualLevel = Math.max(.06, level);
    [...elements.wave.children].forEach((bar, index, bars) => {
      const center = 1 - Math.abs(index - (bars.length - 1) / 2) / (bars.length / 2);
      const flutter = .76 + Math.sin(Date.now() / 95 + index * 1.7) * .24;
      bar.style.height = `${5 + visualLevel * center * flutter * 38}px`;
    });
  } catch (_) {}
}, 80);

load();
