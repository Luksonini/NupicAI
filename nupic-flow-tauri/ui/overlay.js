const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const button = document.querySelector('#dictate');
const quit = document.querySelector('#quit');
const english = (navigator.language || '').toLowerCase().startsWith('en');
const labels = english
  ? { ready: 'Start dictation', recording: 'Stop and paste', processing: 'Transcribing…', error: 'Open NupicAI Flow to view the error', microphone: 'microphone', system: 'system audio' }
  : { ready: 'Rozpocznij dyktowanie', recording: 'Zatrzymaj i wklej', processing: 'Transkrybuję…', error: 'Otwórz NupicAI Flow, aby sprawdzić błąd', microphone: 'mikrofon', system: 'dźwięk systemu' };

let phase = 'ready';
let inputSource = 'microphone';

function render(nextPhase = phase) {
  phase = nextPhase;
  document.body.classList.toggle('recording', phase === 'recording' || phase === 'listening');
  document.body.classList.toggle('processing', phase === 'processing');
  document.body.classList.toggle('error', phase === 'error');
  document.body.classList.toggle('system', inputSource === 'system');
  const label = phase === 'recording' || phase === 'listening'
    ? labels.recording
    : phase === 'processing' ? labels.processing
      : phase === 'error' ? labels.error : labels.ready;
  button.title = button.ariaLabel = `${label} · ${labels[inputSource]}`;
  button.disabled = phase === 'processing';
}

function setInputSource(source) {
  inputSource = source === 'system' ? 'system' : 'microphone';
  render();
}

button.addEventListener('click', async () => {
  try { await invoke('toggle_floating_recording'); }
  catch (error) { render('error'); button.title = button.ariaLabel = String(error); }
});

button.addEventListener('contextmenu', async event => {
  event.preventDefault();
  await invoke('show_main_window');
});

quit.addEventListener('click', async event => {
  event.stopPropagation();
  await invoke('quit_app');
});

listen('flow-state', ({ payload }) => render(payload.phase));
listen('input-source-changed', ({ payload }) => setInputSource(payload));

setInterval(async () => {
  try {
    const status = await invoke('recording_status');
    if (status.recording && !document.body.classList.contains('recording')) render('recording');
  } catch (_) {}
}, 500);

invoke('current_input_source').then(setInputSource).catch(() => render('ready'));
