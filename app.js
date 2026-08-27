/* Parakeet Transcription — app.js */
'use strict';

const API = 'http://localhost:8765';

// ── State ──────────────────────────────────────────────────────────────────
let state = {
  file: null,
  objectUrl: null,
  result: null,
  currentWordIdx: -1,
  rafId: null,
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const fileName   = document.getElementById('file-name');
const transcribeBtn = document.getElementById('transcribe-btn');
const statusBar  = document.getElementById('status');
const spinner    = document.getElementById('spinner');
const results    = document.getElementById('results');
const audioPlayer = document.getElementById('audio-player');
const transcript  = document.getElementById('transcript-text');
const segList    = document.getElementById('seg-list');
const wordTrack  = document.getElementById('word-track');
const copyBtn    = document.getElementById('copy-btn');
const srtBtn     = document.getElementById('srt-btn');
const langBadge  = document.getElementById('lang-badge');
const statsEl    = document.getElementById('stats');

// ── File drop / pick ───────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(f) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = f;
  state.objectUrl = URL.createObjectURL(f);
  state.result = null;
  state.currentWordIdx = -1;

  fileName.textContent = f.name;
  fileName.title = `${f.name}  (${formatBytes(f.size)})`;
  dropZone.classList.add('has-file');
  transcribeBtn.disabled = false;
  results.hidden = true;
  setStatus('');
}

// ── Transcribe ─────────────────────────────────────────────────────────────
transcribeBtn.addEventListener('click', async () => {
  if (!state.file) return;
  await runTranscription();
});

async function runTranscription() {
  transcribeBtn.disabled = true;
  spinner.hidden = false;
  results.hidden = true;
  setStatus('Wysyłam plik…');

  const form = new FormData();
  form.append('file', state.file);

  const t0 = performance.now();
  try {
    setStatus('Transkrybuję… (model ASR ładuje się przy pierwszym wywołaniu ~20 s)');
    const resp = await fetch(`${API}/transcribe`, { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Gotowe! (${elapsed} s)`);
    state.result = data;
    renderResults(data);
  } catch (err) {
    setStatus(`❌ Błąd: ${err.message}`, true);
    console.error(err);
  } finally {
    spinner.hidden = true;
    transcribeBtn.disabled = false;
  }
}

// ── Render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  results.hidden = false;

  // Audio player
  audioPlayer.src = state.objectUrl;
  audioPlayer.hidden = false;

  // Language + stats
  langBadge.textContent = (data.detected_language || 'auto').toUpperCase();
  statsEl.textContent =
    `${data.word_count} słów · ${data.segment_count} segmentów · ${formatTime(data.duration)}`;

  // Full transcript
  transcript.textContent = data.transcript || '(brak transkrypcji)';

  // Segments
  segList.innerHTML = '';
  for (const seg of (data.segments || [])) {
    const li = document.createElement('li');
    li.dataset.start = seg.start;
    li.dataset.end = seg.end;
    li.innerHTML =
      `<span class="seg-time">${formatTime(seg.start)}</span>` +
      `<span class="seg-text">${escHtml(seg.text)}</span>`;
    li.addEventListener('click', () => seekTo(seg.start));
    segList.appendChild(li);
  }

  // Word timeline
  wordTrack.innerHTML = '';
  const totalDur = data.duration || 1;
  for (let i = 0; i < data.words.length; i++) {
    const w = data.words[i];
    const span = document.createElement('span');
    span.className = 'word-chip';
    span.dataset.idx = i;
    span.dataset.start = w.start;
    span.dataset.end = w.end;
    span.textContent = w.word;
    const left = (w.start / totalDur) * 100;
    const width = Math.max(0.3, ((w.end - w.start) / totalDur) * 100);
    span.style.cssText = `left:${left.toFixed(3)}%;width:${width.toFixed(3)}%`;
    span.addEventListener('click', () => seekTo(w.start));
    wordTrack.appendChild(span);
  }

  // Start word tracking
  stopWordTracking();
  startWordTracking(data.words, data.duration);
}

// ── Word highlight tracking ────────────────────────────────────────────────
function startWordTracking(words, duration) {
  function tick() {
    const t = audioPlayer.currentTime;
    const idx = bisect(words, t);
    if (idx !== state.currentWordIdx) {
      state.currentWordIdx = idx;
      highlightWord(idx, words);
      highlightSegment(t);
    }
    state.rafId = requestAnimationFrame(tick);
  }
  state.rafId = requestAnimationFrame(tick);
}

function stopWordTracking() {
  if (state.rafId !== null) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
}

function bisect(words, t) {
  // Find the word at time t
  let lo = 0, hi = words.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].start <= t) {
      if (words[mid].end >= t) best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  // Fallback: last word whose start <= t
  if (best === -1) {
    for (let i = words.length - 1; i >= 0; i--) {
      if (words[i].start <= t) { best = i; break; }
    }
  }
  return best;
}

function highlightWord(idx, words) {
  // Word chips
  const active = wordTrack.querySelector('.word-chip.active');
  if (active) active.classList.remove('active');
  if (idx >= 0) {
    const chip = wordTrack.querySelector(`[data-idx="${idx}"]`);
    if (chip) {
      chip.classList.add('active');
    }
  }
}

function highlightSegment(t) {
  const items = segList.querySelectorAll('li');
  items.forEach(li => {
    const start = parseFloat(li.dataset.start);
    const end = parseFloat(li.dataset.end);
    li.classList.toggle('active', t >= start && t <= end);
  });
  // Scroll active segment into view
  const activeLi = segList.querySelector('li.active');
  if (activeLi) {
    activeLi.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function seekTo(seconds) {
  if (!audioPlayer.src) return;
  audioPlayer.currentTime = seconds;
  audioPlayer.play();
}

// ── Copy / Export ──────────────────────────────────────────────────────────
copyBtn.addEventListener('click', () => {
  if (!state.result) return;
  navigator.clipboard.writeText(state.result.transcript || '').then(() => {
    copyBtn.textContent = '✓ Skopiowano';
    setTimeout(() => { copyBtn.textContent = 'Kopiuj'; }, 1500);
  });
});

srtBtn.addEventListener('click', () => {
  if (!state.result?.segments?.length) return;
  const lines = state.result.segments.map((seg, i) =>
    `${i + 1}\n${srtTime(seg.start)} --> ${srtTime(seg.end)}\n${seg.text}`
  );
  const blob = new Blob([lines.join('\n\n') + '\n'], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (state.file?.name.replace(/\.[^.]+$/, '') || 'transcript') + '.srt';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

// ── Utilities ──────────────────────────────────────────────────────────────
function setStatus(msg, isError = false) {
  statusBar.textContent = msg;
  statusBar.className = isError ? 'error' : '';
}

function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function srtTime(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.round((sec % 1) * 1000);
  return `${pad2(h)}:${pad2(m)}:${pad2(s)},${String(ms).padStart(3, '0')}`;
}

function pad2(n) { return String(n).padStart(2, '0'); }

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Health check on load ───────────────────────────────────────────────────
(async () => {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2000) });
    if (!r.ok) throw new Error();
    setStatus('Serwer gotowy.');
  } catch {
    setStatus('⚠ Serwer niedostępny — uruchom server.py (port 8765)', true);
  }
})();
