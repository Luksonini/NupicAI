'use client';
import { useEffect, useState } from 'react';
import type { Speaker, TTSChunk, TTSToken, TTSResult, TTSModelProfile } from '@/lib/types';
import { listSpeakers, listTTSModels, streamJob } from '@/lib/api';
import JobProgress from './JobProgress';

const BASE = '';

async function submitTextTTS(params: {
  text: string; speaker_label: string; tts_model_profile: string; language: string;
  speed: number; dur_scale: number;
  mel_steps_first: number; mel_steps_second: number;
  mel_twopass_t_noise: number; digital_silence: boolean; pause_edge_frames: number;
  short_continuity_ms: number;
  emotion_group: string;
  emotion_strength: number;
}): Promise<string> {
  const res = await fetch(`${BASE}/tts_text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return (await res.json()).job_id as string;
}

function Slider({ label, value, min, max, step, onChange }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-muted mb-1">
        <span>{label}</span><span className="text-slate-300">{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-accent h-1.5" />
    </div>
  );
}

function TokenRow({ t }: { t: TTSToken }) {
  const isPause = t.is_pause;
  const isLow = t.low && !isPause;
  return (
    <tr className={`text-xs border-b border-border/40 ${isPause ? 'text-blue-400/80' : isLow ? 'text-orange-400' : 'text-slate-300'}`}>
      <td className="py-0.5 pr-3 font-mono">{t.token}</td>
      <td className="py-0.5 pr-3 text-right tabular-nums">{t.dur}</td>
      <td className="py-0.5 text-right tabular-nums">{t.dur_sec.toFixed(3)}</td>
    </tr>
  );
}

function ChunkCard({ chunk, idx }: { chunk: TTSChunk; idx: number }) {
  const [open, setOpen] = useState(false);
  const pauseTokens = chunk.tokens.filter(t => t.is_pause);
  const pauseSec = pauseTokens.reduce((s, t) => s + t.dur_sec, 0);
  const contentSec = (chunk.mel_sec ?? 0) - pauseSec;

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-surface2 transition-colors"
      >
        <span className="shrink-0 w-5 h-5 rounded-full bg-accent-dim text-white text-xs flex items-center justify-center font-bold">
          {idx + 1}
        </span>
        <div className="flex items-center gap-3 flex-1 min-w-0 text-xs text-muted">
          {chunk.mel_sec != null && <span className="shrink-0">{chunk.mel_sec.toFixed(2)}s</span>}
          <span className="shrink-0">{chunk.token_count} tok</span>
          <span className="shrink-0 text-blue-400/70">pauzy {pauseSec.toFixed(2)}s</span>
          <span className="text-slate-400 truncate min-w-0">{chunk.text}</span>
        </div>
        <span className="text-muted text-xs shrink-0">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="border-t border-border px-3 py-2 bg-surface2/50">
          <p className="text-xs text-slate-300 mb-2 leading-relaxed">{chunk.text}</p>
          {/* Pause bar */}
          {chunk.mel_sec != null && chunk.mel_sec > 0 && (
            <div className="mb-2">
              <div className="flex justify-between text-xs text-muted mb-0.5">
                <span>treść {contentSec.toFixed(2)}s</span>
                <span>pauzy {pauseSec.toFixed(2)}s</span>
              </div>
              <div className="h-1.5 rounded-full bg-surface2 overflow-hidden flex">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${Math.min(100, (contentSec / chunk.mel_sec) * 100)}%` }}
                />
                <div className="h-full bg-blue-500/60 flex-1" />
              </div>
            </div>
          )}
          <table className="w-full">
            <thead>
              <tr className="text-xs text-muted border-b border-border">
                <th className="text-left pb-1 pr-3 font-normal">token</th>
                <th className="text-right pb-1 pr-3 font-normal">klatki</th>
                <th className="text-right pb-1 font-normal">sekundy</th>
              </tr>
            </thead>
            <tbody>
              {chunk.tokens.map((t, i) => <TokenRow key={i} t={t} />)}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function TextTTSPanel() {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState('');
  const [ttsModels, setTtsModels] = useState<TTSModelProfile[]>([]);
  const [ttsModel, setTtsModel] = useState('');
  const [text, setText] = useState('');
  const [lang, setLang] = useState('pl');
  const [speed, setSpeed] = useState(1.0);
  const [durScale, setDurScale] = useState(1.0);
  const [melFirst, setMelFirst] = useState(8);
  const [melSecond, setMelSecond] = useState(3);
  const [tNoise, setTNoise] = useState(0.12);
  const [digSilence, setDigSilence] = useState(true);
  const [pauseEdge, setPauseEdge] = useState(10);
  const [continuityMs, setContinuityMs] = useState(128);
  const [showAdv, setShowAdv] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [jobId, setJobId] = useState('');
  const [debugLog, setDebugLog] = useState('');
  const [chunks, setChunks] = useState<TTSChunk[]>([]);
  const [showDebug, setShowDebug] = useState(false);

  useEffect(() => {
    listSpeakers().then(list => {
      setSpeakers(list);
      if (list.length > 0) setSpeaker(list[0].label);
    });
    listTTSModels().then(data => {
      setTtsModels(data.models);
      setTtsModel(data.active || data.default || data.models[0]?.key || '');
    });
  }, []);

  const run = async () => {
    if (!text.trim() || !speaker) return;
    setRunning(true); setError(''); setProgress(0); setMessage('Inicjalizuję…');
    setAudioUrl(''); setDebugLog(''); setChunks([]);
    try {
      const jid = await submitTextTTS({
        text, speaker_label: speaker, tts_model_profile: ttsModel, language: lang,
        speed, dur_scale: durScale,
        mel_steps_first: melFirst, mel_steps_second: melSecond,
        mel_twopass_t_noise: tNoise, digital_silence: digSilence, pause_edge_frames: pauseEdge,
        short_continuity_ms: continuityMs,
        emotion_group: 'neutral',
        emotion_strength: 0,
      });
      setJobId(jid);
      await new Promise<void>((resolve, reject) => {
        streamJob(jid, (ev) => {
          if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
          else if (ev.type === 'done') {
            const res = ev.result as TTSResult | undefined;
            setAudioUrl(`${BASE}/jobs/${jid}/audio`);
            setDebugLog(res?.debug_log ?? '');
            setChunks(res?.chunks ?? []);
            resolve();
          }
          else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd TTS'));
        });
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 max-w-2xl mx-auto">
      <h2 className="text-base font-semibold">TTS z tekstu</h2>

      <div className="space-y-3">
        <div>
          <label className="text-xs text-muted block mb-1">Tekst</label>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            rows={4}
            placeholder="Wpisz tekst do syntezy…"
            className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm resize-y"
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted block mb-1">Lektor ({speakers.length})</label>
            <select value={speaker} onChange={e => setSpeaker(e.target.value)}
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm">
              {speakers.map(s => <option key={s.id} value={s.label}>{s.label}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted block mb-1">Model TTS</label>
            <select value={ttsModel} onChange={e => setTtsModel(e.target.value)}
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm">
              {ttsModels.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted block mb-1">Język</label>
            <select value={lang} onChange={e => setLang(e.target.value)}
              className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm">
              <option value="pl">Polski</option>
              <option value="en">English</option>
            </select>
          </div>
        </div>

        <Slider label="Szybkość mówienia" value={speed} min={0.5} max={2.0} step={0.05}
          onChange={setSpeed} />

        <button onClick={() => setShowAdv(v => !v)}
          className="text-xs text-muted hover:text-slate-300 flex items-center gap-1">
          {showAdv ? '▲' : '▼'} Parametry TTS
        </button>
        {showAdv && (
          <div className="space-y-3 p-3 bg-surface2 rounded-lg border border-border">
            <div className="text-xs text-muted">
              Duration model: <span className="text-slate-300">checkpoint deterministic</span>
            </div>
            <Slider label="Duration Scale" value={durScale} min={0.6} max={1.8} step={0.01}
              onChange={setDurScale} />
            <Slider label="Mel Steps First" value={melFirst} min={1} max={24} step={1}
              onChange={setMelFirst} />
            <Slider label="Mel Steps Second" value={melSecond} min={0} max={24} step={1}
              onChange={setMelSecond} />
            <Slider label="Two-pass T Noise" value={tNoise} min={0} max={0.5} step={0.01}
              onChange={setTNoise} />
            <Slider label="Pause edge frames" value={pauseEdge} min={0} max={30} step={1}
              onChange={setPauseEdge} />
            <Slider label="Flow continuity prefix (ms)" value={continuityMs} min={0} max={1000} step={1}
              onChange={setContinuityMs} />
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input type="checkbox" checked={digSilence} onChange={e => setDigSilence(e.target.checked)}
                className="accent-accent" />
              Digital silence on pause middles
            </label>
          </div>
        )}

        <button onClick={run} disabled={running || !text.trim() || !speaker || !ttsModel}
          className="w-full py-2.5 bg-accent text-white font-semibold rounded-lg hover:bg-blue-400 disabled:opacity-40 transition-colors">
          {running ? 'Syntezuję…' : '▶ Syntetyzuj'}
        </button>

        {(running || error) && (
          <JobProgress message={message} progress={progress} error={error || undefined} />
        )}

        {audioUrl && !running && (
          <div className="space-y-2">
            <audio controls src={audioUrl} className="w-full" autoPlay />
            <a href={audioUrl} download={`tts_${jobId}.wav`}
              className="text-xs px-3 py-1.5 rounded bg-surface2 border border-border hover:bg-border transition-colors inline-block">
              Pobierz WAV
            </a>
            {debugLog && (
              <div className="text-xs text-muted break-all">log JSON: {debugLog}</div>
            )}
          </div>
        )}

        {chunks.length > 0 && !running && (
          <div className="space-y-2">
            <button
              onClick={() => setShowDebug(v => !v)}
              className="flex items-center gap-2 text-xs text-muted hover:text-slate-300 transition-colors"
            >
              {showDebug ? '▲' : '▼'}
              Segmenty i duracje
              <span className="px-1.5 py-0.5 bg-surface2 rounded text-slate-400">
                {chunks.length} {chunks.length === 1 ? 'segment' : 'segmentów'}
              </span>
            </button>

            {showDebug && (
              <div className="space-y-2">
                {chunks.map((ch, i) => <ChunkCard key={i} chunk={ch} idx={i} />)}
                <p className="text-xs text-muted px-1">
                  Niebieski = pauza {'<sp>'} · Pomarańczowy = bardzo krótki token
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
