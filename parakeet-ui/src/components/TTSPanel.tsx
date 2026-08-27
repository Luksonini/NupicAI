'use client';
import { useEffect, useRef, useState } from 'react';
import type { MutableRefObject, RefObject } from 'react';
import type { Segment, Speaker, DubResult, TTSModelProfile } from '@/lib/types';
import { listSpeakers, listTTSModels, uploadVoicePrompt, submitDub, streamJob, dubAudioUrl, mixVideoUrl } from '@/lib/api';
import JobProgress from './JobProgress';

interface Props {
  segments: Segment[];
  targetLang: string;
  transcribeJobId: string;
  originalSrc: string;         // blob URL of uploaded file
  hasVideo: boolean;
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

function fmt(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export default function TTSPanel({ segments, targetLang, transcribeJobId, originalSrc, hasVideo }: Props) {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState('');
  const [ttsModels, setTtsModels] = useState<TTSModelProfile[]>([]);
  const [ttsModel, setTtsModel] = useState('');
  const [baseSpeed, setBaseSpeed] = useState(1.0);
  const [maxSpeed, setMaxSpeed] = useState(1.3);
  const [extraTail, setExtraTail] = useState(0.0);
  const [durScale, setDurScale] = useState(1.0);
  const [melFirst, setMelFirst] = useState(8);
  const [melSecond, setMelSecond] = useState(3);
  const [tNoise, setTNoise] = useState(0.12);
  const [digSilence, setDigSilence] = useState(true);
  const [pauseEdge, setPauseEdge] = useState(10);
  const [continuityMs, setContinuityMs] = useState(128);
  const [editableSegments, setEditableSegments] = useState<Segment[]>(segments);
  const [showAdv, setShowAdv] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [promptStatus, setPromptStatus] = useState('');
  const [dubJobId, setDubJobId] = useState('');
  const [result, setResult] = useState<DubResult | null>(null);
  const [audioMode, setAudioMode] = useState<'original' | 'dubbed'>('original');
  const originalRef = useRef<HTMLAudioElement | HTMLVideoElement | null>(null);
  const dubbedAudioRef = useRef<HTMLAudioElement | null>(null);
  const dubbedVideoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    setEditableSegments(segments.map(s => ({ ...s, translation: s.translation ?? s.text })));
    setResult(null);
    setDubJobId('');
  }, [segments]);

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
    setRunning(true); setError(''); setProgress(0); setMessage('Inicjalizuję…');
    try {
      const jobId = await submitDub({
        segments: editableSegments,
        speaker_label: speaker,
        tts_model_profile: ttsModel,
        transcribe_job_id: transcribeJobId,
        target_lang: targetLang,
        base_speed: baseSpeed,
        max_adaptive_speed: maxSpeed,
        extra_tail_sec: extraTail,
        dur_scale: durScale,
        mel_steps_first: melFirst,
        mel_steps_second: melSecond,
        mel_twopass_t_noise: tNoise,
        digital_silence: digSilence,
        pause_edge_frames: pauseEdge,
        short_continuity_ms: continuityMs,
        emotion_group: 'neutral',
        emotion_strength: 0,
      });
      setDubJobId(jobId);
      await new Promise<void>((resolve, reject) => {
        streamJob(jobId, (ev) => {
          if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
          else if (ev.type === 'done') { setResult(ev.result as DubResult); resolve(); }
          else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd TTS'));
        });
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const addVoicePrompt = async (file: File | null) => {
    if (!file) return;
    setPromptStatus('Koduję voice prompt…');
    setError('');
    try {
      const sp = await uploadVoicePrompt(file);
      setSpeakers(prev => [sp, ...prev.filter(x => x.label !== sp.label)]);
      setSpeaker(sp.label);
      setPromptStatus(`Dodano prompt: ${sp.label}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setPromptStatus('');
      setError(msg);
    }
  };

  const dubbedUrl = dubJobId ? dubAudioUrl(dubJobId) : '';
  const seekTo = (sec: number) => {
    const primary = audioMode === 'dubbed' ? (dubbedVideoRef.current ?? dubbedAudioRef.current) : originalRef.current;
    if (primary) {
      primary.currentTime = sec;
      primary.play().catch(() => {});
    }
    if (audioMode === 'dubbed' && dubbedAudioRef.current) {
      dubbedAudioRef.current.currentTime = sec;
      dubbedAudioRef.current.play().catch(() => {});
    }
  };

  const updateSegmentText = (idx: number, value: string) => {
    setEditableSegments(prev => prev.map((seg, i) => (
      i === idx ? { ...seg, translation: value, text: seg.text } : seg
    )));
  };

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-base font-semibold">Dubbing TTS</h2>

      {(!result || true) && (
        <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
          {/* Speaker */}
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
          <div>
            <label className="text-xs text-muted block mb-1">Własny voice prompt (WAV/MP3/MP4)</label>
            <input
              type="file"
              accept="audio/*,video/*,.mp4,.mkv,.mov,.webm,.wav,.mp3,.flac,.m4a"
              onChange={e => addVoicePrompt(e.target.files?.[0] ?? null)}
              className="block w-full text-xs text-muted file:mr-3 file:px-3 file:py-1.5 file:rounded file:border-0 file:bg-surface2 file:text-slate-200 hover:file:bg-border"
            />
            {promptStatus && <p className="text-xs text-green-300 mt-1">{promptStatus}</p>}
          </div>
          {/* Speed */}
          <Slider label="Base speed" value={baseSpeed} min={0.7} max={1.4} step={0.05}
            onChange={setBaseSpeed} />
          <Slider label="Max adaptive speed" value={maxSpeed} min={1.0} max={1.3} step={0.05}
            onChange={setMaxSpeed} />
          <Slider label="Extra tail (s)" value={extraTail} min={0} max={1} step={0.05}
            onChange={setExtraTail} />

          {/* Advanced */}
          <button onClick={() => setShowAdv(v => !v)}
            className="text-xs text-muted hover:text-slate-300 flex items-center gap-1">
            {showAdv ? '▲' : '▼'} TTS params
          </button>
          {showAdv && (
            <div className="space-y-3 pt-1">
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

          <button onClick={run} disabled={running || !speaker || !ttsModel}
            className="w-full py-2.5 bg-accent text-white font-semibold rounded-lg hover:bg-blue-400 disabled:opacity-40 transition-colors">
            {running ? 'Syntezuję…' : '▶ Dubbinguj'}
          </button>

          {(running || error) && (
            <JobProgress message={message} progress={progress} error={error || undefined} />
          )}
        </div>
      )}

      <div className="bg-surface rounded-xl border border-border p-4 space-y-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Segmenty TTS</h3>
          <span className="text-xs text-muted">{editableSegments.length}</span>
        </div>
        <div className="max-h-96 overflow-y-auto space-y-2 pr-1">
          {editableSegments.map((seg, i) => (
            <div key={`${seg.index}-${i}`} className="border border-border rounded-lg p-2 bg-surface2">
              <div className="flex items-center gap-2 mb-1">
                <button
                  onClick={() => seekTo(Number(seg.start || 0))}
                  className="text-xs px-2 py-1 rounded border border-border hover:border-accent text-accent">
                  {fmt(Number(seg.start || 0))}
                </button>
                <span className="text-xs text-muted truncate">{seg.source_text ?? seg.text}</span>
              </div>
              <div className="text-[11px] text-muted mb-1">Tekst dubbingu / tłumaczenie</div>
              <textarea
                value={seg.translation ?? seg.text}
                onChange={e => updateSegmentText(i, e.target.value)}
                className="w-full min-h-[64px] bg-surface border border-border rounded-md px-2 py-1.5 text-sm resize-y outline-none focus:border-accent"
              />
            </div>
          ))}
        </div>
      </div>

      {result && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-xs text-muted">
            <span className="bg-green-800/40 text-green-300 px-2 py-0.5 rounded font-semibold">
              Dubbing gotowy — {result.duration.toFixed(1)} s
            </span>
            <button onClick={() => { setResult(null); setDubJobId(''); setProgress(0); setMessage(''); }}
              className="ml-auto hover:text-slate-200">Nowy dubbing</button>
          </div>

          {result.segments && result.segments.length > 0 && (
            <details className="bg-surface2 rounded-xl border border-border p-3">
              <summary className="text-xs text-muted cursor-pointer">Debug duracji segmentów</summary>
              <div className="mt-2 space-y-2 max-h-80 overflow-y-auto">
                {result.debug_log && (
                  <div className="text-xs text-muted break-all">log JSON: {result.debug_log}</div>
                )}
                {result.segments.map((seg, i) => {
                  const chunks = ((seg.tts_debug as Record<string, unknown> | undefined)?.chunks ?? []) as Array<Record<string, unknown>>;
                  const summary = (seg.tts_debug_summary ?? {}) as Record<string, unknown>;
                  const warnings = (summary.warnings ?? []) as string[];
                  const lowTokens = (summary.low_tokens ?? []) as Array<Record<string, unknown>>;
                  return (
                    <details key={i} className="border border-border rounded p-2 bg-surface">
                      <summary className="text-xs cursor-pointer">
                        #{String(seg.index ?? i)} start={String(seg.start)}s audio={String(seg.audio_duration)}s budget={String(seg.target_budget)}s speed={String(seg.speed)} chunks={String(summary.chunk_count ?? chunks.length)} low={String(summary.low_token_count ?? 0)}
                      </summary>
                      <div className="mt-2 space-y-2">
                        <div className="text-xs text-muted">
                          tekst={String(summary.text_len ?? '')} znaków · tokeny={String(summary.total_allowed_tokens ?? '')} · retries={String(seg.fit_retries)} · over={String(seg.over_budget)}s
                        </div>
                        {warnings.length > 0 && (
                          <div className="text-xs text-amber-300 space-y-1">
                            {warnings.map((w, wi) => <div key={wi}>⚠ {w}</div>)}
                          </div>
                        )}
                        {lowTokens.length > 0 && (
                          <div className="text-xs text-red-300 break-words">
                            Małe duracje: {lowTokens.slice(0, 24).map((t, ti) => (
                              <span key={ti}>{String(t.token)}:{String(t.dur)}f/{String(t.dur_sec)}s </span>
                            ))}
                          </div>
                        )}
                        {chunks.map((ch, ci) => {
                          const dbg = (ch.debug ?? {}) as Record<string, unknown>;
                          const durs = (dbg.durations ?? []) as Array<Record<string, unknown>>;
                          return (
                            <div key={ci} className="text-xs">
                              <div className="text-muted mb-1">
                                chunk {ci + 1}: pred={String(dbg.pred_sec ?? '')}s mel={String(dbg.mel_sec ?? '')}s prefix={String(dbg.prefix_sec ?? '')}s · {String(ch.text ?? '')}
                              </div>
                              <div className="font-mono text-[11px] leading-relaxed break-words">
                                {durs.filter(d => d.allowed).map((d, di) => (
                                  <span key={di} className={Number(d.dur ?? 0) <= 1.25 ? 'text-red-300' : 'text-slate-300'}>
                                    {String(d.token)}:{String(d.dur)}f/{String(d.dur_sec)}s{' '}
                                  </span>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </details>
                  );
                })}
              </div>
            </details>
          )}

          {/* Video player with audio toggle */}
          <div className="bg-surface2 rounded-xl border border-border p-3 space-y-2">
            {hasVideo ? (
              <VideoPlayer originalSrc={originalSrc} dubbedAudioUrl={dubbedUrl}
                audioMode={audioMode}
                originalRef={originalRef}
                dubbedVideoRef={dubbedVideoRef}
                dubbedAudioRef={dubbedAudioRef}
                onToggle={() => setAudioMode(m => m === 'original' ? 'dubbed' : 'original')} />
            ) : (
              <div className="space-y-2">
                <p className="text-xs text-muted">Oryginalny plik audio:</p>
                <audio ref={originalRef as RefObject<HTMLAudioElement>} controls src={originalSrc} className="w-full" />
                <p className="text-xs text-muted">Dubbed audio:</p>
                <audio ref={dubbedAudioRef} controls src={dubbedUrl} className="w-full" />
              </div>
            )}
          </div>

          {/* Download buttons */}
          <div className="flex gap-2 flex-wrap">
            <a href={dubbedUrl} download="dubbed.wav"
              className="text-xs px-3 py-1.5 rounded bg-surface2 border border-border hover:bg-border transition-colors">
              Pobierz WAV
            </a>
            {transcribeJobId && (
              <a href={mixVideoUrl(dubJobId, transcribeJobId)} download="dubbed_video.mp4"
                className="text-xs px-3 py-1.5 rounded bg-accent/20 border border-accent/40 hover:bg-accent/30 transition-colors text-accent">
                Pobierz Mixed MP4
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function VideoPlayer({ originalSrc, dubbedAudioUrl, audioMode, originalRef, dubbedVideoRef, dubbedAudioRef, onToggle }: {
  originalSrc: string;
  dubbedAudioUrl: string;
  audioMode: 'original' | 'dubbed';
  originalRef: MutableRefObject<HTMLAudioElement | HTMLVideoElement | null>;
  dubbedVideoRef: MutableRefObject<HTMLVideoElement | null>;
  dubbedAudioRef: MutableRefObject<HTMLAudioElement | null>;
  onToggle: () => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <button onClick={onToggle}
          className={`text-xs px-3 py-1 rounded border transition-colors ${
            audioMode === 'original'
              ? 'bg-accent text-white border-accent'
              : 'bg-surface2 text-muted border-border hover:border-accent/50'
          }`}>
          Oryginalne audio
        </button>
        <button onClick={onToggle}
          className={`text-xs px-3 py-1 rounded border transition-colors ${
            audioMode === 'dubbed'
              ? 'bg-accent text-white border-accent'
              : 'bg-surface2 text-muted border-border hover:border-accent/50'
          }`}>
          Dubbing
        </button>
      </div>
      {audioMode === 'original' ? (
        <video ref={originalRef as RefObject<HTMLVideoElement>} key="orig" controls src={originalSrc}
          className="w-full rounded-lg" style={{ maxHeight: 320 }} />
      ) : (
        <SyncedPlayer videoSrc={originalSrc} audioSrc={dubbedAudioUrl}
          videoRef={dubbedVideoRef} audioRef={dubbedAudioRef} />
      )}
    </div>
  );
}

function SyncedPlayer({ videoSrc, audioSrc, videoRef, audioRef }: {
  videoSrc: string;
  audioSrc: string;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  audioRef: MutableRefObject<HTMLAudioElement | null>;
}) {
  return (
    <div className="space-y-1">
      <video ref={videoRef} controls src={videoSrc} muted
        className="w-full rounded-lg" style={{ maxHeight: 260 }}
        onPlay={e => {
          const vid = e.currentTarget;
          const aud = vid.parentElement?.querySelector('audio') as HTMLAudioElement | null;
          if (aud) { aud.currentTime = vid.currentTime; aud.play().catch(() => {}); }
        }}
        onPause={e => {
          const aud = e.currentTarget.parentElement?.querySelector('audio') as HTMLAudioElement | null;
          if (aud) aud.pause();
        }}
        onSeeked={e => {
          const vid = e.currentTarget;
          const aud = vid.parentElement?.querySelector('audio') as HTMLAudioElement | null;
          if (aud) aud.currentTime = vid.currentTime;
        }}
      />
      <audio ref={audioRef} src={audioSrc} />
      <p className="text-xs text-muted text-center">Wideo (wyciszone) + dubbed audio</p>
    </div>
  );
}
