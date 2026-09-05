'use client';

import { useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import {
  AlertTriangle, CheckCircle2, Download, Gauge, Mic2, Music2, Play,
  Link2, Redo2, RefreshCw, RotateCcw, Scissors, SlidersHorizontal, Undo2, Upload, UserRound,
  Volume2, WandSparkles,
} from 'lucide-react';
import type { Segment, Speaker, DubResult } from '@/lib/types';
import {
  ApiRequestError, listSpeakers, listTTSModels, uploadVoicePrompt, submitDub, streamJob,
  dubAudioUrl, mixAudioUrl, mixVideoUrl,
} from '@/lib/api';
import JobProgress from './JobProgress';
import QuotaExceededNotice from './QuotaExceededNotice';
import { useLocale } from '@/lib/locale';

interface Props {
  segments: Segment[];
  targetLang: string;
  transcribeJobId: string;
  originalSrc: string;
  hasVideo: boolean;
}

function Slider({ label, value, display, min, max, step, onChange }: {
  label: string; value: number; display?: string; min: number; max: number; step: number;
  onChange: (value: number) => void;
}) {
  return <label className="slider-field"><span><b>{label}</b><output>{display ?? value}</output></span>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} />
  </label>;
}

function fmt(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function freshSegmentId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : `segment-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function prepareSegments(segments: Segment[]): Segment[] {
  return segments.map((segment, index) => ({
    ...segment,
    translation: segment.translation ?? segment.text,
    segment_id: segment.segment_id ?? freshSegmentId(),
    seed: segment.seed ?? 1234 + index,
    render_nonce: segment.render_nonce ?? 0,
  }));
}

function cloneSegments(segments: Segment[]): Segment[] {
  return segments.map(segment => ({
    ...segment,
    words: segment.words?.map(word => ({ ...word })),
  }));
}

function splitTextNearRatio(text: string, ratio: number): [string, string] {
  const value = text.trim();
  if (!value) return ['', ''];
  const wanted = Math.max(1, Math.min(value.length - 1, Math.round(value.length * ratio)));
  const candidates = [value.lastIndexOf(' ', wanted), value.indexOf(' ', wanted)].filter(pos => pos > 0);
  const cut = candidates.sort((a, b) => Math.abs(a - wanted) - Math.abs(b - wanted))[0] ?? wanted;
  return [value.slice(0, cut).trim(), value.slice(cut).trim()];
}

export default function TTSPanel({ segments, targetLang, transcribeJobId, originalSrc, hasVideo }: Props) {
  const { locale, t } = useLocale();
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState('');
  const [ttsModel, setTtsModel] = useState('');
  const [speechLang, setSpeechLang] = useState(targetLang === 'en' ? 'en' : 'pl');
  const [flowSettings, setFlowSettings] = useState({ mel_steps_first: 8, mel_steps_second: 3, mel_twopass_t_noise: 0.12 });
  const [baseSpeed, setBaseSpeed] = useState(1.0);
  const [maxSpeed, setMaxSpeed] = useState(1.3);
  const [originalGain, setOriginalGain] = useState(0.22);
  const [dubbingGain, setDubbingGain] = useState(1.0);
  const [ducking, setDucking] = useState(0.65);
  const [editableSegments, setEditableSegments] = useState<Segment[]>(segments);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [quotaExceeded, setQuotaExceeded] = useState(false);
  const [promptStatus, setPromptStatus] = useState('');
  const [dubJobId, setDubJobId] = useState('');
  const [result, setResult] = useState<DubResult | null>(null);
  const [audioMode, setAudioMode] = useState<'original' | 'mix' | 'voice'>('original');
  const mediaRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const textareaRefs = useRef<Record<string, HTMLTextAreaElement | null>>({});
  const textEditStartRef = useRef<Record<string, Segment[]>>({});
  const undoStackRef = useRef<Segment[][]>([]);
  const redoStackRef = useRef<Segment[][]>([]);
  const [historySize, setHistorySize] = useState({ undo: 0, redo: 0 });

  const refreshHistorySize = () => setHistorySize({
    undo: undoStackRef.current.length,
    redo: redoStackRef.current.length,
  });

  const rememberSnapshot = (snapshot: Segment[]) => {
    undoStackRef.current.push(cloneSegments(snapshot));
    if (undoStackRef.current.length > 100) undoStackRef.current.shift();
    redoStackRef.current = [];
    refreshHistorySize();
  };

  const commitSegments = (next: Segment[]) => {
    rememberSnapshot(editableSegments);
    setEditableSegments(next);
    setResult(null);
    setError('');
  };

  const undoEdit = () => {
    const previous = undoStackRef.current.pop();
    if (!previous) return;
    redoStackRef.current.push(cloneSegments(editableSegments));
    setEditableSegments(cloneSegments(previous));
    setResult(null); setError(''); refreshHistorySize();
  };

  const redoEdit = () => {
    const next = redoStackRef.current.pop();
    if (!next) return;
    undoStackRef.current.push(cloneSegments(editableSegments));
    setEditableSegments(cloneSegments(next));
    setResult(null); setError(''); refreshHistorySize();
  };

  useEffect(() => {
    setEditableSegments(prepareSegments(segments));
    undoStackRef.current = []; redoStackRef.current = []; textEditStartRef.current = {};
    setHistorySize({ undo: 0, redo: 0 });
    setResult(null); setDubJobId('');
  }, [segments]);

  useEffect(() => {
    setSpeechLang(targetLang === 'en' ? 'en' : 'pl');
  }, [targetLang]);

  useEffect(() => {
    const handleHistoryShortcut = (event: KeyboardEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.matches('input, textarea, select, [contenteditable="true"]')) return;
      if (event.key.toLowerCase() === 'z' && event.shiftKey) {
        event.preventDefault(); redoEdit();
      } else if (event.key.toLowerCase() === 'z') {
        event.preventDefault(); undoEdit();
      } else if (event.key.toLowerCase() === 'y') {
        event.preventDefault(); redoEdit();
      }
    };
    window.addEventListener('keydown', handleHistoryShortcut);
    return () => window.removeEventListener('keydown', handleHistoryShortcut);
  });

  useEffect(() => {
    listSpeakers().then(list => { setSpeakers(list); if (list.length) setSpeaker(list[0].label); });
    listTTSModels().then(data => {
      setTtsModel(data.default || data.active || data.models[0]?.key || '');
      setFlowSettings(data.flow_defaults);
    });
  }, []);

  const run = async (segmentsOverride?: Segment[]) => {
    const renderSegments = segmentsOverride ?? editableSegments;
    setRunning(true); setError(''); setQuotaExceeded(false); setProgress(0); setMessage(locale === 'pl' ? 'Przygotowuję rendering…' : 'Preparing render…');
    try {
      const jobId = await submitDub({
        segments: renderSegments, speaker_label: speaker, tts_model_profile: ttsModel,
        transcribe_job_id: transcribeJobId, reuse_dub_job_id: dubJobId, target_lang: speechLang,
        base_speed: baseSpeed, max_adaptive_speed: maxSpeed, extra_tail_sec: 0,
        dur_scale: 1, ...flowSettings,
        digital_silence: true, pause_edge_frames: 10, short_continuity_ms: 0,
        emotion_group: 'neutral', emotion_strength: 0,
        original_gain: originalGain, dubbing_gain: dubbingGain, ducking_strength: ducking,
      });
      setDubJobId(jobId);
      await new Promise<void>((resolve, reject) => streamJob(jobId, ev => {
        if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
        else if (ev.type === 'done') {
          setResult(ev.result as DubResult); setAudioMode('mix');
          window.dispatchEvent(new Event('nupicai-usage-changed')); resolve();
        }
        else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd TTS'));
      }));
    } catch (e: unknown) {
      setMessage('');
      if (e instanceof ApiRequestError && e.status === 402) setQuotaExceeded(true);
      else setError(e instanceof Error ? e.message : String(e));
    } finally { setRunning(false); }
  };

  const regenerateSegment = (index: number) => {
    const next = editableSegments.map((item, idx) => idx === index
      ? { ...item, render_nonce: Number(item.render_nonce ?? 0) + 1 }
      : item);
    commitSegments(next);
    void run(next);
  };

  const mergeWithNext = (index: number) => {
    if (index >= editableSegments.length - 1) return;
    const first = editableSegments[index], second = editableSegments[index + 1];
    const merged: Segment = {
      ...first,
      end: second.end,
      text: `${first.text ?? ''} ${second.text ?? ''}`.trim(),
      source_text: `${first.source_text ?? first.text ?? ''} ${second.source_text ?? second.text ?? ''}`.trim(),
      translation: `${first.translation ?? first.text ?? ''} ${second.translation ?? second.text ?? ''}`.trim(),
      words: [...(first.words ?? []), ...(second.words ?? [])],
      segment_id: freshSegmentId(),
      seed: 1234 + index,
      render_nonce: 0,
      speaker_label: first.speaker_label || second.speaker_label,
    };
    commitSegments([...editableSegments.slice(0, index), merged, ...editableSegments.slice(index + 2)]);
  };

  const splitAtCursor = (index: number) => {
    const segment = editableSegments[index];
    const text = String(segment.translation ?? segment.text ?? '').trim();
    const cursor = textareaRefs.current[String(segment.segment_id)]?.selectionStart ?? -1;
    if (cursor <= 0 || cursor >= text.length) {
      setError(locale === 'pl' ? 'Ustaw kursor w miejscu podziału tekstu.' : 'Place the cursor where the text should be split.');
      return;
    }
    const [left, right] = splitTextNearRatio(text, cursor / text.length);
    if (!left || !right) return;
    const ratio = left.length / Math.max(1, left.length + right.length);
    const middle = Number(segment.start) + (Number(segment.end) - Number(segment.start)) * ratio;
    const [sourceLeft, sourceRight] = splitTextNearRatio(
      String(segment.source_text ?? segment.text ?? ''), ratio,
    );
    const common = { ...segment, words: undefined, render_nonce: 0 };
    const first: Segment = { ...common, end: middle, text: sourceLeft || left, source_text: sourceLeft, translation: left, segment_id: freshSegmentId(), seed: 1234 + index };
    const second: Segment = { ...common, start: middle, text: sourceRight || right, source_text: sourceRight, translation: right, segment_id: freshSegmentId(), seed: 2234 + index };
    commitSegments([...editableSegments.slice(0, index), first, second, ...editableSegments.slice(index + 1)]);
  };

  const restoreSegments = () => {
    commitSegments(prepareSegments(segments));
  };

  const beginTextEdit = (segmentId: string) => {
    if (!textEditStartRef.current[segmentId]) {
      textEditStartRef.current[segmentId] = cloneSegments(editableSegments);
    }
  };

  const finishTextEdit = (segmentId: string) => {
    const snapshot = textEditStartRef.current[segmentId];
    delete textEditStartRef.current[segmentId];
    if (!snapshot) return;
    const before = snapshot.find(item => String(item.segment_id) === segmentId)?.translation ?? '';
    const after = editableSegments.find(item => String(item.segment_id) === segmentId)?.translation ?? '';
    if (before !== after) rememberSnapshot(snapshot);
  };

  const addVoicePrompt = async (file: File | null) => {
    if (!file) return;
    setPromptStatus(locale === 'pl' ? 'Koduję próbkę głosu…' : 'Encoding voice sample…'); setError('');
    try {
      const custom = await uploadVoicePrompt(file);
      setSpeakers(prev => [custom, ...prev.filter(item => item.label !== custom.label)]);
      setSpeaker(custom.label); setPromptStatus(custom.display_name ?? custom.label);
    } catch (e: unknown) {
      setPromptStatus(''); setError(e instanceof Error ? e.message : String(e));
    }
  };

  const voiceUrl = dubJobId ? dubAudioUrl(dubJobId) : '';
  const mixedUrl = dubJobId ? mixAudioUrl(dubJobId) : '';
  const selectedAudio = audioMode === 'mix' ? mixedUrl : audioMode === 'voice' ? voiceUrl : originalSrc;
  const warningCount = (result?.segments ?? []).filter(item => Number(item.over_budget ?? 0) > 0.05).length;

  const seekTo = (sec: number) => {
    if (mediaRef.current) mediaRef.current.currentTime = sec;
    if (audioRef.current) { audioRef.current.currentTime = sec; void audioRef.current.play(); }
  };

  return (
    <div className="dub-layout">
      <section className="dub-main">
        <div className="media-stage">
          <div className="section-toolbar">
            <div className="panel-heading compact"><span className="icon-box"><Play size={18} /></span><div><h2>{t('preview')}</h2><p>{fmt(segments.at(-1)?.end ?? 0)}</p></div></div>
            {result && <div className="segmented-control">
              <button className={audioMode === 'original' ? 'active' : ''} onClick={() => setAudioMode('original')}>{t('original')}</button>
              <button className={audioMode === 'mix' ? 'active' : ''} onClick={() => setAudioMode('mix')}>{t('mix')}</button>
              <button className={audioMode === 'voice' ? 'active' : ''} onClick={() => setAudioMode('voice')}>{t('voiceOnly')}</button>
            </div>}
          </div>
          {hasVideo ? (
            <SyncedVideo videoSrc={originalSrc} audioSrc={selectedAudio} original={audioMode === 'original'} videoRef={mediaRef} audioRef={audioRef} />
          ) : <audio ref={audioRef} controls src={selectedAudio} className="w-full" />}
        </div>

        {result && <div className={`render-summary ${warningCount ? 'warning' : ''}`}>
          {warningCount ? <AlertTriangle size={18} /> : <CheckCircle2 size={18} />}
          <div><strong>{warningCount ? `${warningCount} ${locale === 'pl' ? 'segmentów wymaga odsłuchu' : 'segments need review'}` : (locale === 'pl' ? 'Rendering gotowy' : 'Render complete')}</strong><span>{result.duration.toFixed(1)} s · {editableSegments.length} {locale === 'pl' ? 'segmentów' : 'segments'}{Number(result.reused_segments ?? 0) > 0 ? ` · ${result.reused_segments} ${locale === 'pl' ? 'użytych ponownie' : 'reused'}` : ''}</span></div>
          <div className="download-group">
            <a className="button button-secondary" href={voiceUrl} download="dubbing.wav"><Download size={15} />{locale === 'pl' ? 'Głos' : 'Voice'}</a>
            {result.mixed_audio_path && <a className="button button-secondary" href={mixedUrl} download="mix.wav"><Download size={15} />{t('mix')} WAV</a>}
            {hasVideo && <a className="button button-primary" href={mixVideoUrl(dubJobId, transcribeJobId)} download="dubbing.mp4"><Download size={15} />MP4</a>}
          </div>
        </div>}

        <div className="segment-editor">
          <div className="section-toolbar"><div><h2>{t('dubbingSegments')}</h2><p>{editableSegments.length}</p></div><div className="segment-history-actions">
            <button className="icon-button" disabled={!historySize.undo} onClick={undoEdit} title={locale === 'pl' ? 'Cofnij zmianę' : 'Undo change'}><Undo2 size={15} /></button>
            <button className="icon-button" disabled={!historySize.redo} onClick={redoEdit} title={locale === 'pl' ? 'Ponów zmianę' : 'Redo change'}><Redo2 size={15} /></button>
            <button className="button button-ghost button-small" onClick={restoreSegments}><RotateCcw size={14} />{locale === 'pl' ? 'Przywróć podział' : 'Restore split'}</button>
          </div></div>
          <div className="segment-table">
            {editableSegments.map((seg, i) => {
              const budget = Math.max(0.1, Number(seg.end) - Number(seg.start));
              const charsPerSec = String(seg.translation ?? seg.text).length / budget;
              const segmentId = String(seg.segment_id);
              return <article className="edit-segment-row" key={segmentId}>
                <button className="timecode" onClick={() => seekTo(Number(seg.start))}>{fmt(Number(seg.start))}</button>
                <div className="segment-copy"><p className="source-line">{seg.source_text ?? seg.text}</p>
                  <textarea ref={node => { textareaRefs.current[segmentId] = node; }} value={seg.translation ?? seg.text}
                    onFocus={() => beginTextEdit(segmentId)} onBlur={() => finishTextEdit(segmentId)}
                    onChange={e => { setEditableSegments(prev => prev.map((item, idx) => idx === i ? { ...item, translation: e.target.value } : item)); setResult(null); }} />
                  <div className="segment-tools">
                    <label title={locale === 'pl' ? 'Głos tylko dla tego fragmentu' : 'Voice for this segment only'}><UserRound size={14} /><select value={seg.speaker_label ?? ''} onChange={e => commitSegments(editableSegments.map((item, idx) => idx === i ? { ...item, speaker_label: e.target.value || undefined } : item))}><option value="">{locale === 'pl' ? 'Głos główny' : 'Main voice'}</option>{speakers.map(item => <option key={item.label} value={item.label}>{item.display_name ?? item.label}</option>)}</select></label>
                    <button className="icon-button" title={locale === 'pl' ? 'Podziel przy kursorze' : 'Split at cursor'} onClick={() => splitAtCursor(i)}><Scissors size={15} /></button>
                    {i < editableSegments.length - 1 && <button className="icon-button" title={locale === 'pl' ? 'Połącz z następnym fragmentem' : 'Merge with next segment'} onClick={() => mergeWithNext(i)}><Link2 size={15} /></button>}
                    {result && <button className="icon-button" disabled={running} title={locale === 'pl' ? 'Wygeneruj ponownie tylko ten fragment' : 'Regenerate only this segment'} onClick={() => regenerateSegment(i)}><RefreshCw size={15} /></button>}
                  </div>
                </div>
                <span className={`density-indicator ${charsPerSec > 19 ? 'risk' : ''}`} title={locale === 'pl' ? 'Gęstość tekstu względem czasu' : 'Text density against available time'}>{charsPerSec.toFixed(1)} {locale === 'pl' ? 'zn./s' : 'char/s'}</span>
              </article>;
            })}
          </div>
        </div>
      </section>

      <aside className="dub-inspector">
        <section className="inspector-section">
          <div className="inspector-title"><Mic2 size={17} /><h3>{locale === 'pl' ? 'Głos' : 'Voice'}</h3></div>
          <label><span className="field-label">{t('speaker')}</span><select value={speaker} onChange={e => setSpeaker(e.target.value)}>
            {speakers.map(item => <option key={`${item.id}-${item.label}`} value={item.label}>{item.display_name ?? item.label}</option>)}
          </select></label>
          <label><span className="field-label">{locale === 'pl' ? 'Język dubbingu' : 'Dubbing language'}</span>
            <select value={speechLang} onChange={e => { setSpeechLang(e.target.value); setResult(null); }} disabled={running}>
              <option value="pl">Polski</option>
              <option value="en">English</option>
            </select>
          </label>
          <label className="upload-button"><Upload size={15} /><span>{t('addVoice')}</span><input type="file" className="sr-only" accept="audio/*,video/*" onChange={e => void addVoicePrompt(e.target.files?.[0] ?? null)} /></label>
          {promptStatus && <p className="mini-success"><CheckCircle2 size={14} />{promptStatus}</p>}
        </section>

        <section className="inspector-section">
          <div className="inspector-title"><Gauge size={17} /><h3>{t('tempo')}</h3></div>
          <Slider label={locale === 'pl' ? 'Bazowe' : 'Base'} value={baseSpeed} display={`${baseSpeed.toFixed(2)}×`} min={0.75} max={1.35} step={0.05} onChange={setBaseSpeed} />
          <Slider label={locale === 'pl' ? 'Maksymalne dopasowanie' : 'Maximum fitting'} value={maxSpeed} display={`${maxSpeed.toFixed(2)}×`} min={1} max={1.3} step={0.05} onChange={setMaxSpeed} />
        </section>

        <section className="inspector-section">
          <div className="inspector-title"><SlidersHorizontal size={17} /><h3>{t('mix')}</h3></div>
          <Slider label={locale === 'pl' ? 'Tło oryginału' : 'Original background'} value={originalGain} display={`${Math.round(originalGain * 100)}%`} min={0} max={1} step={0.01} onChange={setOriginalGain} />
          <Slider label="Dubbing" value={dubbingGain} display={`${Math.round(dubbingGain * 100)}%`} min={0.5} max={1.3} step={0.01} onChange={setDubbingGain} />
          <Slider label="Ducking" value={ducking} display={`${Math.round(ducking * 100)}%`} min={0} max={1} step={0.01} onChange={setDucking} />
          <div className="mix-meter"><Music2 size={14} /><span>{t('originalBackground')}</span><Volume2 size={14} /></div>
        </section>

        {(running || error) && <JobProgress message={message} progress={progress} error={error || undefined} />}
        {quotaExceeded && <QuotaExceededNotice />}
        <button className="button button-primary render-button" onClick={() => void run()} disabled={running || !speaker || !ttsModel}>
          <WandSparkles size={17} />{running ? (locale === 'pl' ? 'Renderuję…' : 'Rendering…') : result ? (locale === 'pl' ? 'Renderuj ponownie' : 'Render again') : (locale === 'pl' ? 'Renderuj dubbing' : 'Render dubbing')}
        </button>
      </aside>
    </div>
  );
}

function SyncedVideo({ videoSrc, audioSrc, original, videoRef, audioRef }: {
  videoSrc: string; audioSrc: string; original: boolean;
  videoRef: MutableRefObject<HTMLVideoElement | null>; audioRef: MutableRefObject<HTMLAudioElement | null>;
}) {
  return <div className="synced-player">
    <video ref={videoRef} controls src={videoSrc} muted={!original}
      onPlay={e => { if (!original && audioRef.current) { audioRef.current.currentTime = e.currentTarget.currentTime; void audioRef.current.play(); } }}
      onPause={() => audioRef.current?.pause()}
      onSeeked={e => { if (audioRef.current) audioRef.current.currentTime = e.currentTarget.currentTime; }}
      onTimeUpdate={e => { if (!original && audioRef.current && Math.abs(audioRef.current.currentTime - e.currentTarget.currentTime) > 0.15) audioRef.current.currentTime = e.currentTarget.currentTime; }} />
    {!original && <audio ref={audioRef} src={audioSrc} />}
  </div>;
}
