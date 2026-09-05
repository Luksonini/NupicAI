'use client';

import { useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import {
  AlertTriangle, Check, CheckCircle2, Download, Gauge, Loader2, Mic2, Music2, Play,
  Link2, Redo2, RefreshCw, RotateCcw, Scissors, SlidersHorizontal, Undo2, Upload, UserRound,
  Volume2, WandSparkles, X,
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
  const usedIds = new Set<string>();
  return segments.map((segment, index) => {
    const requestedId = String(segment.segment_id ?? '').trim();
    const segmentId = requestedId && !usedIds.has(requestedId) ? requestedId : freshSegmentId();
    usedIds.add(segmentId);
    return {
      ...segment,
      index,
      translation: segment.translation ?? segment.text,
      segment_id: segmentId,
      seed: segment.seed ?? 1234 + index,
      render_nonce: segment.render_nonce ?? 0,
    };
  });
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

function splitPositions(text: string): number[] {
  const positions: number[] = [];
  const whitespace = /\s+/g;
  let match: RegExpExecArray | null;
  while ((match = whitespace.exec(text)) !== null) {
    const position = match.index;
    if (position > 0 && position < text.length - match[0].length) positions.push(position);
  }
  return positions;
}

function nearestSplitPosition(text: string, wanted: number): number | null {
  const positions = splitPositions(text);
  if (!positions.length) return null;
  return positions.reduce((best, position) => (
    Math.abs(position - wanted) < Math.abs(best - wanted) ? position : best
  ));
}

function splitPositionLabel(text: string, position: number): string {
  const left = text.slice(0, position).trim().split(/\s+/).slice(-3).join(' ');
  const right = text.slice(position).trim().split(/\s+/).slice(0, 3).join(' ');
  return `${left}  |  ${right}`;
}

function SplitBoundaryPicker({ text, position, onChange, locale }: {
  text: string; position: number; onChange: (position: number) => void; locale: string;
}) {
  const words = Array.from(text.matchAll(/\S+/g));
  return <div className="split-boundary-picker" role="group" aria-label={locale === 'pl' ? 'Miejsce podziału tekstu' : 'Text split position'}>
    {words.map((match, index) => {
      const word = match[0];
      const boundary = Number(match.index) + word.length;
      const hasBoundary = index < words.length - 1;
      return <span className="split-picker-word" key={`${boundary}-${word}`}>
        <span>{word}</span>
        {hasBoundary && <button
          type="button"
          className={boundary === position ? 'active' : ''}
          onClick={() => onChange(boundary)}
          title={`${locale === 'pl' ? 'Podziel tutaj' : 'Split here'}: ${splitPositionLabel(text, boundary)}`}
          aria-label={`${locale === 'pl' ? 'Podziel po' : 'Split after'} ${word}`}
          aria-pressed={boundary === position}
        />}
      </span>;
    })}
  </div>;
}

export default function TTSPanel({ segments, targetLang, transcribeJobId, originalSrc, hasVideo }: Props) {
  const { locale, t } = useLocale();
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speakersLoading, setSpeakersLoading] = useState(true);
  const [speakerLoadError, setSpeakerLoadError] = useState('');
  const [speaker, setSpeaker] = useState('');
  const [ttsModel, setTtsModel] = useState('');
  const [speechLang, setSpeechLang] = useState(targetLang === 'en' ? 'en' : 'pl');
  const [flowSettings, setFlowSettings] = useState({ mel_steps_first: 8, mel_steps_second: 3, mel_twopass_t_noise: 0.12 });
  const [baseSpeed, setBaseSpeed] = useState(1.0);
  const [maxSpeed, setMaxSpeed] = useState(1.3);
  const [originalGain, setOriginalGain] = useState(0.22);
  const [dubbingGain, setDubbingGain] = useState(1.0);
  const [ducking, setDucking] = useState(0.65);
  // Stable unique keys must exist on the first render. Assigning them later in
  // an effect leaves React reconciling a list whose every key was "undefined".
  const [editableSegments, setEditableSegments] = useState<Segment[]>(() => prepareSegments(segments));
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
  const cursorPositionRefs = useRef<Record<string, number>>({});
  const textEditStartRef = useRef<Record<string, Segment[]>>({});
  const undoStackRef = useRef<Segment[][]>([]);
  const redoStackRef = useRef<Segment[][]>([]);
  const [historySize, setHistorySize] = useState({ undo: 0, redo: 0 });
  const [splitDraft, setSplitDraft] = useState<{ segmentId: string; position: number } | null>(null);
  const [editorNotice, setEditorNotice] = useState('');

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

  const commitSegments = (next: Segment[], notice = '') => {
    rememberSnapshot(editableSegments);
    setEditableSegments(next.map((segment, index) => ({ ...segment, index })));
    setResult(null);
    setError('');
    setSplitDraft(null);
    setEditorNotice(notice);
  };

  const undoEdit = () => {
    const previous = undoStackRef.current.pop();
    if (!previous) return;
    redoStackRef.current.push(cloneSegments(editableSegments));
    setEditableSegments(cloneSegments(previous));
    setResult(null); setError(''); setSplitDraft(null); setEditorNotice(''); refreshHistorySize();
  };

  const redoEdit = () => {
    const next = redoStackRef.current.pop();
    if (!next) return;
    undoStackRef.current.push(cloneSegments(editableSegments));
    setEditableSegments(cloneSegments(next));
    setResult(null); setError(''); setSplitDraft(null); setEditorNotice(''); refreshHistorySize();
  };

  useEffect(() => {
    setEditableSegments(prepareSegments(segments));
    undoStackRef.current = []; redoStackRef.current = []; textEditStartRef.current = {};
    setHistorySize({ undo: 0, redo: 0 });
    setSplitDraft(null); setEditorNotice('');
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

  const loadSpeakerOptions = async () => {
    setSpeakersLoading(true); setSpeakerLoadError('');
    try {
      const list = await listSpeakers();
      if (!list.length) throw new Error(locale === 'pl' ? 'Serwer nie zwrócił listy głosów.' : 'The server returned no voices.');
      setSpeakers(list);
      setSpeaker(current => list.some(item => item.label === current) ? current : list[0].label);
    } catch (loadError: unknown) {
      setSpeakerLoadError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      setSpeakersLoading(false);
    }
  };

  useEffect(() => {
    void loadSpeakerOptions();
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
    commitSegments(
      [...editableSegments.slice(0, index), merged, ...editableSegments.slice(index + 2)],
      locale === 'pl' ? `Połączono segment ${index + 1} z następnym.` : `Merged segment ${index + 1} with the next segment.`,
    );
  };

  const openSplitEditor = (index: number) => {
    const segment = editableSegments[index];
    const segmentId = String(segment.segment_id);
    const text = String(segment.translation ?? segment.text ?? '');
    const cursor = cursorPositionRefs.current[segmentId]
      ?? textareaRefs.current[segmentId]?.selectionStart
      ?? Math.round(text.length / 2);
    const position = nearestSplitPosition(text, cursor);
    if (position === null) {
      setEditorNotice(locale === 'pl' ? 'Ten segment nie ma granicy między słowami.' : 'This segment has no boundary between words.');
      return;
    }
    setSplitDraft({ segmentId, position });
    setEditorNotice('');
  };

  const confirmSplit = (index: number) => {
    const segment = editableSegments[index];
    const segmentId = String(segment.segment_id);
    if (!splitDraft || splitDraft.segmentId !== segmentId) return;
    const text = String(segment.translation ?? segment.text ?? '');
    const left = text.slice(0, splitDraft.position).trim();
    const right = text.slice(splitDraft.position).trim();
    if (!left || !right) return;
    const ratio = left.length / Math.max(1, left.length + right.length);
    const middle = Number(segment.start) + (Number(segment.end) - Number(segment.start)) * ratio;
    const [sourceLeft, sourceRight] = splitTextNearRatio(
      String(segment.source_text ?? segment.text ?? ''), ratio,
    );
    const common = { ...segment, words: undefined, render_nonce: 0 };
    const first: Segment = { ...common, end: middle, text: sourceLeft || left, source_text: sourceLeft, translation: left, segment_id: freshSegmentId(), seed: 1234 + index };
    const second: Segment = { ...common, start: middle, text: sourceRight || right, source_text: sourceRight, translation: right, segment_id: freshSegmentId(), seed: 2234 + index };
    commitSegments(
      [...editableSegments.slice(0, index), first, second, ...editableSegments.slice(index + 1)],
      locale === 'pl' ? `Podzielono segment ${index + 1}.` : `Split segment ${index + 1}.`,
    );
  };

  const restoreSegments = () => {
    commitSegments(
      prepareSegments(segments),
      locale === 'pl' ? 'Przywrócono pierwotny podział.' : 'Original segmentation restored.',
    );
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
          {editorNotice && <div className="segment-editor-notice" role="status"><CheckCircle2 size={15} />{editorNotice}</div>}
          <div className="segment-table">
            {editableSegments.map((seg, i) => {
              const budget = Math.max(0.1, Number(seg.end) - Number(seg.start));
              const charsPerSec = String(seg.translation ?? seg.text).length / budget;
              const segmentId = String(seg.segment_id);
              const activeSplit = splitDraft?.segmentId === segmentId;
              return <article className="edit-segment-row" key={segmentId}>
                <button className="timecode" onClick={() => seekTo(Number(seg.start))}>{fmt(Number(seg.start))}</button>
                <div className="segment-copy"><p className="source-line"><span>{locale === 'pl' ? 'Oryginał' : 'Original'}</span>{seg.source_text ?? seg.text}</p>
                  <label className="segment-text-label" htmlFor={`segment-text-${segmentId}`}>{locale === 'pl' ? 'Tekst dubbingu' : 'Dubbing text'}</label>
                  <textarea ref={node => { textareaRefs.current[segmentId] = node; }} value={seg.translation ?? seg.text}
                    id={`segment-text-${segmentId}`}
                    onFocus={() => beginTextEdit(segmentId)} onBlur={() => finishTextEdit(segmentId)}
                    onSelect={event => { cursorPositionRefs.current[segmentId] = event.currentTarget.selectionStart; }}
                    onChange={e => { setEditableSegments(prev => prev.map((item, idx) => idx === i ? { ...item, translation: e.target.value } : item)); setSplitDraft(null); setEditorNotice(''); setResult(null); }} />
                  <div className="segment-tools">
                    <label title={locale === 'pl' ? 'Głos tylko dla tego fragmentu' : 'Voice for this segment only'}><UserRound size={14} /><select value={seg.speaker_label ?? ''} disabled={speakersLoading || !!speakerLoadError} onChange={e => commitSegments(editableSegments.map((item, idx) => idx === i ? { ...item, speaker_label: e.target.value || undefined } : item))}><option value="">{speakersLoading ? (locale === 'pl' ? 'Ładowanie głosów…' : 'Loading voices…') : `${locale === 'pl' ? 'Głos główny' : 'Main voice'}${speaker ? `: ${speakers.find(item => item.label === speaker)?.display_name ?? speaker}` : ''}`}</option>{speakers.map(item => <option key={item.label} value={item.label}>{item.display_name ?? item.label}</option>)}</select></label>
                    <button className={`button button-ghost button-small segment-action${activeSplit ? ' active' : ''}`} onClick={() => openSplitEditor(i)}><Scissors size={14} />{locale === 'pl' ? 'Podziel' : 'Split'}</button>
                    {i < editableSegments.length - 1 && <button className="button button-ghost button-small segment-action" onClick={() => mergeWithNext(i)}><Link2 size={14} />{locale === 'pl' ? 'Połącz z następnym' : 'Merge next'}</button>}
                    {result && <button className="icon-button" disabled={running} title={locale === 'pl' ? 'Wygeneruj ponownie tylko ten fragment' : 'Regenerate only this segment'} onClick={() => regenerateSegment(i)}><RefreshCw size={15} /></button>}
                  </div>
                  {activeSplit && <div className="split-editor">
                    <div className="split-boundary-field"><span className="field-label">{locale === 'pl' ? 'Miejsce podziału' : 'Split position'}</span><SplitBoundaryPicker text={String(seg.translation ?? seg.text ?? '')} position={splitDraft.position} onChange={position => setSplitDraft({ segmentId, position })} locale={locale} /></div>
                    <button className="button button-primary button-small" onClick={() => confirmSplit(i)}><Check size={14} />{locale === 'pl' ? 'Zatwierdź' : 'Apply'}</button>
                    <button className="icon-button" onClick={() => setSplitDraft(null)} title={locale === 'pl' ? 'Anuluj podział' : 'Cancel split'}><X size={15} /></button>
                  </div>}
                </div>
                <span className={`density-indicator ${charsPerSec > 19 ? 'risk' : ''}`} title={locale === 'pl' ? 'Gęstość tekstu względem czasu' : 'Text density against available time'}>{charsPerSec.toFixed(1)} {locale === 'pl' ? 'zn./s' : 'char/s'}</span>
              </article>;
            })}
          </div>
        </div>
      </section>

      <aside className="dub-inspector">
        <div className="inspector-scroll">
          <section className="inspector-section">
            <div className="inspector-title"><Mic2 size={17} /><h3>{locale === 'pl' ? 'Głos' : 'Voice'}</h3></div>
            <label><span className="field-label">{t('speaker')}</span><select value={speaker} onChange={e => setSpeaker(e.target.value)} disabled={speakersLoading || !!speakerLoadError}>
              {!speakers.length && <option value="">{speakersLoading ? (locale === 'pl' ? 'Ładowanie głosów…' : 'Loading voices…') : (locale === 'pl' ? 'Brak głosów' : 'No voices')}</option>}
              {speakers.map(item => <option key={`${item.id}-${item.label}`} value={item.label}>{item.display_name ?? item.label}</option>)}
            </select></label>
            {speakersLoading && <p className="speaker-loading"><Loader2 className="spin" size={14} />{locale === 'pl' ? 'Ładowanie listy głosów…' : 'Loading voice list…'}</p>}
            {speakerLoadError && <div className="speaker-load-error"><AlertTriangle size={15} /><span>{speakerLoadError}</span><button className="button button-secondary button-small" onClick={() => void loadSpeakerOptions()}><RefreshCw size={14} />{locale === 'pl' ? 'Ponów' : 'Retry'}</button></div>}
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
        </div>

        <div className="inspector-actions">
          {(running || error) && <div className="inspector-progress"><JobProgress message={message} progress={progress} error={error || undefined} /></div>}
          {quotaExceeded && <QuotaExceededNotice />}
          <button className="button button-primary render-button" onClick={() => void run()} disabled={running || !speaker || !ttsModel}>
            <WandSparkles size={17} />{running ? (locale === 'pl' ? 'Renderuję…' : 'Rendering…') : result ? (locale === 'pl' ? 'Renderuj ponownie' : 'Render again') : (locale === 'pl' ? 'Renderuj dubbing' : 'Render dubbing')}
          </button>
        </div>
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
