'use client';

import { useEffect, useRef, useState } from 'react';
import type { MutableRefObject } from 'react';
import {
  AlertTriangle, CheckCircle2, Download, Gauge, Mic2, Music2, Play,
  SlidersHorizontal, Upload, Volume2, WandSparkles,
} from 'lucide-react';
import type { Segment, Speaker, DubResult } from '@/lib/types';
import {
  listSpeakers, listTTSModels, uploadVoicePrompt, submitDub, streamJob,
  dubAudioUrl, mixAudioUrl, mixVideoUrl,
} from '@/lib/api';
import JobProgress from './JobProgress';
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

export default function TTSPanel({ segments, targetLang, transcribeJobId, originalSrc, hasVideo }: Props) {
  const { locale, t } = useLocale();
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState('');
  const [ttsModel, setTtsModel] = useState('');
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
  const [promptStatus, setPromptStatus] = useState('');
  const [dubJobId, setDubJobId] = useState('');
  const [result, setResult] = useState<DubResult | null>(null);
  const [audioMode, setAudioMode] = useState<'original' | 'mix' | 'voice'>('original');
  const mediaRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    setEditableSegments(segments.map(s => ({ ...s, translation: s.translation ?? s.text })));
    setResult(null); setDubJobId('');
  }, [segments]);

  useEffect(() => {
    listSpeakers().then(list => { setSpeakers(list); if (list.length) setSpeaker(list[0].label); });
    listTTSModels().then(data => setTtsModel(data.active || data.default || data.models[0]?.key || ''));
  }, []);

  const run = async () => {
    setRunning(true); setError(''); setProgress(0); setMessage(locale === 'pl' ? 'Przygotowuję rendering…' : 'Preparing render…');
    try {
      const jobId = await submitDub({
        segments: editableSegments, speaker_label: speaker, tts_model_profile: ttsModel,
        transcribe_job_id: transcribeJobId, target_lang: targetLang,
        base_speed: baseSpeed, max_adaptive_speed: maxSpeed, extra_tail_sec: 0,
        dur_scale: 1, mel_steps_first: 8, mel_steps_second: 3, mel_twopass_t_noise: 0.12,
        digital_silence: true, pause_edge_frames: 10, short_continuity_ms: 0,
        emotion_group: 'neutral', emotion_strength: 0,
        original_gain: originalGain, dubbing_gain: dubbingGain, ducking_strength: ducking,
      });
      setDubJobId(jobId);
      await new Promise<void>((resolve, reject) => streamJob(jobId, ev => {
        if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
        else if (ev.type === 'done') { setResult(ev.result as DubResult); setAudioMode('mix'); resolve(); }
        else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd TTS'));
      }));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setRunning(false); }
  };

  const addVoicePrompt = async (file: File | null) => {
    if (!file) return;
    setPromptStatus(locale === 'pl' ? 'Koduję próbkę głosu…' : 'Encoding voice sample…'); setError('');
    try {
      const custom = await uploadVoicePrompt(file);
      setSpeakers(prev => [custom, ...prev.filter(item => item.label !== custom.label)]);
      setSpeaker(custom.label); setPromptStatus(custom.label);
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
          <div><strong>{warningCount ? `${warningCount} ${locale === 'pl' ? 'segmentów wymaga odsłuchu' : 'segments need review'}` : (locale === 'pl' ? 'Rendering gotowy' : 'Render complete')}</strong><span>{result.duration.toFixed(1)} s · {editableSegments.length} {locale === 'pl' ? 'segmentów' : 'segments'}</span></div>
          <div className="download-group">
            <a className="button button-secondary" href={voiceUrl} download="dubbing.wav"><Download size={15} />{locale === 'pl' ? 'Głos' : 'Voice'}</a>
            {result.mixed_audio_path && <a className="button button-secondary" href={mixedUrl} download="mix.wav"><Download size={15} />{t('mix')} WAV</a>}
            {hasVideo && <a className="button button-primary" href={mixVideoUrl(dubJobId, transcribeJobId)} download="dubbing.mp4"><Download size={15} />MP4</a>}
          </div>
        </div>}

        <div className="segment-editor">
          <div className="section-toolbar"><div><h2>{t('dubbingSegments')}</h2><p>{editableSegments.length}</p></div></div>
          <div className="segment-table">
            {editableSegments.map((seg, i) => {
              const budget = Math.max(0.1, Number(seg.end) - Number(seg.start));
              const charsPerSec = String(seg.translation ?? seg.text).length / budget;
              return <article className="edit-segment-row" key={`${seg.index}-${i}`}>
                <button className="timecode" onClick={() => seekTo(Number(seg.start))}>{fmt(Number(seg.start))}</button>
                <div className="segment-copy"><p className="source-line">{seg.source_text ?? seg.text}</p>
                  <textarea value={seg.translation ?? seg.text} onChange={e => setEditableSegments(prev => prev.map((item, idx) => idx === i ? { ...item, translation: e.target.value } : item))} />
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
            {speakers.map(item => <option key={`${item.id}-${item.label}`} value={item.label}>{item.label}</option>)}
          </select></label>
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
