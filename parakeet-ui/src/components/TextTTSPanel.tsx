'use client';

import { useEffect, useState } from 'react';
import { Download, Mic2, Play, WandSparkles } from 'lucide-react';
import type { Speaker, TTSResult } from '@/lib/types';
import { listSpeakers, listTTSModels, streamJob } from '@/lib/api';
import JobProgress from './JobProgress';
import { useLocale } from '@/lib/locale';

async function submitTextTTS(params: Record<string, unknown>): Promise<string> {
  const res = await fetch('/tts_text', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return (await res.json()).job_id as string;
}

export default function TextTTSPanel() {
  const { t } = useLocale();
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [speaker, setSpeaker] = useState('');
  const [ttsModel, setTtsModel] = useState('');
  const [text, setText] = useState('');
  const [lang, setLang] = useState('pl');
  const [speed, setSpeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [jobId, setJobId] = useState('');
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    listSpeakers().then(list => { setSpeakers(list); if (list.length) setSpeaker(list[0].label); });
    listTTSModels().then(data => setTtsModel(data.active || data.default || data.models[0]?.key || ''));
  }, []);

  const run = async () => {
    setRunning(true); setError(''); setAudioUrl(''); setProgress(0); setMessage('Przygotowuję syntezę…');
    try {
      const id = await submitTextTTS({
        text, speaker_label: speaker, tts_model_profile: ttsModel, language: lang,
        speed, dur_scale: 1, mel_steps_first: 8, mel_steps_second: 3,
        mel_twopass_t_noise: 0.12, digital_silence: true, pause_edge_frames: 10,
        short_continuity_ms: 0, emotion_group: 'neutral', emotion_strength: 0,
      });
      setJobId(id);
      await new Promise<void>((resolve, reject) => streamJob(id, ev => {
        if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
        else if (ev.type === 'done') {
          const result = ev.result as TTSResult;
          setDuration(result.duration); setAudioUrl(`/jobs/${id}/audio`); resolve();
        } else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd TTS'));
      }));
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setRunning(false); }
  };

  return <div className="voice-studio-grid">
    <section className="panel voice-editor">
      <div className="panel-heading"><span className="icon-box"><Mic2 size={18} /></span><div><h2>{t('voiceStudio')}</h2><p>{t('singleSpeech')}</p></div></div>
      <textarea className="voice-textarea" rows={9} value={text} onChange={e => setText(e.target.value)} placeholder={t('synthesisText')} />
      <div className="character-count">{text.length} znaków</div>
    </section>
    <aside className="panel voice-controls">
      <label><span className="field-label">{t('speaker')}</span><select value={speaker} onChange={e => setSpeaker(e.target.value)}>{speakers.map(item => <option key={`${item.id}-${item.label}`} value={item.label}>{item.label}</option>)}</select></label>
      <label><span className="field-label">{t('language')}</span><select value={lang} onChange={e => setLang(e.target.value)}><option value="pl">Polski</option><option value="en">English</option></select></label>
      <label className="slider-field"><span><b>{t('tempo')}</b><output>{speed.toFixed(2)}×</output></span><input type="range" min={0.7} max={1.5} step={0.05} value={speed} onChange={e => setSpeed(Number(e.target.value))} /></label>
      {(running || error) && <JobProgress message={message} progress={progress} error={error || undefined} />}
      <button className="button button-primary render-button" disabled={running || !text.trim() || !speaker || !ttsModel} onClick={() => void run()}><WandSparkles size={17} />{running ? t('synthesizing') : t('generateSpeech')}</button>
      {audioUrl && <div className="voice-result"><div><Play size={16} /><span>OK · {duration.toFixed(1)} s</span></div><audio controls autoPlay src={audioUrl} /><a className="button button-secondary" href={audioUrl} download={`tts_${jobId}.wav`}><Download size={15} />{t('downloadWav')}</a></div>}
    </aside>
  </div>;
}
