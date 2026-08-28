'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AudioLines, ChevronRight, FileAudio, Languages, Menu, Mic2,
  PanelLeftClose, Plus, Settings, ShieldCheck, Upload, Youtube,
} from 'lucide-react';
import type { TranscribeResult, TranslateResult, Step } from '@/lib/types';
import { uploadAndTranscribe, transcribeYoutube, sourceUrl, streamJob, checkHealth } from '@/lib/api';
import DropZone from '@/components/DropZone';
import JobProgress from '@/components/JobProgress';
import TranscriptPanel from '@/components/TranscriptPanel';
import TranslatePanel from '@/components/TranslatePanel';
import TTSPanel from '@/components/TTSPanel';
import TextTTSPanel from '@/components/TextTTSPanel';
import AdminPanel from '@/components/AdminPanel';

type Service = 'transcribe' | 'translate' | 'dub' | 'voice' | 'admin';

const SERVICES = [
  { id: 'transcribe' as const, label: 'Transkrypcja', icon: FileAudio },
  { id: 'translate' as const, label: 'Tłumaczenie', icon: Languages },
  { id: 'dub' as const, label: 'Dubbing', icon: AudioLines },
  { id: 'voice' as const, label: 'Studio głosu', icon: Mic2 },
  { id: 'admin' as const, label: 'Administrator', icon: ShieldCheck },
];

const SERVICE_META: Record<Service, { title: string; caption: string }> = {
  transcribe: { title: 'Transkrypcja', caption: 'Materiał źródłowy i napisy' },
  translate: { title: 'Tłumaczenie', caption: 'Tekst źródłowy i wersja docelowa' },
  dub: { title: 'Dubbing', caption: 'Głosy, segmenty i miks końcowy' },
  voice: { title: 'Studio głosu', caption: 'Synteza mowy z tekstu' },
  admin: { title: 'Administrator', caption: 'Modele, integracje i diagnostyka' },
};

export default function Home() {
  const [service, setService] = useState<Service>('transcribe');
  const [railOpen, setRailOpen] = useState(true);
  const [health, setHealth] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [sourceKind, setSourceKind] = useState<'upload' | 'youtube'>('upload');
  const [audioSrc, setAudioSrc] = useState('');
  const [step, setStep] = useState<Step>('idle');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [transcribeResult, setTranscribeResult] = useState<TranscribeResult | null>(null);
  const [translateResult, setTranslateResult] = useState<TranslateResult | null>(null);
  const [transcribeJobId, setTranscribeJobId] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const refresh = () => checkHealth().then(setHealth);
    refresh(); const id = window.setInterval(refresh, 30000); return () => window.clearInterval(id);
  }, []);

  const resetProject = useCallback(() => {
    abortRef.current?.abort();
    if (audioSrc.startsWith('blob:')) URL.revokeObjectURL(audioSrc);
    setFile(null); setYoutubeUrl(''); setAudioSrc(''); setStep('idle');
    setTranscribeResult(null); setTranslateResult(null); setTranscribeJobId(''); setError('');
    setService('transcribe');
  }, [audioSrc]);

  const handleFile = useCallback((nextFile: File) => {
    if (audioSrc.startsWith('blob:')) URL.revokeObjectURL(audioSrc);
    setFile(nextFile); setSourceKind('upload'); setAudioSrc(URL.createObjectURL(nextFile));
    setStep('idle'); setTranscribeResult(null); setTranslateResult(null); setTranscribeJobId(''); setError('');
  }, [audioSrc]);

  const runTranscriptionJob = async (createJob: () => Promise<string>, nextAudioSrc?: (jobId: string) => string) => {
    abortRef.current?.abort(); const ctrl = new AbortController(); abortRef.current = ctrl;
    setStep('transcribing'); setProgress(0); setMessage('Analizuję materiał…'); setError('');
    try {
      const jobId = await createJob(); setTranscribeJobId(jobId);
      await new Promise<void>((resolve, reject) => {
        ctrl.signal.addEventListener('abort', () => reject(new Error('Przerwano')));
        streamJob(jobId, ev => {
          if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
          else if (ev.type === 'done') { setTranscribeResult(ev.result as TranscribeResult); resolve(); }
          else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd transkrypcji'));
        }, ctrl.signal);
      });
      if (nextAudioSrc) setAudioSrc(nextAudioSrc(jobId));
      setStep('transcribed');
    } catch (e: unknown) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof Error ? e.message : String(e)); setStep('error');
    }
  };

  const transcribe = () => file ? runTranscriptionJob(() => uploadAndTranscribe(file)) : Promise.resolve();
  const transcribeFromYoutube = async () => {
    const url = youtubeUrl.trim(); if (!url) return;
    setFile(null); setSourceKind('youtube'); setAudioSrc(''); setTranscribeResult(null); setTranslateResult(null);
    await runTranscriptionJob(() => transcribeYoutube(url), sourceUrl);
  };

  const isVideo = sourceKind === 'youtube' || !!file && (file.type.startsWith('video/') || /\.(mp4|mkv|webm|avi|mov)$/i.test(file.name));
  const projectName = useMemo(() => file?.name || (youtubeUrl ? 'Materiał YouTube' : 'Nowy projekt'), [file, youtubeUrl]);
  const meta = SERVICE_META[service];

  const intake = <SourceIntake
    file={file} youtubeUrl={youtubeUrl} setYoutubeUrl={setYoutubeUrl} onFile={handleFile}
    transcribe={() => void transcribe()} transcribeYoutube={() => void transcribeFromYoutube()}
    busy={step === 'transcribing'} progress={progress} message={message} error={error}
  />;

  return <div className={`app-shell ${railOpen ? '' : 'rail-collapsed'}`}>
    <aside className="app-rail">
      <BrandMark />
      <button className="button button-primary new-project" onClick={resetProject}><Plus size={16} /><span>Nowy projekt</span></button>
      <nav className="service-nav" aria-label="Usługi">
        <span className="nav-label">Usługi</span>
        {SERVICES.map(item => <button key={item.id} className={service === item.id ? 'active' : ''} onClick={() => setService(item.id)} title={item.label}>
          <item.icon size={18} /><span>{item.label}</span>
          {item.id === 'transcribe' && transcribeResult && <i />}
          {item.id === 'translate' && translateResult && <i />}
        </button>)}
      </nav>
      <div className="rail-project">
        <span className="nav-label">Bieżący projekt</span>
        <div className="project-chip"><FileAudio size={16} /><div><strong>{projectName}</strong><span>{transcribeResult ? `${transcribeResult.segment_count} segmentów` : 'Bez transkrypcji'}</span></div></div>
      </div>
    </aside>

    <div className="app-body">
      <header className="topbar">
        <button className="icon-button rail-toggle" title="Nawigacja" onClick={() => setRailOpen(v => !v)}>{railOpen ? <PanelLeftClose size={18} /> : <Menu size={18} />}</button>
        <div className="page-title"><h1>{meta.title}</h1><p>{meta.caption}</p></div>
        <div className={`server-status ${health === null ? 'checking' : health ? 'online' : 'offline'}`}><Activity size={14} /><span>{health === null ? 'Sprawdzam' : health ? 'System gotowy' : 'System offline'}</span></div>
      </header>

      <main className="workspace">
        {service !== 'voice' && service !== 'admin' && <PipelineBar service={service} hasTranscript={!!transcribeResult} hasTranslation={!!translateResult} onSelect={setService} />}

        {service === 'transcribe' && (!transcribeResult ? intake : <>
          <div className="workspace-toolbar"><div><h2>{projectName}</h2><p>{transcribeResult.detected_language.toUpperCase()} · {transcribeResult.word_count} słów · {transcribeResult.segment_count} segmentów</p></div>
            <button className="button button-primary" onClick={() => setService('translate')}>Tłumaczenie <ChevronRight size={16} /></button></div>
          <section className="panel transcript-surface"><TranscriptPanel result={transcribeResult} audioSrc={audioSrc} /></section>
        </>)}

        {service === 'translate' && (!transcribeResult ? intake : <div className="translation-layout">
          <section className="panel source-reference"><TranscriptPanel result={transcribeResult} audioSrc={audioSrc} /></section>
          <section className="panel translation-surface"><TranslatePanel segments={transcribeResult.segments} sourceLang={transcribeResult.detected_language}
            onDone={result => { setTranslateResult(result); setStep('translated'); }} onContinue={() => setService('dub')} /></section>
        </div>)}

        {service === 'dub' && (!translateResult ? <PrerequisiteState hasTranscript={!!transcribeResult} onAction={() => setService(transcribeResult ? 'translate' : 'transcribe')} /> :
          <TTSPanel segments={translateResult.segments} targetLang={translateResult.target_lang} transcribeJobId={transcribeJobId} originalSrc={audioSrc} hasVideo={isVideo} />)}

        {service === 'voice' && <TextTTSPanel />}
        {service === 'admin' && <AdminPanel />}
      </main>
    </div>
  </div>;
}

function BrandMark() {
  const [logoAvailable, setLogoAvailable] = useState(true);
  return <div className={`brand-mark ${logoAvailable ? 'has-logo' : 'fallback'}`} aria-label="NupicAI">
    <img src="/brand/logo.png" alt="NupicAI" onError={() => setLogoAvailable(false)} />
    <AudioLines className="brand-icon" size={21} />
    <span>Nupic<strong>AI</strong></span>
  </div>;
}

function PipelineBar({ service, hasTranscript, hasTranslation, onSelect }: { service: Service; hasTranscript: boolean; hasTranslation: boolean; onSelect: (service: Service) => void }) {
  const steps = [
    { id: 'transcribe' as const, label: 'Transkrypcja', done: hasTranscript },
    { id: 'translate' as const, label: 'Tłumaczenie', done: hasTranslation },
    { id: 'dub' as const, label: 'Dubbing', done: false },
  ];
  return <div className="pipeline-bar">{steps.map((item, index) => <div className="pipeline-node" key={item.id}>
    <button className={`${service === item.id ? 'active' : ''} ${item.done ? 'done' : ''}`} onClick={() => onSelect(item.id)}><span>{item.done ? '✓' : index + 1}</span>{item.label}</button>
    {index < steps.length - 1 && <ChevronRight size={15} />}
  </div>)}</div>;
}

function SourceIntake({ file, youtubeUrl, setYoutubeUrl, onFile, transcribe, transcribeYoutube, busy, progress, message, error }: {
  file: File | null; youtubeUrl: string; setYoutubeUrl: (value: string) => void; onFile: (file: File) => void;
  transcribe: () => void; transcribeYoutube: () => void; busy: boolean; progress: number; message: string; error: string;
}) {
  const [sourceTab, setSourceTab] = useState<'file' | 'youtube'>('file');
  return <div className="intake-layout">
    <section className="panel intake-panel">
      <div className="panel-heading"><span className="icon-box"><Upload size={18} /></span><div><h2>Materiał źródłowy</h2><p>Audio lub wideo</p></div></div>
      <div className="segmented-control source-tabs"><button className={sourceTab === 'file' ? 'active' : ''} onClick={() => setSourceTab('file')}><FileAudio size={15} />Plik</button><button className={sourceTab === 'youtube' ? 'active' : ''} onClick={() => setSourceTab('youtube')}><Youtube size={15} />YouTube</button></div>
      {sourceTab === 'file' ? <><DropZone onFile={onFile} disabled={busy} />{file && <button className="button button-primary intake-action" onClick={transcribe} disabled={busy}><FileAudio size={16} />Transkrybuj materiał</button>}</> :
        <div className="youtube-source"><label><span className="field-label">Adres filmu</span><div className="field-with-icon"><Youtube size={16} /><input value={youtubeUrl} onChange={e => setYoutubeUrl(e.target.value)} placeholder="https://youtube.com/watch?v=…" /></div></label><button className="button button-primary" onClick={transcribeYoutube} disabled={busy || !youtubeUrl.trim()}>Pobierz i transkrybuj</button></div>}
      {(busy || error) && <JobProgress message={message} progress={progress} error={error || undefined} />}
    </section>
    <aside className="intake-summary"><div><FileAudio size={20} /><strong>Transkrypcja</strong><span>TXT, SRT, VTT</span></div><div><Languages size={20} /><strong>Tłumaczenie</strong><span>PL i EN</span></div><div><AudioLines size={20} /><strong>Dubbing</strong><span>WAV i MP4</span></div></aside>
  </div>;
}

function PrerequisiteState({ hasTranscript, onAction }: { hasTranscript: boolean; onAction: () => void }) {
  return <section className="panel prerequisite"><AudioLines size={28} /><div><h2>{hasTranscript ? 'Brakuje tłumaczenia' : 'Brakuje materiału źródłowego'}</h2><p>{hasTranscript ? 'Transkrypcja jest gotowa.' : 'Rozpocznij od transkrypcji.'}</p></div><button className="button button-primary" onClick={onAction}>{hasTranscript ? 'Przejdź do tłumaczenia' : 'Dodaj materiał'}<ChevronRight size={16} /></button></section>;
}
