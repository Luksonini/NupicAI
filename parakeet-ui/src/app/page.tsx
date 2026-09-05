'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AudioLines, ChevronRight, Clock3, FileAudio, Languages, Menu, Mic2,
  House, PanelLeftClose, Plus, ShieldCheck, Upload, UserRound, Youtube,
} from 'lucide-react';
import type { TranscribeResult, TranslateResult, Step, Usage, User } from '@/lib/types';
import { accountUsage, uploadAndTranscribe, transcribeYoutube, sourceUrl, streamJob, checkHealth, currentUser } from '@/lib/api';
import DropZone from '@/components/DropZone';
import JobProgress from '@/components/JobProgress';
import TranscriptPanel from '@/components/TranscriptPanel';
import TranslatePanel from '@/components/TranslatePanel';
import TTSPanel from '@/components/TTSPanel';
import TextTTSPanel from '@/components/TextTTSPanel';
import AdminPanel from '@/components/AdminPanel';
import AccountPanel from '@/components/AccountPanel';
import LandingPage from '@/components/LandingPage';
import { LanguageSwitch, LocaleProvider, type Locale, useLocale } from '@/lib/locale';

type Service = 'transcribe' | 'translate' | 'dub' | 'voice' | 'account' | 'admin';

const SERVICES = [
  { id: 'transcribe' as const, label: 'Transkrypcja', icon: FileAudio },
  { id: 'translate' as const, label: 'Tłumaczenie', icon: Languages },
  { id: 'dub' as const, label: 'Dubbing', icon: AudioLines },
  { id: 'voice' as const, label: 'Studio głosu', icon: Mic2 },
  { id: 'account' as const, label: 'Moje konto', icon: UserRound },
  { id: 'admin' as const, label: 'Administrator', icon: ShieldCheck },
];

const SERVICE_META: Record<Service, { title: string; caption: string }> = {
  transcribe: { title: 'Transkrypcja', caption: 'Materiał źródłowy i napisy' },
  translate: { title: 'Tłumaczenie', caption: 'Tekst źródłowy i wersja docelowa' },
  dub: { title: 'Dubbing', caption: 'Głosy, segmenty i miks końcowy' },
  voice: { title: 'Studio głosu', caption: 'Synteza mowy z tekstu' },
  account: { title: 'Moje konto', caption: 'Profil, sesja i prywatność danych' },
  admin: { title: 'Administrator', caption: 'Modele, integracje i diagnostyka' },
};

export default function Home() {
  return <NupicAIApp initialLocale="pl" />;
}

export function NupicAIApp({ initialLocale }: { initialLocale: Locale }) {
  return <LocaleProvider initialLocale={initialLocale}><AuthenticatedApp /></LocaleProvider>;
}

function AuthenticatedApp() {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const [showLanding, setShowLanding] = useState(false);

  useEffect(() => {
    currentUser().then(setUser).catch(() => setUser(null)).finally(() => setReady(true));
  }, []);

  if (!ready) return <div className="app-loading"><img src="/brand/mark.png" alt="" /><span>Wczytuję NupicAI…</span></div>;
  if (!user || showLanding) return <LandingPage
    onAuthenticated={next => { setUser(next); setShowLanding(false); }}
    authenticatedUser={user}
    onOpenStudio={() => setShowLanding(false)}
  />;
  return <Studio user={user} onLogout={() => setUser(null)} onHome={() => setShowLanding(true)} />;
}

function Studio({ user, onLogout, onHome }: { user: User; onLogout: () => void; onHome: () => void }) {
  const { locale, t } = useLocale();
  const [service, setService] = useState<Service>('transcribe');
  const [railOpen, setRailOpen] = useState(true);
  const [health, setHealth] = useState<boolean | null>(null);
  const [usage, setUsage] = useState<Usage>(user.usage);
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

  useEffect(() => {
    const openAccount = () => setService('account');
    window.addEventListener('nupicai-open-account', openAccount);
    return () => window.removeEventListener('nupicai-open-account', openAccount);
  }, []);

  useEffect(() => {
    const refresh = () => accountUsage().then(setUsage).catch(() => undefined);
    window.addEventListener('nupicai-usage-changed', refresh);
    const id = window.setInterval(refresh, 60000);
    return () => {
      window.removeEventListener('nupicai-usage-changed', refresh);
      window.clearInterval(id);
    };
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
    setStep('transcribing'); setProgress(0); setMessage(locale === 'pl' ? 'Analizuję materiał…' : 'Analyzing media…'); setError('');
    try {
      const jobId = await createJob(); setTranscribeJobId(jobId);
      await new Promise<void>((resolve, reject) => {
        ctrl.signal.addEventListener('abort', () => reject(new Error(locale === 'pl' ? 'Przerwano' : 'Cancelled')));
        streamJob(jobId, ev => {
          if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
          else if (ev.type === 'done') { setTranscribeResult(ev.result as TranscribeResult); resolve(); }
          else if (ev.type === 'error') reject(new Error(ev.error ?? (locale === 'pl' ? 'Błąd transkrypcji' : 'Transcription error')));
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
  const projectName = useMemo(() => file?.name || (youtubeUrl ? (locale === 'pl' ? 'Materiał YouTube' : 'YouTube media') : t('newProject')), [file, youtubeUrl, locale, t]);
  const meta = SERVICE_META[service];
  const visibleServices = useMemo(() => SERVICES.filter(item => item.id !== 'admin' || user.is_admin), [user.is_admin]);

  const intake = <SourceIntake
    file={file} youtubeUrl={youtubeUrl} setYoutubeUrl={setYoutubeUrl} onFile={handleFile}
    transcribe={() => void transcribe()} transcribeYoutube={() => void transcribeFromYoutube()}
    busy={step === 'transcribing'} progress={progress} message={message} error={error}
  />;

  return <div className={`app-shell ${railOpen ? '' : 'rail-collapsed'}`}>
    <aside className="app-rail">
      <button className="brand-home" onClick={onHome} title={locale === 'pl' ? 'Strona główna' : 'Home'}><BrandMark /></button>
      <button className="button button-primary new-project" onClick={resetProject}><Plus size={16} /><span>{t('newProject')}</span></button>
      <nav className="service-nav" aria-label={t('services')}>
        <span className="nav-label">{t('services')}</span>
        {visibleServices.map(item => <button key={item.id} className={service === item.id ? 'active' : ''} onClick={() => setService(item.id)} title={t(item.id)}>
          <item.icon size={18} /><span>{t(item.id)}</span>
          {item.id === 'transcribe' && transcribeResult && <i />}
          {item.id === 'translate' && translateResult && <i />}
        </button>)}
      </nav>
      <div className="rail-project">
        <span className="nav-label">{t('currentProject')}</span>
        <div className="project-chip"><FileAudio size={16} /><div><strong>{projectName}</strong><span>{transcribeResult ? `${transcribeResult.segment_count} ${locale === 'pl' ? 'segmentów' : 'segments'}` : t('noTranscript')}</span></div></div>
      </div>
      <button className="rail-account" onClick={() => setService('account')} title={t('account')}><UserRound size={17} /><span><strong>{user.display_name}</strong><small>{user.email}</small></span></button>
    </aside>

    <div className="app-body">
      <header className="topbar">
        <button className="icon-button rail-toggle" title={t('navigation')} onClick={() => setRailOpen(v => !v)}>{railOpen ? <PanelLeftClose size={18} /> : <Menu size={18} />}</button>
        <button className="icon-button home-button" title={locale === 'pl' ? 'Wróć do strony NupicAI' : 'Return to NupicAI home'} onClick={onHome}><House size={18} /></button>
        <div className="page-title"><h1>{t(service)}</h1><p>{locale === 'pl' ? meta.caption : ({ transcribe: 'Source media and subtitles', translate: 'Source text and target version', dub: 'Voices, segments and final mix', voice: 'Speech synthesis from text', account: 'Profile, session and data privacy', admin: 'Models, integrations and diagnostics' } as Record<Service, string>)[service]}</p></div>
        <LanguageSwitch compact />
        <button className="usage-badge" onClick={() => setService('account')} title={locale === 'pl' ? 'Pozostały limit audio' : 'Remaining audio allowance'}>
          <Clock3 size={14} /><span>{usage.unlimited ? (locale === 'pl' ? 'Bez limitu' : 'Unlimited') : `${formatUsageMinutes(usage.available_seconds)} ${locale === 'pl' ? 'pozostało' : 'left'}`}</span>
        </button>
        <div className={`server-status ${health === null ? 'checking' : health ? 'online' : 'offline'}`}><Activity size={14} /><span>{health === null ? t('checking') : health ? t('systemReady') : t('systemOffline')}</span></div>
      </header>

      <main className="workspace">
        {service !== 'voice' && service !== 'account' && service !== 'admin' && <PipelineBar service={service} hasTranscript={!!transcribeResult} hasTranslation={!!translateResult} onSelect={setService} />}

        {service === 'transcribe' && (!transcribeResult ? intake : <>
          <div className="workspace-toolbar"><div><h2>{projectName}</h2><p>{transcribeResult.detected_language.toUpperCase()} · {transcribeResult.word_count} {locale === 'pl' ? 'słów' : 'words'} · {transcribeResult.segment_count} {locale === 'pl' ? 'segmentów' : 'segments'}</p></div>
            <button className="button button-primary" onClick={() => setService('translate')}>{t('translate')} <ChevronRight size={16} /></button></div>
          <section className="panel transcript-surface"><TranscriptPanel result={transcribeResult} audioSrc={audioSrc} /></section>
        </>)}

        {service === 'translate' && (transcribeResult ? <div className="translation-layout">
          <section className="panel source-reference"><TranscriptPanel result={transcribeResult} audioSrc={audioSrc} /></section>
          <section className="panel translation-surface"><TranslatePanel segments={transcribeResult.segments} sourceLang={transcribeResult.detected_language}
            onDone={result => { setTranslateResult(result); setStep('translated'); }} onContinue={() => setService('dub')} /></section>
        </div> : <section className="panel translation-surface pasted-translation"><TranslatePanel segments={[]} sourceLang="en"
          onDone={result => { setTranslateResult(result); setStep('translated'); }} /></section>)}

        {service === 'dub' && (!translateResult || !transcribeResult ? <PrerequisiteState hasTranscript={!!transcribeResult} onAction={() => setService(transcribeResult ? 'translate' : 'transcribe')} /> :
          <TTSPanel segments={translateResult.segments} targetLang={translateResult.target_lang} transcribeJobId={transcribeJobId} originalSrc={audioSrc} hasVideo={isVideo} />)}

        {service === 'voice' && <TextTTSPanel />}
        {service === 'account' && <AccountPanel user={{ ...user, usage }} onLogout={onLogout} />}
        {service === 'admin' && user.is_admin && <AdminPanel />}
      </main>
    </div>
  </div>;
}

function formatUsageMinutes(seconds: number): string {
  const minutes = Math.max(0, seconds) / 60;
  return minutes < 10 ? `${minutes.toFixed(1)} min` : `${Math.floor(minutes)} min`;
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
  const { t } = useLocale();
  const steps = [
    { id: 'transcribe' as const, label: t('transcribe'), done: hasTranscript },
    { id: 'translate' as const, label: t('translate'), done: hasTranslation },
    { id: 'dub' as const, label: t('dub'), done: false },
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
  const { t } = useLocale();
  const [sourceTab, setSourceTab] = useState<'file' | 'youtube'>('file');
  return <div className="intake-layout">
    <section className="panel intake-panel">
      <div className="panel-heading"><span className="icon-box"><Upload size={18} /></span><div><h2>{t('sourceMaterial')}</h2><p>{t('audioVideo')}</p></div></div>
      <div className="segmented-control source-tabs"><button className={sourceTab === 'file' ? 'active' : ''} onClick={() => setSourceTab('file')}><FileAudio size={15} />{t('file')}</button><button className={sourceTab === 'youtube' ? 'active' : ''} onClick={() => setSourceTab('youtube')}><Youtube size={15} />YouTube</button></div>
      {sourceTab === 'file' ? <><DropZone onFile={onFile} disabled={busy} />{file && <button className="button button-primary intake-action" onClick={transcribe} disabled={busy}><FileAudio size={16} />{t('transcribeMedia')}</button>}</> :
        <div className="youtube-source"><label><span className="field-label">{t('movieAddress')}</span><div className="field-with-icon"><Youtube size={16} /><input value={youtubeUrl} onChange={e => setYoutubeUrl(e.target.value)} placeholder="https://youtube.com/watch?v=…" /></div></label><button className="button button-primary" onClick={transcribeYoutube} disabled={busy || !youtubeUrl.trim()}>{t('downloadTranscribe')}</button></div>}
      {(busy || error) && <JobProgress message={message} progress={progress} error={error || undefined} />}
    </section>
    <aside className="intake-summary"><div><FileAudio size={20} /><strong>Transkrypcja</strong><span>TXT, SRT, VTT</span></div><div><Languages size={20} /><strong>Tłumaczenie</strong><span>PL i EN</span></div><div><AudioLines size={20} /><strong>Dubbing</strong><span>WAV i MP4</span></div></aside>
  </div>;
}

function PrerequisiteState({ hasTranscript, onAction }: { hasTranscript: boolean; onAction: () => void }) {
  const { t } = useLocale();
  return <section className="panel prerequisite"><AudioLines size={28} /><div><h2>{hasTranscript ? t('translationMissing') : t('sourceMissing')}</h2><p>{hasTranscript ? t('transcriptReady') : t('startTranscription')}</p></div><button className="button button-primary" onClick={onAction}>{hasTranscript ? t('goTranslation') : t('addMaterial')}<ChevronRight size={16} /></button></section>;
}
