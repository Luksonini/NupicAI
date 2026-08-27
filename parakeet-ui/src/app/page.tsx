'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { TranscribeResult, TranslateResult, Step } from '@/lib/types';
import { uploadAndTranscribe, transcribeYoutube, sourceUrl, streamJob, checkHealth } from '@/lib/api';
import DropZone from '@/components/DropZone';
import JobProgress from '@/components/JobProgress';
import TranscriptPanel from '@/components/TranscriptPanel';
import TranslatePanel from '@/components/TranslatePanel';
import TTSPanel from '@/components/TTSPanel';
import TextTTSPanel from '@/components/TextTTSPanel';

export default function Home() {
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
  const [ttsTab, setTtsTab] = useState<'dubbing' | 'text'>('dubbing');
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    checkHealth().then(ok => setHealth(ok));
  }, []);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setSourceKind('upload');
    if (audioSrc) URL.revokeObjectURL(audioSrc);
    setAudioSrc(URL.createObjectURL(f));
    setStep('idle');
    setTranscribeResult(null);
    setTranslateResult(null);
    setTranscribeJobId('');
    setError('');
  }, [audioSrc]);

  const runTranscriptionJob = async (createJob: () => Promise<string>, nextAudioSrc?: (jobId: string) => string) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStep('transcribing'); setProgress(0); setMessage('Startuję transkrypcję…'); setError('');
    try {
      const jobId = await createJob();
      setTranscribeJobId(jobId);
      await new Promise<void>((resolve, reject) => {
        ctrl.signal.addEventListener('abort', reject);
        streamJob(jobId, (ev) => {
          if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
          else if (ev.type === 'done') { setTranscribeResult(ev.result as TranscribeResult); resolve(); }
          else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd transkrypcji'));
        }, ctrl.signal);
      });
      if (nextAudioSrc) setAudioSrc(nextAudioSrc(jobId));
      setStep('transcribed');
    } catch (e: unknown) {
      if (ctrl.signal.aborted) return;
      setError(e instanceof Error ? e.message : String(e));
      setStep('error');
    }
  };

  const transcribe = async () => {
    if (!file) return;
    await runTranscriptionJob(() => uploadAndTranscribe(file));
  };

  const transcribeFromYoutube = async () => {
    const url = youtubeUrl.trim();
    if (!url) return;
    setFile(null);
    setSourceKind('youtube');
    setAudioSrc('');
    setTranscribeResult(null);
    setTranslateResult(null);
    setTranscribeJobId('');
    await runTranscriptionJob(() => transcribeYoutube(url), sourceUrl);
  };

  const isVideo = sourceKind === 'youtube'
    ? true
    : file ? file.type.startsWith('video/') || /\.(mp4|mkv|webm|avi|mov)$/i.test(file.name) : false;

  const STEPS = [
    { id: 'idle', label: 'Upload' },
    { id: 'transcribing', label: 'Transkrypcja' },
    { id: 'transcribed', label: 'Tłumaczenie' },
    { id: 'translated', label: 'Dubbing' },
  ];
  const stepIdx = { idle: 0, transcribing: 1, transcribed: 2, translating: 2, translated: 3, error: 0 }[step];

  return (
    <main className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Węgorz Dubbing Studio</h1>
          <p className="text-sm text-muted">Transkrypcja + Tłumaczenie + Dubbing TTS</p>
        </div>
        <div className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border ${
          health === null ? 'border-border text-muted' :
          health ? 'border-green-700 bg-green-900/20 text-green-400' : 'border-red-700 bg-red-900/20 text-red-400'
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${health === null ? 'bg-muted' : health ? 'bg-green-400' : 'bg-red-400'}`} />
          {health === null ? 'Sprawdzam…' : health ? 'Serwer OK' : 'Serwer offline'}
        </div>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-0">
        {STEPS.map((s, i) => (
          <div key={s.id} className="flex items-center">
            <div className={`flex items-center gap-2 text-sm px-3 py-1.5 rounded-lg ${
              i < stepIdx ? 'text-green-400' : i === stepIdx ? 'text-accent font-semibold' : 'text-muted'
            }`}>
              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
                i < stepIdx ? 'bg-green-800/50 text-green-400' :
                i === stepIdx ? 'bg-accent-dim text-white' : 'bg-surface2 text-muted'
              }`}>{i < stepIdx ? '✓' : i + 1}</span>
              {s.label}
            </div>
            {i < STEPS.length - 1 && <div className="w-8 h-px bg-border mx-1" />}
          </div>
        ))}
      </div>

      {/* Upload + transcribe */}
      <div className="grid grid-cols-1 gap-4">
        <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
          <label className="text-xs text-muted block">YouTube URL</label>
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              value={youtubeUrl}
              onChange={e => setYoutubeUrl(e.target.value)}
              disabled={step === 'transcribing' || step === 'translating'}
              placeholder="https://www.youtube.com/watch?v=..."
              className="flex-1 bg-surface2 border border-border rounded-lg px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              onClick={transcribeFromYoutube}
              disabled={!youtubeUrl.trim() || step === 'transcribing' || step === 'translating'}
              className="px-4 py-2 bg-accent text-white font-semibold rounded-lg hover:bg-blue-400 disabled:opacity-40 transition-colors">
              Pobierz i transkrybuj
            </button>
          </div>
        </div>
        <DropZone onFile={handleFile} disabled={step === 'transcribing' || step === 'translating'} />
        {file && step === 'idle' && (
          <button onClick={transcribe}
            className="w-full py-3 bg-accent text-white font-semibold rounded-xl hover:bg-blue-400 transition-colors">
            ▶ Transkrybuj
          </button>
        )}
        {step === 'transcribing' && (
          <JobProgress message={message} progress={progress} error={error || undefined} />
        )}
        {step === 'error' && (
          <div className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded-xl px-4 py-3">
            ❌ {error}
            <button onClick={() => setStep('idle')} className="ml-3 underline text-red-300">Resetuj</button>
          </div>
        )}
      </div>

      {/* Results: transcript + translation */}
      {transcribeResult && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-surface rounded-xl border border-border p-5">
            <TranscriptPanel result={transcribeResult} audioSrc={audioSrc} />
          </div>
          <div className="bg-surface rounded-xl border border-border p-5">
            <TranslatePanel
              segments={transcribeResult.segments}
              sourceLang={transcribeResult.detected_language}
              onDone={(r) => { setTranslateResult(r); setStep('translated'); }}
            />
          </div>
        </div>
      )}

      {/* TTS section with tabs */}
      {(translateResult || true) && (
        <div className="bg-surface rounded-xl border border-border">
          {/* Tabs */}
          <div className="flex border-b border-border">
            {translateResult && (
              <button
                onClick={() => setTtsTab('dubbing')}
                className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  ttsTab === 'dubbing'
                    ? 'border-accent text-accent'
                    : 'border-transparent text-muted hover:text-slate-300'
                }`}>
                Dubbing
              </button>
            )}
            <button
              onClick={() => setTtsTab('text')}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                ttsTab === 'text'
                  ? 'border-accent text-accent'
                  : 'border-transparent text-muted hover:text-slate-300'
              }`}>
              TTS z tekstu
            </button>
          </div>
          <div className="p-5">
            {ttsTab === 'dubbing' && translateResult ? (
              <TTSPanel
                segments={translateResult.segments}
                targetLang={translateResult.target_lang}
                transcribeJobId={transcribeJobId}
                originalSrc={audioSrc}
                hasVideo={isVideo}
              />
            ) : (
              <TextTTSPanel />
            )}
          </div>
        </div>
      )}
    </main>
  );
}
