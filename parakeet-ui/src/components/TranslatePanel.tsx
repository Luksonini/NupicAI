'use client';
import { useState } from 'react';
import type { Segment, TranslateResult } from '@/lib/types';
import { TRANSLATION_MODES } from '@/lib/types';
import { submitTranslation, streamJob } from '@/lib/api';
import JobProgress from './JobProgress';

interface Props {
  segments: Segment[];
  sourceLang: string;
  onDone: (r: TranslateResult) => void;
}

function fmt(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export default function TranslatePanel({ segments, sourceLang, onDone }: Props) {
  const [mode, setMode] = useState<string>(TRANSLATION_MODES[0].value);
  const [model, setModel] = useState('qwen3.5:35b-mtp');
  const [apiKey, setApiKey] = useState('');
  const [targetLang, setTargetLang] = useState('pl');
  const [batchSize, setBatchSize] = useState(8);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState<TranslateResult | null>(null);

  const run = async () => {
    setRunning(true); setError(''); setProgress(0); setMessage('Inicjalizuję…');
    try {
      const jobId = await submitTranslation({
        segments,
        source_lang: sourceLang,
        target_lang: targetLang,
        mode,
        model,
        api_key: apiKey,
        batch_segments: batchSize,
      });
      await new Promise<void>((resolve, reject) => {
        streamJob(jobId, (ev) => {
          if (ev.type === 'progress') {
            setProgress(ev.progress ?? 0);
            setMessage(ev.message ?? '');
          } else if (ev.type === 'done') {
            const r = ev.result as TranslateResult;
            setResult(r); onDone(r); resolve();
          } else if (ev.type === 'error') {
            reject(new Error(ev.error ?? 'Błąd tłumaczenia'));
          }
        });
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-base font-semibold">Tłumaczenie</h2>

      {/* Config */}
      {!result && (
        <div className="bg-surface rounded-xl border border-border p-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted block mb-1">Tryb</label>
              <select value={mode} onChange={e => {
                const nextMode = e.target.value;
                setMode(nextMode);
                if (nextMode === 'qwen_mtp_35b_json_overlap') setModel('qwen3.5:35b-mtp');
                else if (nextMode === 'api_json_overlap' && model === 'qwen3.5:35b-mtp') setModel('qwen3.5:35b-mtp');
                else if (nextMode === 'api_numbered' && model === 'qwen3.5:35b-mtp') setModel('qwen3.5:35b-mtp');
              }}
                className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm">
                {TRANSLATION_MODES.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted block mb-1">Język docelowy</label>
              <select value={targetLang} onChange={e => setTargetLang(e.target.value)}
                className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm">
                <option value="pl">Polski</option>
                <option value="en">English</option>
              </select>
            </div>
          </div>

          {mode !== 'wegorz_local_sentence_split' && (
            <>
              <div>
                <label className="text-xs text-muted block mb-1">Model API</label>
                <input value={model} onChange={e => setModel(e.target.value)}
                  className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm"
                  placeholder="qwen3.5:35b-mtp" />
              </div>
              <div>
                <label className="text-xs text-muted block mb-1">API Key</label>
                <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
                  className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm"
                  placeholder="sk-…  (lub ustaw NUPIC_API_KEY w środowisku)" />
              </div>
              <div>
                <label className="text-xs text-muted block mb-1">Batch (segmentów)</label>
                <input type="number" min={1} max={20} value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}
                  className="w-full bg-surface2 border border-border rounded-lg px-3 py-2 text-sm" />
              </div>
            </>
          )}

          <button onClick={run} disabled={running}
            className="w-full py-2.5 bg-accent text-white font-semibold rounded-lg hover:bg-blue-400 disabled:opacity-40 transition-colors">
            {running ? 'Tłumaczę…' : '▶ Tłumacz'}
          </button>

          {(running || error) && (
            <JobProgress message={message} progress={progress} error={error || undefined} />
          )}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-xs text-muted">
            <span className="bg-green-800/40 text-green-300 px-2 py-0.5 rounded font-semibold">
              {result.source_lang.toUpperCase()} → {result.target_lang.toUpperCase()}
            </span>
            <span>{result.model}</span>
            <span>{result.elapsed.toFixed(1)} s</span>
            <button onClick={() => { setResult(null); setProgress(0); setMessage(''); }}
              className="ml-auto text-muted hover:text-slate-200">Zmień ustawienia</button>
          </div>

          <div className="flex gap-2">
            <button onClick={() => navigator.clipboard.writeText(result.translation)}
              className="text-xs px-3 py-1 rounded bg-surface2 border border-border hover:bg-border transition-colors">
              Kopiuj
            </button>
            <button onClick={() => {
              const lines = result.segments.map((s, i) =>
                `${i + 1}\n${fmt(s.start)} --> ${fmt(s.end)}\n${s.translation ?? s.text}`
              ).join('\n\n');
              const a = document.createElement('a');
              a.href = URL.createObjectURL(new Blob([lines], { type: 'text/plain' }));
              a.download = 'translation.srt'; a.click();
            }} className="text-xs px-3 py-1 rounded bg-surface2 border border-border hover:bg-border transition-colors">
              SRT
            </button>
          </div>

          <ul className="flex flex-col gap-1 max-h-96 overflow-y-auto pr-1">
            {result.segments.map((seg, i) => (
              <li key={i} className="flex gap-3 items-start px-3 py-2 rounded-lg bg-surface border border-border">
                <span className="text-accent text-xs font-semibold tabular-nums pt-0.5 min-w-[36px]">{fmt(seg.start)}</span>
                <div className="flex flex-col gap-0.5">
                  <span className="text-xs text-muted">{seg.source_text ?? seg.text}</span>
                  <span className="text-sm">{seg.translation}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
