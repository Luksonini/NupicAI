'use client';

import { useState } from 'react';
import { ArrowRight, CheckCircle2, Copy, Download, Languages } from 'lucide-react';
import type { Segment, TranslateResult } from '@/lib/types';
import { submitTranslation, streamJob } from '@/lib/api';
import JobProgress from './JobProgress';
import { useLocale } from '@/lib/locale';

interface Props {
  segments: Segment[];
  sourceLang: string;
  onDone: (r: TranslateResult) => void;
  onContinue?: () => void;
}

function fmt(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export default function TranslatePanel({ segments, sourceLang, onDone, onContinue }: Props) {
  const { locale, t } = useLocale();
  const [targetLang, setTargetLang] = useState(() => sourceLang.toLowerCase().startsWith('pl') ? 'en' : 'pl');
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [result, setResult] = useState<TranslateResult | null>(null);

  const run = async () => {
    setRunning(true); setError(''); setProgress(0); setMessage('Przygotowuję tłumaczenie…');
    try {
      const jobId = await submitTranslation({
        segments, source_lang: sourceLang, target_lang: targetLang,
        mode: '', model: '', api_key: '', batch_segments: 0,
      });
      await new Promise<void>((resolve, reject) => {
        streamJob(jobId, ev => {
          if (ev.type === 'progress') { setProgress(ev.progress ?? 0); setMessage(ev.message ?? ''); }
          else if (ev.type === 'done') {
            const translated = ev.result as TranslateResult;
            setResult(translated); onDone(translated); resolve();
          } else if (ev.type === 'error') reject(new Error(ev.error ?? 'Błąd tłumaczenia'));
        });
      });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setRunning(false); }
  };

  const exportSrt = () => {
    if (!result) return;
    const lines = result.segments.map((s, i) =>
      `${i + 1}\n${fmt(s.start)} --> ${fmt(s.end)}\n${s.translation ?? s.text}`
    ).join('\n\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines], { type: 'text/plain' }));
    a.download = 'translation.srt'; a.click();
  };

  return (
    <section className="translation-workspace">
      <div className="section-toolbar">
        <div className="panel-heading compact"><span className="icon-box"><Languages size={18} /></span><div><h2>{t('translate')}</h2><p>{segments.length} {locale === 'pl' ? 'segmentów' : 'segments'}</p></div></div>
        <div className="toolbar-actions">
          <label className="inline-field"><span>{t('targetLanguage')}</span>
            <select value={targetLang} onChange={e => setTargetLang(e.target.value)} disabled={running || !!result}>
              <option value="pl">Polski</option><option value="en">English</option>
            </select>
          </label>
          {!result && <button className="button button-primary" onClick={() => void run()} disabled={running}>
            <Languages size={16} />{running ? t('translating') : t('translateAction')}
          </button>}
        </div>
      </div>

      {(running || error) && <JobProgress message={message} progress={progress} error={error || undefined} />}

      {result ? (
        <>
          <div className="success-strip"><CheckCircle2 size={17} /><span>{result.source_lang.toUpperCase()} → {result.target_lang.toUpperCase()}</span><span>{result.elapsed.toFixed(1)} s</span>
            <div className="ml-auto flex gap-2">
              <button className="icon-button" title={t('copy')} onClick={() => navigator.clipboard.writeText(result.translation)}><Copy size={16} /></button>
              <button className="icon-button" title={t('downloadSrt')} onClick={exportSrt}><Download size={16} /></button>
              {onContinue && <button className="button button-primary" onClick={onContinue}>{t('goDubbing')} <ArrowRight size={16} /></button>}
            </div>
          </div>
          <div className="segment-table">
            {result.segments.map((seg, i) => (
              <article className="translation-row" key={`${seg.index}-${i}`}>
                <button className="timecode">{fmt(seg.start)}</button>
                <div><p className="source-line">{seg.source_text ?? seg.text}</p><p className="translation-line">{seg.translation}</p></div>
              </article>
            ))}
          </div>
        </>
      ) : !running && (
        <div className="empty-state compact-empty"><Languages size={24} /><strong>{t('readyTranslation')}</strong><span>{sourceLang.toUpperCase()} · {segments.length} {locale === 'pl' ? 'segmentów' : 'segments'}</span></div>
      )}
    </section>
  );
}
