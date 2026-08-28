'use client';
import { useEffect, useRef, useState } from 'react';
import type { Segment, Word, TranscribeResult } from '@/lib/types';
import { useLocale } from '@/lib/locale';

interface Props {
  result: TranscribeResult;
  audioSrc: string;
}

type Tab = 'transcript' | 'segments' | 'timeline';

function fmt(s: number) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

function srtTs(s: number) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60),
    sec = Math.floor(s % 60), ms = Math.round((s % 1) * 1000);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

export default function TranscriptPanel({ result, audioSrc }: Props) {
  const { locale, t } = useLocale();
  const [tab, setTab] = useState<Tab>('transcript');
  const [activeWord, setActiveWord] = useState(-1);
  const [activeSeg, setActiveSeg] = useState(-1);
  const audioRef = useRef<HTMLAudioElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const tick = () => {
      const t = audio.currentTime;
      const wi = result.words.findIndex(w => w.start <= t && w.end >= t);
      setActiveWord(wi >= 0 ? wi : result.words.filter(w => w.start <= t).length - 1);
      const si = result.segments.findIndex(s => s.start <= t && s.end >= t);
      setActiveSeg(si);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [result]);

  const seek = (t: number) => { if (audioRef.current) { audioRef.current.currentTime = t; audioRef.current.play(); } };

  const exportSrt = () => {
    const lines = result.segments.map((s, i) =>
      `${i + 1}\n${srtTs(s.start)} --> ${srtTs(s.end)}\n${s.text}`
    ).join('\n\n');
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lines], { type: 'text/plain' }));
    a.download = 'transcript.srt'; a.click();
  };

  const downloadText = (contents: string, filename: string, type = 'text/plain') => {
    const url = URL.createObjectURL(new Blob([contents], { type }));
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportText = () => downloadText(result.transcript, 'transcript.txt');

  const exportVtt = () => {
    const lines = result.segments.map((s, i) =>
      `${i + 1}\n${srtTs(s.start).replace(',', '.')} --> ${srtTs(s.end).replace(',', '.')}\n${s.text}`
    ).join('\n\n');
    downloadText(`WEBVTT\n\n${lines}\n`, 'transcript.vtt', 'text/vtt');
  };

  const tabs: { id: Tab; label: string }[] = [
    { id: 'transcript', label: t('transcribe') },
    { id: 'segments', label: `${locale === 'pl' ? 'Segmenty' : 'Segments'} (${result.segment_count})` },
    { id: 'timeline', label: locale === 'pl' ? 'Oś czasu' : 'Timeline' },
  ];

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center gap-3 flex-wrap">
        <h2 className="text-base font-semibold">{t('materialSource')}</h2>
        <span className="bg-accent-dim text-blue-200 text-xs font-bold px-2 py-0.5 rounded">
          {(result.detected_language || 'auto').toUpperCase()}
        </span>
        <span className="text-xs text-muted">
          {result.word_count} {locale === 'pl' ? 'słów' : 'words'} · {result.segment_count} {locale === 'pl' ? 'segmentów' : 'segments'} · {fmt(result.duration)}
        </span>
        <div className="ml-auto flex gap-2">
          <button onClick={() => navigator.clipboard.writeText(result.transcript)}
            className="text-xs px-3 py-1 rounded bg-surface2 border border-border hover:bg-border transition-colors">
            {t('copy')}
          </button>
          <button onClick={exportSrt}
            className="text-xs px-3 py-1 rounded bg-surface2 border border-border hover:bg-border transition-colors">
            SRT
          </button>
          <button onClick={exportVtt}
            className="text-xs px-3 py-1 rounded bg-surface2 border border-border hover:bg-border transition-colors">
            VTT
          </button>
          <button onClick={exportText}
            className="text-xs px-3 py-1 rounded bg-surface2 border border-border hover:bg-border transition-colors">
            TXT
          </button>
        </div>
      </div>

      {/* Audio player */}
      <audio ref={audioRef} src={audioSrc} controls className="w-full accent-accent" />

      {/* Tabs */}
      <div className="flex border-b border-border">
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 transition-colors ${
              tab === t.id ? 'border-accent text-accent' : 'border-transparent text-muted hover:text-slate-200'
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Transcript */}
      {tab === 'transcript' && (
        <pre className="bg-surface rounded-xl border border-border p-4 text-sm leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">
          {result.transcript}
        </pre>
      )}

      {/* Segments */}
      {tab === 'segments' && (
        <ul className="flex flex-col gap-1 max-h-96 overflow-y-auto pr-1">
          {result.segments.map((seg, i) => (
            <li key={i} onClick={() => seek(seg.start)}
              className={`flex gap-3 items-start px-3 py-2 rounded-lg cursor-pointer border transition-colors ${
                activeSeg === i ? 'bg-accent/10 border-accent' : 'border-transparent hover:bg-surface2'
              }`}>
              <span className="text-accent text-xs font-semibold tabular-nums pt-0.5 min-w-[36px]">{fmt(seg.start)}</span>
              <span className="text-sm">{seg.text}</span>
            </li>
          ))}
        </ul>
      )}

      {/* Word timeline */}
      {tab === 'timeline' && (
        <div>
          <div className="relative bg-surface border border-border rounded-xl h-10 overflow-hidden">
            {result.words.map((w, i) => {
              const left = (w.start / result.duration) * 100;
              const width = Math.max(0.3, ((w.end - w.start) / result.duration) * 100);
              return (
                <span key={i} onClick={() => seek(w.start)}
                  title={`${w.word} (${fmt(w.start)})`}
                  className={`absolute top-1.5 h-7 rounded text-xs flex items-center px-1 cursor-pointer border overflow-hidden transition-colors ${
                    activeWord === i ? 'bg-word text-black border-word z-10' : 'bg-surface2 border-border hover:bg-border'
                  }`}
                  style={{ left: `${left}%`, width: `${width}%` }}>
                  <span className="truncate">{w.word}</span>
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
