'use client';
import { useCallback, useState } from 'react';

interface Props {
  onFile: (f: File) => void;
  disabled?: boolean;
}

export default function DropZone({ onFile, disabled }: Props) {
  const [drag, setDrag] = useState(false);
  const [picked, setPicked] = useState<File | null>(null);

  const handle = useCallback((f: File) => {
    setPicked(f);
    onFile(f);
  }, [onFile]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) handle(f);
  }, [handle]);

  const fmt = (n: number) =>
    n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`;

  return (
    <label
      className={[
        'relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors cursor-pointer select-none',
        drag ? 'border-accent bg-[#1a2040]' : picked ? 'border-accent-dim bg-surface' : 'border-border bg-surface hover:border-accent',
        disabled ? 'opacity-50 pointer-events-none' : '',
      ].join(' ')}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
    >
      <span className="text-4xl">{picked ? '🎵' : '🎙️'}</span>
      {picked ? (
        <>
          <span className="font-medium text-slate-200 max-w-xs truncate">{picked.name}</span>
          <span className="text-xs text-muted">{fmt(picked.size)} · kliknij aby zmienić</span>
        </>
      ) : (
        <>
          <span className="text-slate-300">Przeciągnij plik audio lub wideo</span>
          <span className="text-sm text-muted">mp3 · wav · mp4 · mkv · m4a · flac · ogg · webm</span>
        </>
      )}
      <input
        type="file"
        className="sr-only"
        accept="audio/*,video/*,.mp3,.wav,.mp4,.mkv,.m4a,.flac,.ogg,.webm,.aac"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handle(f); }}
        disabled={disabled}
      />
    </label>
  );
}
