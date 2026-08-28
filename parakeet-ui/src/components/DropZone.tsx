'use client';

import { useCallback, useState } from 'react';
import { CheckCircle2, FileAudio, UploadCloud } from 'lucide-react';
import { useLocale } from '@/lib/locale';

interface Props { onFile: (file: File) => void; disabled?: boolean; }

export default function DropZone({ onFile, disabled }: Props) {
  const { locale, t } = useLocale();
  const [drag, setDrag] = useState(false);
  const [picked, setPicked] = useState<File | null>(null);
  const handle = useCallback((file: File) => { setPicked(file); onFile(file); }, [onFile]);
  const fmt = (bytes: number) => bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;

  return <label className={`drop-zone ${drag ? 'dragging' : ''} ${picked ? 'picked' : ''} ${disabled ? 'disabled' : ''}`}
    onDragOver={event => { event.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
    onDrop={event => { event.preventDefault(); setDrag(false); const file = event.dataTransfer.files[0]; if (file) handle(file); }}>
    <span className="drop-icon">{picked ? <CheckCircle2 size={25} /> : <UploadCloud size={25} />}</span>
    {picked ? <div className="drop-copy"><strong><FileAudio size={15} />{picked.name}</strong><span>{fmt(picked.size)} · {locale === 'pl' ? 'wybierz inny plik' : 'choose another file'}</span></div> :
      <div className="drop-copy"><strong>{t('dropFile')}</strong><span>MP3, WAV, MP4, MKV, M4A, FLAC, OGG, WEBM</span></div>}
    <input type="file" className="sr-only" accept="audio/*,video/*,.mp3,.wav,.mp4,.mkv,.m4a,.flac,.ogg,.webm,.aac"
      onChange={event => { const file = event.target.files?.[0]; if (file) handle(file); }} disabled={disabled} />
  </label>;
}
