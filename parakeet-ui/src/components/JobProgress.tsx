'use client';

interface Props {
  message: string;
  progress: number; // 0-1
  error?: string;
}

export default function JobProgress({ message, progress, error }: Props) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className={error ? 'text-red-400' : 'text-muted'}>{message}</span>
        <span className="text-muted tabular-nums">{Math.round(progress * 100)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-surface2 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${error ? 'bg-red-500' : 'bg-accent'}`}
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </div>
  );
}
