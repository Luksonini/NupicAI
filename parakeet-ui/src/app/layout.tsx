import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Węgorz Dubbing Studio',
  description: 'Transkrypcja, tłumaczenie i dubbing TTS z kontrolą głosu oraz emocji',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl">
      <body className="min-h-screen bg-bg text-slate-200 font-sans antialiased">{children}</body>
    </html>
  );
}
