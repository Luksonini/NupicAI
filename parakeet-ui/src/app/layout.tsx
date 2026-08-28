import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'NupicAI Studio',
  description: 'Transkrypcja, tłumaczenie, dubbing i synteza głosu',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl">
      <body className="min-h-screen bg-bg text-slate-200 font-sans antialiased">{children}</body>
    </html>
  );
}
