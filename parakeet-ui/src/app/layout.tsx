import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Węgorz Studio',
  description: 'Transkrypcja, tłumaczenie i dubbing',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl">
      <body className="min-h-screen bg-bg text-slate-200 font-sans antialiased">{children}</body>
    </html>
  );
}
