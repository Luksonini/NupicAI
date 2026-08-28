import type { Metadata } from 'next';
import './globals.css';

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:8765';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: 'NupicAI - transkrypcja, tłumaczenie i dubbing AI',
  description: 'Zamień audio lub wideo w gotową polską wersję z naturalnym głosem, zachowanym tempem i oryginalnym tłem.',
  icons: { icon: '/brand/mark.png', apple: '/brand/mark.png' },
  manifest: '/manifest.webmanifest',
  alternates: { canonical: '/', languages: { pl: '/', en: '/en' } },
  openGraph: {
    title: 'NupicAI - transkrypcja, tłumaczenie i dubbing AI',
    description: 'Jedno studio do transkrypcji, tłumaczenia i naturalnego dubbingu audio oraz wideo.',
    locale: 'pl_PL',
    type: 'website',
    images: [{ url: '/marketing/hero-nupicai-dubbing-studio.webp', width: 1672, height: 941, alt: 'Studio dubbingowe NupicAI podczas pracy nad materiałem wideo' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NupicAI',
    description: 'Transkrypcja, tłumaczenie i naturalny dubbing w jednym miejscu.',
    images: ['/marketing/hero-nupicai-dubbing-studio.webp'],
  },
};

const structuredData = {
  '@context': 'https://schema.org',
  '@type': 'WebApplication',
  name: 'NupicAI',
  applicationCategory: 'MultimediaApplication',
  operatingSystem: 'Web',
  inLanguage: ['pl', 'en'],
  description: 'Studio AI do transkrypcji, tłumaczenia i dubbingu audio oraz wideo.',
  featureList: [
    'Time-aligned audio and video transcription',
    'Polish and English translation',
    'Natural AI dubbing',
    'Subtitle, WAV and MP4 export',
  ],
  offers: { '@type': 'Offer', price: '0', priceCurrency: 'PLN', description: 'Bezpłatny dostęp w okresie pilotażowym' },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pl">
      <body className="min-h-screen bg-bg text-slate-200 font-sans antialiased">
        <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }} />
        {children}
      </body>
    </html>
  );
}
