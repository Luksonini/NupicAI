import type { Metadata } from 'next';
import './globals.css';

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:8765';

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: 'NupicAI - transkrypcja, tłumaczenie i dubbing AI',
  description: 'Transkrybuj audio i wideo w 25 językach, tłumacz i twórz naturalny dubbing po polsku lub angielsku.',
  icons: { icon: '/brand/mark.png', apple: '/brand/mark.png' },
  manifest: '/manifest.webmanifest',
  alternates: { canonical: '/', languages: { pl: '/', en: '/en' } },
  openGraph: {
    title: 'NupicAI - transkrypcja, tłumaczenie i dubbing AI',
    description: 'Transkrypcja 25 języków, tłumaczenie i naturalny dubbing po polsku lub angielsku w jednym studio.',
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
  alternateName: 'Neural Unified Platform for Intelligent Communication',
  applicationCategory: 'MultimediaApplication',
  operatingSystem: 'Web',
  inLanguage: ['pl', 'en'],
  description: 'Studio AI do transkrypcji, tłumaczenia i dubbingu audio oraz wideo.',
  featureList: [
    'Time-aligned transcription in 25 European languages',
    'Translation from supported languages into Polish or English',
    'Natural Polish and English AI dubbing',
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
