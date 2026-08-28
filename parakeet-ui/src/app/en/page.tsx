import type { Metadata } from 'next';
import { NupicAIApp } from '../page';

export const metadata: Metadata = {
  title: 'NupicAI - AI transcription, translation and dubbing',
  description: 'Turn audio or video into a polished Polish or English version with natural voices, matched pacing and the original background.',
  alternates: { canonical: '/en', languages: { pl: '/', en: '/en' } },
  openGraph: {
    title: 'NupicAI - AI transcription, translation and dubbing',
    description: 'One workspace for time-aligned transcripts, contextual translation and natural dubbing.',
    locale: 'en_US',
    type: 'website',
    images: [{ url: '/marketing/hero-nupicai-dubbing-studio.webp', width: 1672, height: 941, alt: 'NupicAI dubbing studio processing a video project' }],
  },
};

export default function EnglishPage() { return <NupicAIApp initialLocale="en" />; }
