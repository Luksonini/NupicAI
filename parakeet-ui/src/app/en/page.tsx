import type { Metadata } from 'next';
import { NupicAIApp } from '../page';

export const metadata: Metadata = {
  title: 'NupicAI - AI transcription, translation and dubbing',
  description: 'Transcribe audio and video in 25 European languages, translate it and create natural Polish or English dubbing.',
  alternates: { canonical: '/en', languages: { pl: '/', en: '/en' } },
  openGraph: {
    title: 'NupicAI - AI transcription, translation and dubbing',
    description: 'One workspace for transcription in 25 European languages, contextual translation and natural Polish or English dubbing.',
    locale: 'en_US',
    type: 'website',
    images: [{ url: '/marketing/hero-nupicai-dubbing-studio.webp', width: 1672, height: 941, alt: 'NupicAI dubbing studio processing a video project' }],
  },
};

export default function EnglishPage() { return <NupicAIApp initialLocale="en" />; }
