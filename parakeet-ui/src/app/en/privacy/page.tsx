import type { Metadata } from 'next';
import LegalPage from '@/components/LegalPage';
export const metadata: Metadata = { title: 'NupicAI privacy policy', description: 'How NupicAI processes account data and working media.', alternates: { canonical: '/en/privacy', languages: { pl: '/privacy', en: '/en/privacy' } } };
export default function PrivacyPage() { return <LegalPage locale="en" kind="privacy" />; }
