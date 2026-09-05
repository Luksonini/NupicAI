import type { Metadata } from 'next';
import LegalPage from '@/components/LegalPage';
export const metadata: Metadata = { title: 'Polityka prywatności NupicAI', description: 'Informacje o przetwarzaniu danych w NupicAI.', alternates: { canonical: '/privacy', languages: { pl: '/privacy', en: '/en/privacy' } } };
export default function PrivacyPage() { return <LegalPage locale="pl" kind="privacy" />; }
