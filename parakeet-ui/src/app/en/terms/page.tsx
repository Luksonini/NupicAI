import type { Metadata } from 'next';
import LegalPage from '@/components/LegalPage';
export const metadata: Metadata = { title: 'NupicAI terms of service', description: 'Terms for NupicAI electronic services.', alternates: { canonical: '/en/terms', languages: { pl: '/regulamin', en: '/en/terms' } } };
export default function TermsPage() { return <LegalPage locale="en" kind="terms" />; }
