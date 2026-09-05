import type { Metadata } from 'next';
import LegalPage from '@/components/LegalPage';
export const metadata: Metadata = { title: 'Regulamin NupicAI', description: 'Regulamin świadczenia usług drogą elektroniczną NupicAI.', alternates: { canonical: '/regulamin', languages: { pl: '/regulamin', en: '/en/terms' } } };
export default function TermsPage() { return <LegalPage locale="pl" kind="terms" />; }
