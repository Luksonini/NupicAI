'use client';

import { AlertTriangle, ArrowRight, Clock3 } from 'lucide-react';
import { useLocale } from '@/lib/locale';

export default function QuotaExceededNotice() {
  const { locale } = useLocale();
  const openAccount = () => window.dispatchEvent(new CustomEvent('nupicai-open-account'));

  return <div className="quota-notice" role="alert">
    <AlertTriangle size={20} />
    <div>
      <strong>{locale === 'pl' ? 'Limit generowania został wykorzystany' : 'Your generation allowance is exhausted'}</strong>
      <p>{locale === 'pl'
        ? 'Projekt i ustawienia zostały zachowane. Przejdź do konta, aby sprawdzić wykorzystanie i dostępne pakiety.'
        : 'Your project and settings are preserved. Open your account to review usage and available plans.'}</p>
    </div>
    <button className="button button-secondary" onClick={openAccount}><Clock3 size={15} />{locale === 'pl' ? 'Zobacz konto' : 'View account'}<ArrowRight size={15} /></button>
  </div>;
}
