'use client';

import { useState } from 'react';
import { AlertTriangle, CheckCircle2, Clock3, LogOut, ShieldCheck, Trash2, UserRound } from 'lucide-react';
import type { User } from '@/lib/types';
import { deleteMyFiles, logoutAccount } from '@/lib/api';
import { useLocale } from '@/lib/locale';

export default function AccountPanel({ user, onLogout }: { user: User; onLogout: () => void }) {
  const { locale, t } = useLocale();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const removeFiles = async () => {
    if (!window.confirm(locale === 'pl' ? 'Usunąć teraz wszystkie Twoje pliki robocze i wyniki? Tej operacji nie można cofnąć.' : 'Delete all your working files and results now? This cannot be undone.')) return;
    setBusy(true); setMessage(''); setError('');
    try {
      const result = await deleteMyFiles();
      setMessage(`${locale === 'pl' ? 'Usunięto dane' : 'Data removed'} (${formatBytes(result.removed_bytes)}).`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  };

  const logout = async () => {
    setBusy(true); setError('');
    try { await logoutAccount(); onLogout(); }
    catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); setBusy(false); }
  };

  return <div className="account-layout">
    <section className="panel account-profile">
      <div className="panel-heading"><span className="icon-box"><UserRound size={18} /></span><div><h2>{t('account')}</h2><p>{t('profileSession')}</p></div></div>
      <dl className="account-meta">
        <div><dt>{t('name')}</dt><dd>{user.display_name}</dd></div>
        <div><dt>E-mail</dt><dd>{user.email}</dd></div>
        <div><dt>{t('accountSince')}</dt><dd>{new Date(user.created_at * 1000).toLocaleDateString(locale === 'pl' ? 'pl-PL' : 'en-GB')}</dd></div>
      </dl>
      <button className="button button-secondary" disabled={busy} onClick={() => void logout()}><LogOut size={16} />{t('logout')}</button>
    </section>

    <section className="panel account-privacy">
      <div className="panel-heading"><span className="icon-box"><ShieldCheck size={18} /></span><div><h2>{t('filesPrivacy')}</h2><p>{t('dataControl')}</p></div></div>
      <div className="retention-notice"><Clock3 size={21} /><div><strong>{locale === 'pl' ? `Automatyczne usuwanie po ${user.data_retention_hours} godzinach` : `Automatic deletion after ${user.data_retention_hours} hours`}</strong><p>{locale === 'pl' ? 'Dotyczy przesłanych materiałów, promptów głosowych, dubbingu i plików roboczych.' : 'This covers uploaded media, voice prompts, dubbing and working files.'}</p></div></div>
      <p className="account-disclosure">{locale === 'pl' ? 'Transkrypcja i synteza głosu działają lokalnie. Tekst może być wysyłany do zewnętrznego API wyłącznie podczas korzystania ze zdalnego tłumaczenia.' : 'Transcription and speech synthesis run locally. Text may be sent to an external API only when remote translation is used.'}</p>
      {message && <div className="notice notice-success"><CheckCircle2 size={16} />{message}</div>}
      {error && <div className="notice notice-error"><AlertTriangle size={16} />{error}</div>}
      <button className="button button-danger" disabled={busy} onClick={() => void removeFiles()}><Trash2 size={16} />{t('deleteFiles')}</button>
    </section>
  </div>;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}
