'use client';

import { useState } from 'react';
import { AlertTriangle, CheckCircle2, Clock3, Gauge, LogOut, ShieldCheck, Trash2, UserRound } from 'lucide-react';
import type { User } from '@/lib/types';
import { deleteMyAccount, deleteMyFiles, logoutAccount } from '@/lib/api';
import { useLocale } from '@/lib/locale';

export default function AccountPanel({ user, onLogout }: { user: User; onLogout: () => void }) {
  const { locale, t } = useLocale();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [deletePassword, setDeletePassword] = useState('');
  const usage = user.usage;
  const usedPercent = usage.unlimited ? 0 : usage.total_seconds > 0
    ? Math.min(100, usage.used_seconds / usage.total_seconds * 100)
    : 100;

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

  const removeAccount = async () => {
    if (!deletePassword) return;
    if (!window.confirm(locale === 'pl' ? 'Trwale usunąć konto, sesje, saldo i wszystkie pliki? Tej operacji nie można cofnąć.' : 'Permanently delete your account, sessions, allowance and all files? This cannot be undone.')) return;
    setBusy(true); setMessage(''); setError('');
    try { await deleteMyAccount(deletePassword); onLogout(); }
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

    <section className="panel account-usage">
      <div className="panel-heading"><span className="icon-box"><Gauge size={18} /></span><div><h2>{locale === 'pl' ? 'Limit generowania' : 'Generation allowance'}</h2><p>{locale === 'pl' ? 'Rozliczany za gotowe audio' : 'Charged for completed audio'}</p></div></div>
      <div className="usage-total"><strong>{usage.unlimited ? (locale === 'pl' ? 'Bez limitu' : 'Unlimited') : formatMinutes(usage.available_seconds)}</strong>{!usage.unlimited && <span>{locale === 'pl' ? 'pozostało' : 'remaining'}</span>}</div>
      {!usage.unlimited && <div className="usage-progress" aria-label={locale === 'pl' ? 'Wykorzystanie limitu' : 'Allowance used'}><i style={{ width: `${usedPercent}%` }} /></div>}
      <div className="usage-details">
        <span>{locale === 'pl' ? 'Wykorzystano' : 'Used'} <b>{formatMinutes(usage.used_seconds)}</b></span>
        {usage.reserved_seconds > 0 && <span>{locale === 'pl' ? 'W trakcie' : 'In progress'} <b>{formatMinutes(usage.reserved_seconds)}</b></span>}
      </div>
      <p className="account-disclosure">{usage.unlimited
        ? (locale === 'pl' ? 'To konto ma nielimitowany dostęp do renderowania.' : 'This account has unlimited rendering access.')
        : (locale === 'pl' ? 'Limit jest rezerwowany przed renderingiem. Nieudane zadanie automatycznie zwalnia rezerwację.' : 'The allowance is reserved before rendering. Failed jobs automatically release their reservation.')}</p>
    </section>

    <section className="panel account-privacy">
      <div className="panel-heading"><span className="icon-box"><ShieldCheck size={18} /></span><div><h2>{t('filesPrivacy')}</h2><p>{t('dataControl')}</p></div></div>
      <div className="retention-notice"><Clock3 size={21} /><div><strong>{locale === 'pl' ? `Automatyczne usuwanie po ${user.data_retention_hours} godzinach` : `Automatic deletion after ${user.data_retention_hours} hours`}</strong><p>{locale === 'pl' ? 'Dotyczy przesłanych materiałów, promptów głosowych, dubbingu i plików roboczych.' : 'This covers uploaded media, voice prompts, dubbing and working files.'}</p></div></div>
      <p className="account-disclosure">{locale === 'pl' ? 'Transkrypcja i synteza głosu działają lokalnie. Tekst może być wysyłany do zewnętrznego API wyłącznie podczas korzystania ze zdalnego tłumaczenia.' : 'Transcription and speech synthesis run locally. Text may be sent to an external API only when remote translation is used.'}</p>
      {message && <div className="notice notice-success"><CheckCircle2 size={16} />{message}</div>}
      {error && <div className="notice notice-error"><AlertTriangle size={16} />{error}</div>}
      <button className="button button-danger" disabled={busy} onClick={() => void removeFiles()}><Trash2 size={16} />{t('deleteFiles')}</button>
      <div className="account-delete">
        <strong>{locale === 'pl' ? 'Zamknięcie konta' : 'Close account'}</strong>
        <p>{locale === 'pl' ? 'Podaj hasło, aby trwale usunąć konto i powiązane dane.' : 'Enter your password to permanently delete the account and its data.'}</p>
        <div><input type="password" autoComplete="current-password" value={deletePassword} onChange={event => setDeletePassword(event.target.value)} placeholder={locale === 'pl' ? 'Aktualne hasło' : 'Current password'} /><button className="button button-danger" disabled={busy || !deletePassword} onClick={() => void removeAccount()}><Trash2 size={16} />{locale === 'pl' ? 'Usuń konto' : 'Delete account'}</button></div>
      </div>
    </section>
  </div>;
}

function formatMinutes(seconds: number): string {
  const minutes = Math.max(0, seconds) / 60;
  return `${minutes < 10 ? minutes.toFixed(1) : Math.floor(minutes)} min`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}
