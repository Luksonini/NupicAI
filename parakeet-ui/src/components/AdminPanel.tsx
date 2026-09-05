'use client';

import { useEffect, useState } from 'react';
import {
  Activity, AlertTriangle, CheckCircle2, Cpu, KeyRound, LoaderCircle, RefreshCw,
  Save, Server, ShieldCheck,
} from 'lucide-react';
import type { AdminSettings } from '@/lib/types';
import { getAdminSettings, saveAdminSettings } from '@/lib/api';

const EMPTY_FORM = {
  translation_endpoint: '',
  translation_model: '',
  translation_mode: 'qwen_mtp_35b_json_overlap',
  translation_batch_segments: 8,
  translation_api_key: '',
  clear_translation_api_key: false,
  tts_profile: '',
  mel_steps_first: 8,
  mel_steps_second: 3,
  mel_twopass_t_noise: 0.12,
};

export default function AdminPanel() {
  const [settings, setSettings] = useState<AdminSettings | null>(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    void load();
  }, []);

  const applySettings = (data: AdminSettings) => {
    setSettings(data);
    setForm({
      translation_endpoint: data.translation_endpoint,
      translation_model: data.translation_model,
      translation_mode: data.translation_mode,
      translation_batch_segments: data.translation_batch_segments,
      translation_api_key: '',
      clear_translation_api_key: false,
      tts_profile: data.tts_profile,
      mel_steps_first: data.mel_steps_first,
      mel_steps_second: data.mel_steps_second,
      mel_twopass_t_noise: data.mel_twopass_t_noise,
    });
  };

  const load = async () => {
    setBusy(true); setError(''); setSaved(false);
    try {
      const data = await getAdminSettings();
      applySettings(data);
    } catch (e: unknown) {
      setSettings(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    setBusy(true); setError(''); setSaved(false);
    try {
      const data = await saveAdminSettings(form);
      applySettings(data);
      setSaved(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!settings) {
    return (
      <section className="admin-login panel max-w-md">
        <div className="panel-heading">
          <span className="icon-box"><ShieldCheck size={18} /></span>
          <div><h2>Administrator</h2><p>Uprawnienia przypisane do konta</p></div>
        </div>
        {error && <div className="notice notice-error"><AlertTriangle size={16} />{error}</div>}
        {busy && <div className="admin-loading"><LoaderCircle className="spin" size={18} />Wczytuję konfigurację…</div>}
        {!busy && error && <button className="button button-secondary w-full" onClick={() => void load()}><RefreshCw size={16} />Spróbuj ponownie</button>}
      </section>
    );
  }

  return (
    <div className="admin-grid">
      <section className="panel admin-settings">
        <div className="panel-heading panel-heading-row">
          <div className="flex items-center gap-3">
            <span className="icon-box"><KeyRound size={18} /></span>
            <div><h2>Tłumaczenie API</h2><p>Konfiguracja serwera</p></div>
          </div>
          <button className="icon-button" title="Odśwież" onClick={() => void load()}><RefreshCw size={16} /></button>
        </div>

        <div className="form-grid">
          <label><span className="field-label">Endpoint</span>
            <input value={form.translation_endpoint}
              onChange={e => setForm(v => ({ ...v, translation_endpoint: e.target.value }))} />
          </label>
          <label><span className="field-label">Model</span>
            <input list="translation-models" value={form.translation_model}
              onChange={e => setForm(v => ({ ...v, translation_model: e.target.value }))} />
            <datalist id="translation-models">
              <option value="qwen3.8:27b-mtp" />
              <option value="qwen3.5:35b-mtp" />
            </datalist>
          </label>
          <label><span className="field-label">Tryb</span>
            <select value={form.translation_mode}
              onChange={e => setForm(v => ({ ...v, translation_mode: e.target.value }))}>
              <option value="qwen_mtp_35b_json_overlap">JSON overlap</option>
              <option value="api_numbered">Numbered batches</option>
              <option value="wegorz_local_sentence_split">Lokalny Węgorz</option>
            </select>
          </label>
          <label><span className="field-label">Segmenty w batchu</span>
            <input type="number" min={1} max={20} value={form.translation_batch_segments}
              onChange={e => setForm(v => ({ ...v, translation_batch_segments: Number(e.target.value) }))} />
          </label>
          <label className="form-span-2"><span className="field-label">Nowy API key</span>
            <input type="password" value={form.translation_api_key}
              placeholder={settings.translation_api_key_masked || 'Brak skonfigurowanego klucza'}
              onChange={e => setForm(v => ({ ...v, translation_api_key: e.target.value }))} />
          </label>
          <label className="check-row form-span-2">
            <input type="checkbox" checked={form.clear_translation_api_key}
              onChange={e => setForm(v => ({ ...v, clear_translation_api_key: e.target.checked }))} />
            Usuń zapisany klucz API
          </label>
        </div>
        {error && <div className="notice notice-error"><AlertTriangle size={16} />{error}</div>}
        {saved && <div className="notice notice-success"><CheckCircle2 size={16} />Zapisano konfigurację</div>}
        <button className="button button-primary" disabled={busy} onClick={() => void save()}>
          <Save size={16} /> {busy ? 'Zapisuję…' : 'Zapisz ustawienia'}
        </button>
      </section>

      <section className="panel admin-settings">
        <div className="panel-heading">
          <span className="icon-box"><Cpu size={18} /></span>
          <div><h2>Synteza mowy</h2><p>Domyślny model i parametry flow</p></div>
        </div>
        <div className="form-grid">
          <label className="form-span-2"><span className="field-label">Domyślny model TTS</span>
            <select value={form.tts_profile}
              onChange={e => setForm(v => ({ ...v, tts_profile: e.target.value }))}>
              {settings.tts_models.map(model => <option key={model.key} value={model.key}>{model.label}</option>)}
            </select>
          </label>
          <label><span className="field-label">Kroki first pass</span>
            <input type="number" min={1} max={32} value={form.mel_steps_first}
              onChange={e => setForm(v => ({ ...v, mel_steps_first: Number(e.target.value) }))} />
          </label>
          <label><span className="field-label">Kroki second pass</span>
            <input type="number" min={0} max={16} value={form.mel_steps_second}
              onChange={e => setForm(v => ({ ...v, mel_steps_second: Number(e.target.value) }))} />
          </label>
          <label className="form-span-2"><span className="field-label">Noise drugiego przejścia</span>
            <input type="number" min={0} max={1} step={0.01} value={form.mel_twopass_t_noise}
              onChange={e => setForm(v => ({ ...v, mel_twopass_t_noise: Number(e.target.value) }))} />
          </label>
        </div>
        <button className="button button-primary" disabled={busy || !form.tts_profile} onClick={() => void save()}>
          <Save size={16} /> {busy ? 'Zapisuję…' : 'Zapisz ustawienia TTS'}
        </button>
      </section>

      <aside className="space-y-4">
        <section className="panel">
          <div className="panel-heading"><span className="icon-box"><Server size={18} /></span><div><h2>System</h2><p>Stan usług</p></div></div>
          <div className="status-list">
            <StatusRow label="ASR" ok={settings.model_ready} icon={<Activity size={15} />} />
            <StatusRow label="TTS" ok={settings.tts_ready} icon={<Cpu size={15} />} />
            <StatusRow label="API tłumaczeń" ok={settings.translation_api_key_configured} icon={<KeyRound size={15} />} />
          </div>
          <div className="meta-row"><span>Domyślny profil TTS</span><strong>{settings.tts_profile}</strong></div>
          <div className="meta-row"><span>Aktywny daemon</span><strong>{settings.tts_active_profile}</strong></div>
          <div className="meta-row"><span>Modele w pamięci</span><strong>{settings.tts_loaded_profiles.length}</strong></div>
          <div className="meta-row"><span>Konta użytkowników</span><strong>{settings.registered_users}</strong></div>
          <div className="meta-row"><span>Aktywne sesje</span><strong>{settings.active_sessions}</strong></div>
          <div className="meta-row"><span>Retencja plików</span><strong>{settings.data_retention_hours} h</strong></div>
        </section>
      </aside>

      <section className="panel admin-jobs">
        <div className="panel-heading"><span className="icon-box"><Activity size={18} /></span><div><h2>Ostatnie zadania</h2><p>{settings.recent_jobs.length} zapisów w pamięci serwera</p></div></div>
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Typ</th><th>Status</th><th>ID</th><th>Czas</th><th>Diagnostyka</th></tr></thead>
            <tbody>{settings.recent_jobs.map(job => (
              <tr key={job.id}>
                <td>{job.kind}</td><td><span className={`status-pill ${job.status}`}>{job.status}</span></td>
                <td className="mono">{job.id.slice(0, 10)}</td><td>{job.duration == null ? '—' : `${Number(job.duration).toFixed(1)} s`}</td>
                <td className="mono truncate-cell" title={job.error || job.debug_log}>
                  {job.segments?.length ? <details className="job-debug"><summary>{job.segments.length} segmentów</summary><div>
                    {job.segments.map((segment, index) => <p key={index} className={Number(segment.over_budget ?? 0) > .05 ? 'risk' : ''}>
                      #{segment.index ?? index} · start {segment.start ?? '—'} · audio {segment.audio_duration ?? '—'} s · budżet {segment.target_budget ?? '—'} s · {segment.speed ?? '—'}× · low {segment.low_token_count ?? 0}
                    </p>)}
                  </div></details> : job.error || job.debug_log || '—'}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function StatusRow({ label, ok, icon }: { label: string; ok: boolean; icon: React.ReactNode }) {
  return <div className="status-row"><span>{icon}{label}</span><strong className={ok ? 'ok' : 'bad'}>{ok ? 'Gotowy' : 'Niedostępny'}</strong></div>;
}
