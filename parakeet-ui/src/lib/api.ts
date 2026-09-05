import type { AdminSettings, JobEvent, Speaker, DubParams, TTSFlowDefaults, TTSModelProfile, Usage, User } from './types';

const BASE = '';

export class ApiRequestError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
  }
}

async function apiError(res: Response): Promise<ApiRequestError> {
  const body = await res.json().catch(() => ({ detail: res.statusText }));
  return new ApiRequestError(String(body.detail ?? res.statusText), res.status);
}

export async function currentUser(): Promise<User | null> {
  const res = await fetch(`${BASE}/auth/me`, { credentials: 'same-origin' });
  if (res.status === 401) return null;
  if (!res.ok) throw await apiError(res);
  return (await res.json() as { user: User }).user;
}

export async function registerAccount(params: {
  email: string; display_name: string; password: string; terms_accepted: boolean;
}): Promise<User> {
  const res = await fetch(`${BASE}/auth/register`, {
    method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw await apiError(res);
  return (await res.json() as { user: User }).user;
}

export async function loginAccount(email: string, password: string): Promise<User> {
  const res = await fetch(`${BASE}/auth/login`, {
    method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw await apiError(res);
  return (await res.json() as { user: User }).user;
}

export async function requestPasswordReset(email: string): Promise<string> {
  const res = await fetch(`${BASE}/auth/forgot-password`, {
    method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) throw await apiError(res);
  return String((await res.json() as { message?: string }).message ?? '');
}

export async function resetPassword(token: string, password: string): Promise<string> {
  const res = await fetch(`${BASE}/auth/reset-password`, {
    method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, password }),
  });
  if (!res.ok) throw await apiError(res);
  return String((await res.json() as { message?: string }).message ?? '');
}

export async function logoutAccount(): Promise<void> {
  const res = await fetch(`${BASE}/auth/logout`, { method: 'POST', credentials: 'same-origin' });
  if (!res.ok) throw await apiError(res);
}

export async function deleteMyFiles(): Promise<{ removed_bytes: number }> {
  const res = await fetch(`${BASE}/account/files`, { method: 'DELETE', credentials: 'same-origin' });
  if (!res.ok) throw await apiError(res);
  return await res.json() as { removed_bytes: number };
}

export async function deleteMyAccount(password: string): Promise<void> {
  const res = await fetch(`${BASE}/account`, {
    method: 'DELETE', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) throw await apiError(res);
}

export async function accountUsage(): Promise<Usage> {
  const res = await fetch(`${BASE}/account/usage`, { credentials: 'same-origin' });
  if (!res.ok) throw await apiError(res);
  return (await res.json() as { usage: Usage }).usage;
}

export async function uploadAndTranscribe(file: File): Promise<string> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/transcribe`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  const data = await res.json();
  return data.job_id as string;
}

export async function transcribeYoutube(url: string): Promise<string> {
  const res = await fetch(`${BASE}/transcribe_youtube`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  const data = await res.json();
  return data.job_id as string;
}

export function sourceUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/source`;
}

export async function submitTranslation(params: {
  segments: unknown[];
  source_lang: string;
  target_lang: string;
  mode: string;
  model: string;
  api_key: string;
  batch_segments: number;
}): Promise<string> {
  const res = await fetch(`${BASE}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  const data = await res.json();
  return data.job_id as string;
}

export async function listSpeakers(): Promise<Speaker[]> {
  const res = await fetch(`${BASE}/speakers`);
  if (!res.ok) return [];
  const data = await res.json();
  return (data.speakers ?? []) as Speaker[];
}

export async function listTTSModels(): Promise<{ default: string; active: string; models: TTSModelProfile[]; flow_defaults: TTSFlowDefaults }> {
  const res = await fetch(`${BASE}/tts_models`);
  if (!res.ok) return { default: '', active: '', models: [], flow_defaults: { mel_steps_first: 8, mel_steps_second: 3, mel_twopass_t_noise: 0.12 } };
  const data = await res.json();
  return {
    default: String(data.default ?? ''),
    active: String(data.active ?? ''),
    models: (data.models ?? []) as TTSModelProfile[],
    flow_defaults: {
      mel_steps_first: Number(data.flow_defaults?.mel_steps_first ?? 8),
      mel_steps_second: Number(data.flow_defaults?.mel_steps_second ?? 3),
      mel_twopass_t_noise: Number(data.flow_defaults?.mel_twopass_t_noise ?? 0.12),
    },
  };
}

export async function uploadVoicePrompt(file: File, startSec = 0, maxSec = 12): Promise<Speaker> {
  const form = new FormData();
  form.append('file', file);
  form.append('start_sec', String(startSec));
  form.append('max_sec', String(maxSec));
  const res = await fetch(`${BASE}/voice_prompt`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  const data = await res.json();
  return data.speaker as Speaker;
}

export async function submitDub(params: DubParams): Promise<string> {
  const res = await fetch(`${BASE}/dub`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw await apiError(res);
  const data = await res.json();
  return data.job_id as string;
}

export async function submitTextTTS(params: Record<string, unknown>): Promise<string> {
  const res = await fetch(`${BASE}/tts_text`, {
    method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw await apiError(res);
  return (await res.json() as { job_id: string }).job_id;
}

export function dubAudioUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/audio`;
}

export function mixAudioUrl(jobId: string): string {
  return `${BASE}/jobs/${jobId}/mix_audio`;
}

export function mixVideoUrl(dubJobId: string, transcribeJobId: string): string {
  return `${BASE}/mix_video?dub_job_id=${dubJobId}&transcribe_job_id=${transcribeJobId}`;
}

export function streamJob(
  jobId: string,
  onEvent: (e: JobEvent) => void,
  signal?: AbortSignal,
): void {
  const url = `${BASE}/jobs/${jobId}/stream`;
  const es = new EventSource(url);
  signal?.addEventListener('abort', () => es.close());
  es.onmessage = (ev) => {
    try {
      const parsed = JSON.parse(ev.data) as JobEvent;
      onEvent(parsed);
      if (parsed.type === 'done' || parsed.type === 'error') es.close();
    } catch {}
  };
  es.onerror = () => {
    onEvent({ type: 'error', error: 'SSE connection lost' });
    es.close();
  };
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/ready`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function getAdminSettings(): Promise<AdminSettings> {
  const res = await fetch(`${BASE}/admin/settings`, { credentials: 'same-origin' });
  if (!res.ok) throw await apiError(res);
  return await res.json() as AdminSettings;
}

export async function saveAdminSettings(
  settings: {
    translation_endpoint: string;
    translation_model: string;
    translation_mode: string;
    translation_batch_segments: number;
    translation_api_key: string;
    clear_translation_api_key: boolean;
    tts_profile: string;
    mel_steps_first: number;
    mel_steps_second: number;
    mel_twopass_t_noise: number;
  },
): Promise<AdminSettings> {
  const res = await fetch(`${BASE}/admin/settings`, {
    method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
  if (!res.ok) throw await apiError(res);
  return await res.json() as AdminSettings;
}
