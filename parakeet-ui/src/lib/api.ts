import type { AdminSettings, JobEvent, Speaker, DubParams, TTSModelProfile } from './types';

const BASE = '';

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

export async function listTTSModels(): Promise<{ default: string; active: string; models: TTSModelProfile[] }> {
  const res = await fetch(`${BASE}/tts_models`);
  if (!res.ok) return { default: '', active: '', models: [] };
  const data = await res.json();
  return {
    default: String(data.default ?? ''),
    active: String(data.active ?? ''),
    models: (data.models ?? []) as TTSModelProfile[],
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
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  const data = await res.json();
  return data.job_id as string;
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
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function getAdminSettings(token: string): Promise<AdminSettings> {
  const res = await fetch(`${BASE}/admin/settings`, {
    headers: { 'X-Admin-Token': token },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return await res.json() as AdminSettings;
}

export async function saveAdminSettings(
  token: string,
  settings: {
    translation_endpoint: string;
    translation_model: string;
    translation_mode: string;
    translation_batch_segments: number;
    translation_api_key: string;
    clear_translation_api_key: boolean;
  },
): Promise<AdminSettings> {
  const res = await fetch(`${BASE}/admin/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Admin-Token': token },
    body: JSON.stringify(settings),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return await res.json() as AdminSettings;
}
