export interface Word {
  word: string;
  start: number;
  end: number;
  lang?: string;
}

export interface Segment {
  index: number;
  start: number;
  end: number;
  text: string;
  source_text?: string;
  words?: Word[];
  translation?: string;
  segment_id?: string;
  speaker_label?: string;
  seed?: number;
  render_nonce?: number;
}

export interface TranscribeResult {
  transcript: string;
  words: Word[];
  segments: Segment[];
  detected_language: string;
  language_counts: Record<string, number>;
  duration: number;
  word_count: number;
  segment_count: number;
  upload_path?: string;
}

export interface TranslateResult {
  translation: string;
  segments: Segment[];
  source_lang: string;
  target_lang: string;
  model: string;
  elapsed: number;
}

export interface DubResult {
  audio_path: string;
  mixed_audio_path?: string;
  duration: number;
  transcribe_job_id: string;
  segments?: Array<Record<string, unknown>>;
  debug_log?: string;
  reused_segments?: number;
  generated_segments?: number;
  mix?: {
    original_gain: number;
    dubbing_gain: number;
    ducking_strength: number;
  };
}

export interface TTSToken {
  token: string;
  dur: number;
  dur_sec: number;
  is_pause: boolean;
  allowed: boolean;
  low: boolean;
}

export interface TTSChunk {
  index: number;
  text: string;
  pred_sec: number | null;
  mel_sec: number | null;
  token_count: number;
  tokens: TTSToken[];
}

export interface TTSResult {
  audio_path: string;
  duration: number;
  chunks: TTSChunk[];
  debug_log?: string;
}

export interface JobEvent {
  type: 'progress' | 'done' | 'error';
  progress?: number;
  message?: string;
  result?: TranscribeResult | TranslateResult | DubResult | TTSResult;
  error?: string;
}

export type Step = 'idle' | 'transcribing' | 'transcribed' | 'translating' | 'translated' | 'error';

export interface User {
  id: string;
  email: string;
  display_name: string;
  created_at: number;
  data_retention_hours: number;
  is_admin: boolean;
  unlimited_usage: boolean;
  usage: Usage;
}

export interface Usage {
  plan: string;
  total_seconds: number;
  used_seconds: number;
  reserved_seconds: number;
  available_seconds: number;
  unlimited: boolean;
}

export interface Speaker { label: string; display_name?: string; id: number; }

export interface TTSModelProfile {
  key: string;
  label: string;
  description: string;
  checkpoint: string;
  default: boolean;
  active: boolean;
  loaded?: boolean;
}

export interface TTSFlowDefaults {
  mel_steps_first: number;
  mel_steps_second: number;
  mel_twopass_t_noise: number;
}

export interface DubParams {
  segments: Segment[];
  speaker_label: string;
  tts_model_profile: string;
  transcribe_job_id: string;
  reuse_dub_job_id?: string;
  target_lang: string;
  base_speed: number;
  max_adaptive_speed: number;
  extra_tail_sec: number;
  dur_scale: number;
  mel_steps_first: number;
  mel_steps_second: number;
  mel_twopass_t_noise: number;
  digital_silence: boolean;
  pause_edge_frames: number;
  short_continuity_ms: number;
  emotion_group: string;
  emotion_strength: number;
  original_gain: number;
  dubbing_gain: number;
  ducking_strength: number;
}

export interface AdminJob {
  id: string;
  kind: string;
  status: string;
  message?: string;
  error?: string;
  debug_log?: string;
  duration?: number;
  segments?: Array<{
    index?: number; start?: number; audio_duration?: number; target_budget?: number;
    speed?: number; over_budget?: number; warnings?: string[]; low_token_count?: number;
  }>;
}

export interface AdminSettings {
  translation_endpoint: string;
  translation_model: string;
  translation_mode: string;
  translation_batch_segments: number;
  translation_api_key_configured: boolean;
  translation_api_key_masked: string;
  tts_profile: string;
  tts_active_profile: string;
  tts_models: Array<{ key: string; label: string }>;
  mel_steps_first: number;
  mel_steps_second: number;
  mel_twopass_t_noise: number;
  tts_loaded_profiles: string[];
  model_ready: boolean;
  tts_ready: boolean;
  registered_users: number;
  active_sessions: number;
  data_retention_hours: number;
  recent_jobs: AdminJob[];
}

export const TRANSLATION_MODES = [
  { value: 'qwen_mtp_35b_json_overlap', label: 'Qwen3.5 35B MTP JSON (domyślny)' },
  { value: 'api_json_overlap', label: 'API JSON overlap' },
  { value: 'api_numbered', label: 'API numbered batches' },
  { value: 'wegorz_local_sentence_split', label: 'Lokalny Węgorz (EN→PL)' },
] as const;
