'use client';

import { createContext, useContext, useEffect, useMemo, useState } from 'react';

export type Locale = 'pl' | 'en';

const UI_COPY: Record<Locale, Record<string, string>> = {
  pl: {
    transcribe: 'Transkrypcja', translate: 'Tłumaczenie', dub: 'Dubbing', voice: 'Studio głosu',
    account: 'Moje konto', admin: 'Administrator', newProject: 'Nowy projekt', services: 'Usługi',
    currentProject: 'Bieżący projekt', noTranscript: 'Bez transkrypcji', navigation: 'Nawigacja',
    systemReady: 'System gotowy', systemOffline: 'System offline', checking: 'Sprawdzam',
    sourceMaterial: 'Materiał źródłowy', audioVideo: 'Audio lub wideo', file: 'Plik', movieAddress: 'Adres filmu',
    transcribeMedia: 'Transkrybuj materiał', downloadTranscribe: 'Pobierz i transkrybuj',
    sourceMissing: 'Brakuje materiału źródłowego', translationMissing: 'Brakuje tłumaczenia',
    startTranscription: 'Rozpocznij od transkrypcji.', transcriptReady: 'Transkrypcja jest gotowa.',
    addMaterial: 'Dodaj materiał', goTranslation: 'Przejdź do tłumaczenia',
    materialSource: 'Źródło', targetLanguage: 'Język docelowy', translateAction: 'Tłumacz', translating: 'Tłumaczę…',
    readyTranslation: 'Materiał gotowy do tłumaczenia', goDubbing: 'Przejdź do dubbingu', copy: 'Kopiuj', downloadSrt: 'Pobierz SRT',
    preview: 'Podgląd', original: 'Oryginał', mix: 'Miks', voiceOnly: 'Sam dubbing', dubbingSegments: 'Segmenty dubbingu',
    speaker: 'Lektor', addVoice: 'Dodaj próbkę głosu', tempo: 'Tempo', originalBackground: 'Oryginalna ścieżka pozostaje w tle',
    voiceStudio: 'Studio głosu', singleSpeech: 'Synteza pojedynczej wypowiedzi', synthesisText: 'Tekst do syntezy',
    language: 'Język', generateSpeech: 'Generuj mowę', synthesizing: 'Syntetyzuję…', downloadWav: 'Pobierz WAV',
    dropFile: 'Upuść plik lub wybierz z dysku', profileSession: 'Dane profilu i sesja', name: 'Nazwa',
    accountSince: 'Konto od', logout: 'Wyloguj się', filesPrivacy: 'Pliki i prywatność', dataControl: 'Kontrola nad materiałami',
    deleteFiles: 'Usuń teraz wszystkie moje pliki',
  },
  en: {
    transcribe: 'Transcription', translate: 'Translation', dub: 'Dubbing', voice: 'Voice studio',
    account: 'My account', admin: 'Administrator', newProject: 'New project', services: 'Services',
    currentProject: 'Current project', noTranscript: 'No transcript', navigation: 'Navigation',
    systemReady: 'System ready', systemOffline: 'System offline', checking: 'Checking',
    sourceMaterial: 'Source material', audioVideo: 'Audio or video', file: 'File', movieAddress: 'Video URL',
    transcribeMedia: 'Transcribe media', downloadTranscribe: 'Download and transcribe',
    sourceMissing: 'Source material required', translationMissing: 'Translation required',
    startTranscription: 'Start with transcription.', transcriptReady: 'The transcript is ready.',
    addMaterial: 'Add material', goTranslation: 'Go to translation',
    materialSource: 'Source', targetLanguage: 'Target language', translateAction: 'Translate', translating: 'Translating…',
    readyTranslation: 'Material ready for translation', goDubbing: 'Continue to dubbing', copy: 'Copy', downloadSrt: 'Download SRT',
    preview: 'Preview', original: 'Original', mix: 'Mix', voiceOnly: 'Dubbing only', dubbingSegments: 'Dubbing segments',
    speaker: 'Voice', addVoice: 'Add voice sample', tempo: 'Pace', originalBackground: 'The original soundtrack stays in the background',
    voiceStudio: 'Voice studio', singleSpeech: 'Single utterance synthesis', synthesisText: 'Text to synthesize',
    language: 'Language', generateSpeech: 'Generate speech', synthesizing: 'Synthesizing…', downloadWav: 'Download WAV',
    dropFile: 'Drop a file or choose from disk', profileSession: 'Profile details and session', name: 'Name',
    accountSince: 'Account created', logout: 'Log out', filesPrivacy: 'Files and privacy', dataControl: 'Control your media',
    deleteFiles: 'Delete all my files now',
  },
};

interface LocaleValue { locale: Locale; setLocale: (locale: Locale) => void; t: (key: string) => string; }
const LocaleContext = createContext<LocaleValue>({ locale: 'pl', setLocale: () => {}, t: key => key });

export function LocaleProvider({ initialLocale, children }: { initialLocale: Locale; children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);
  useEffect(() => {
    const query = new URLSearchParams(window.location.search).get('lang');
    const stored = localStorage.getItem('nupicai_locale');
    const routeLocale: Locale | null = window.location.pathname === '/en' || window.location.pathname.startsWith('/en/') ? 'en' : null;
    const detected = navigator.language.toLowerCase().startsWith('pl') ? 'pl' : 'en';
    const selected = query === 'pl' || query === 'en' ? query : stored === 'pl' || stored === 'en' ? stored : routeLocale ?? detected;
    setLocaleState(selected);
  }, []);
  useEffect(() => { document.documentElement.lang = locale; }, [locale]);
  const setLocale = (next: Locale) => { localStorage.setItem('nupicai_locale', next); setLocaleState(next); };
  const value = useMemo(() => ({ locale, setLocale, t: (key: string) => UI_COPY[locale][key] ?? key }), [locale]);
  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

export function useLocale(): LocaleValue { return useContext(LocaleContext); }

export function LanguageSwitch({ compact = false }: { compact?: boolean }) {
  const { locale, setLocale } = useLocale();
  return <div className={`language-switch ${compact ? 'compact' : ''}`} aria-label="Language">
    <button className={locale === 'pl' ? 'active' : ''} onClick={() => setLocale('pl')} aria-pressed={locale === 'pl'}>PL</button>
    <button className={locale === 'en' ? 'active' : ''} onClick={() => setLocale('en')} aria-pressed={locale === 'en'}>EN</button>
  </div>;
}
