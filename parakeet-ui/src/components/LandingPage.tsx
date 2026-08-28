'use client';

import { useState } from 'react';
import {
  ArrowRight, AudioLines, Check, ChevronDown, FileText, Languages,
  LockKeyhole, Mic2, Play, ShieldCheck, Sparkles, Users, Video, X,
} from 'lucide-react';
import type { User } from '@/lib/types';
import { loginAccount, registerAccount } from '@/lib/api';
import { LanguageSwitch, type Locale, useLocale } from '@/lib/locale';

const COPY = {
  pl: {
    nav: ['Jak działa', 'Dla kogo', 'Cennik', 'FAQ'], login: 'Zaloguj się', register: 'Załóż konto',
    kicker: 'Neural Unified Platform for Intelligent Communication', lead: 'Transkrypcja, tłumaczenie i naturalny dubbing w jednym miejscu.',
    support: 'Rozpoznaj materiał w jednym z 25 języków europejskich i przygotuj polską lub angielską wersję z zachowanym rytmem wypowiedzi oraz oryginalnym tłem.',
    start: 'Rozpocznij projekt', see: 'Zobacz jak działa', proof: ['25 języków wejściowych', 'Dubbing po polsku i angielsku', 'Eksport WAV i MP4'],
    capabilities: [['Wielojęzyczna transkrypcja', '25 języków, TXT, SRT i VTT'], ['Tłumaczenie kontekstowe', 'Z dowolnego obsługiwanego języka'], ['Naturalne głosy PL i EN', 'Tempo dopasowane do filmu'], ['Gotowy materiał', 'Miks audio i eksport MP4']],
    workflowLabel: 'Jeden przepływ pracy', workflowTitle: 'Od materiału źródłowego do gotowego dubbingu', workflowIntro: 'Każdy etap możesz sprawdzić i poprawić przed uruchomieniem następnego.',
    workflow: [['Dodaj materiał', 'Prześlij audio, wideo albo wklej adres filmu z YouTube.'], ['Sprawdź tekst', 'Parakeet tworzy transkrypcję z czasami, którą możesz edytować.'], ['Przetłumacz', 'Wybierz język polski lub angielski. Segmenty pozostają powiązane ze źródłem.'], ['Wybierz głos i eksportuj', 'WęgorzAI generuje dubbing, a mikser łączy go z oryginalnym tłem.']],
    audienceLabel: 'Dla kogo', audienceTitle: 'Jedno nagranie, nowa publiczność', audienceIntro: 'NupicAI rozpoznaje 25 języków europejskich, tłumaczy treść i przygotowuje polski lub angielski dubbing bez ręcznego przenoszenia plików między aplikacjami.',
    audiences: ['Twórcy wideo i kanały internetowe', 'Firmy i zespoły marketingowe', 'Podcasty i wywiady', 'Edukacja i szkolenia'],
    privacyLabel: 'Twoje dane', privacyTitle: 'Pliki robocze nie zostają na serwerze bezterminowo',
    privacy: [['Osobna przestrzeń konta', 'Zadania i pliki każdego użytkownika są odseparowane i dostępne tylko po zalogowaniu.'], ['Automatyczne usuwanie', 'Materiały źródłowe, dubbing i pliki robocze są automatycznie usuwane po 24 godzinach.'], ['Jasne zasady przetwarzania', 'Audio jest przetwarzane lokalnie. Przy tłumaczeniu tekst może zostać wysłany do skonfigurowanego API językowego.']],
    pricingLabel: 'Planowany cennik', pricingTitle: 'Płać za minuty gotowego materiału', pricingIntro: 'W okresie pilotażowym konta są bezpłatne. Poniższe pakiety pokazują planowany model rozliczeń.',
    plans: [
      { name: 'Bezpłatny', price: '0 zł', note: '5 minut na start', features: ['Pełny workflow', 'Eksport napisów', 'WAV i MP4'] },
      { name: 'Creator', price: '39 zł', note: '5 godzin miesięcznie', features: ['Wszystkie głosy', 'Edycja segmentów', 'Miks z oryginałem'] },
      { name: 'Studio', price: '99 zł', note: '15 godzin miesięcznie', features: ['Transkrypcja i tłumaczenie', 'Dubbing PL i EN', 'Eksport do publikacji'] },
    ],
    popular: 'Najczęściej wybierany', accountAction: 'Załóż konto', faqTitle: 'Najczęściej zadawane pytania',
    faqs: [
      ['Czy mogę publikować i sprzedawać wygenerowane audio?', 'Tak. Wygenerowany materiał możesz publikować i wykorzystywać komercyjnie bez wymaganej atrybucji NupicAI.'],
      ['Jak długo przechowujecie moje pliki?', 'Pliki źródłowe i wyniki są automatycznie usuwane po 24 godzinach. Możesz też usunąć je natychmiast z poziomu swojego konta.'],
      ['Czy mogę tłumaczyć także na angielski?', 'Tak. W studio wybierasz polski albo angielski jako język docelowy, a później generujesz dubbing w tym samym języku.'],
      ['Jakie języki rozpoznaje NupicAI?', 'Transkrypcja obsługuje 25 języków europejskich, między innymi polski, angielski, niemiecki, francuski, hiszpański, włoski, ukraiński i rosyjski. Gotowy dubbing jest obecnie generowany po polsku lub angielsku.'],
      ['Czy mogę poprawić transkrypcję i tłumaczenie?', 'Tak. Przed dubbingiem możesz edytować każdy segment i zachować jego położenie na osi czasu.'],
      ['Czy system zachowuje muzykę i dźwięki z filmu?', 'Tak. Mikser może zachować oryginalne tło oraz automatycznie ściszać je podczas wypowiedzi lektora.'],
      ['Czy NupicAI jest nieomylne?', 'Nie. Rzadkie nazwiska, słaba jakość nagrania i nietypowa wymowa mogą wymagać ręcznej korekty tekstu lub segmentu.'],
    ],
    ctaTitle: 'Twój materiał może mówić po polsku lub angielsku.', ctaText: 'Załóż konto i przygotuj pierwszy projekt w jednym studio.', cta: 'Rozpocznij bezpłatnie', footer: 'Transkrypcja, tłumaczenie i dubbing AI.', privacyLink: 'Prywatność',
    altHero: 'Studio dubbingowe NupicAI podczas pracy nad materiałem wideo', altWorkflow: 'Proces transkrypcji, tłumaczenia i dubbingu materiału w NupicAI', altAudience: 'Zespół pracujący nad lokalizacją materiału wideo',
  },
  en: {
    nav: ['How it works', 'Who it is for', 'Pricing', 'FAQ'], login: 'Log in', register: 'Create account',
    kicker: 'Neural Unified Platform for Intelligent Communication', lead: 'Transcription, translation and natural dubbing in one workspace.',
    support: 'Transcribe media in 25 European languages and create a polished Polish or English version while preserving pacing and the original background.',
    start: 'Start a project', see: 'See how it works', proof: ['25 input languages', 'Polish and English dubbing', 'WAV and MP4 export'],
    capabilities: [['Multilingual transcription', '25 languages, TXT, SRT and VTT'], ['Context-aware translation', 'From any supported language'], ['Natural Polish and English voices', 'Pacing matched to video'], ['Publish-ready media', 'Audio mix and MP4 export']],
    workflowLabel: 'One production flow', workflowTitle: 'From source media to finished dubbing', workflowIntro: 'Review and edit every stage before starting the next one.',
    workflow: [['Add your media', 'Upload audio or video, or paste a YouTube link.'], ['Review the transcript', 'Parakeet creates an editable time-aligned transcript.'], ['Translate', 'Choose Polish or English. Every segment stays linked to the source.'], ['Choose a voice and export', 'WęgorzAI generates the voice while the mixer preserves the original background.']],
    audienceLabel: 'Built for', audienceTitle: 'One recording, a new audience', audienceIntro: 'NupicAI recognizes 25 European languages, translates the content and creates Polish or English dubbing without moving files between several apps.',
    audiences: ['Video creators and online channels', 'Companies and marketing teams', 'Podcasts and interviews', 'Education and training'],
    privacyLabel: 'Your data', privacyTitle: 'Working files do not remain on the server indefinitely',
    privacy: [['Private account workspace', 'Every user’s jobs and files are isolated and only available after authentication.'], ['Automatic deletion', 'Source media, dubbing and working files are automatically removed after 24 hours.'], ['Transparent processing', 'Audio is processed locally. Translation text may be sent to the configured language API.']],
    pricingLabel: 'Planned pricing', pricingTitle: 'Pay for minutes of finished media', pricingIntro: 'Accounts are free during the pilot. These packages show the planned billing model.',
    plans: [
      { name: 'Free', price: '€0', note: '5 minutes to get started', features: ['Complete workflow', 'Subtitle export', 'WAV and MP4'] },
      { name: 'Creator', price: '€9', note: '5 hours per month', features: ['All voices', 'Segment editing', 'Original audio mix'] },
      { name: 'Studio', price: '€23', note: '15 hours per month', features: ['Transcription and translation', 'Polish and English dubbing', 'Publish-ready export'] },
    ],
    popular: 'Most popular', accountAction: 'Create account', faqTitle: 'Frequently asked questions',
    faqs: [
      ['Can I publish and sell the generated audio?', 'Yes. You can publish and use generated material commercially without mandatory NupicAI attribution.'],
      ['How long do you keep my files?', 'Source files and results are automatically deleted after 24 hours. You can also remove them immediately from your account.'],
      ['Can I translate into English as well?', 'Yes. Choose Polish or English as the target language, then generate dubbing in the same language.'],
      ['Which languages can NupicAI recognize?', 'Transcription supports 25 European languages, including Polish, English, German, French, Spanish, Italian, Ukrainian and Russian. Finished dubbing is currently generated in Polish or English.'],
      ['Can I edit the transcript and translation?', 'Yes. You can edit every segment before dubbing while retaining its place on the timeline.'],
      ['Does the system preserve music and sound effects?', 'Yes. The mixer can retain the original background and automatically duck it while the new voice is speaking.'],
      ['Is NupicAI always perfect?', 'No. Rare names, low-quality recordings and unusual pronunciation may require a manual text or segment correction.'],
    ],
    ctaTitle: 'Your media can speak Polish or English.', ctaText: 'Create an account and prepare your first project in one studio.', cta: 'Start for free', footer: 'AI transcription, translation and dubbing.', privacyLink: 'Privacy',
    altHero: 'NupicAI dubbing studio processing a video project', altWorkflow: 'NupicAI workflow for transcription, translation and dubbing', altAudience: 'Team reviewing localized video content',
  },
};

export default function LandingPage({ onAuthenticated }: { onAuthenticated: (user: User) => void }) {
  const { locale } = useLocale();
  const c = COPY[locale];
  const [authMode, setAuthMode] = useState<'login' | 'register' | null>(null);

  const openStudio = () => setAuthMode('register');
  return <div className="marketing-page">
    <header className="marketing-nav">
      <a className="marketing-brand" href="#top" aria-label={locale === 'pl' ? 'NupicAI - strona główna' : 'NupicAI - home'}>
        <img src="/brand/logo.png" alt="NupicAI" width={2172} height={724} />
      </a>
      <nav aria-label={locale === 'pl' ? 'Nawigacja strony' : 'Page navigation'}>
        <a href="#jak-dziala">{c.nav[0]}</a>
        <a href="#dla-kogo">{c.nav[1]}</a>
        <a href="#cennik">{c.nav[2]}</a>
        <a href="#faq">{c.nav[3]}</a>
      </nav>
      <div className="marketing-nav-actions">
        <LanguageSwitch compact />
        <button className="button button-ghost" onClick={() => setAuthMode('login')}>{c.login}</button>
        <button className="button button-primary" onClick={openStudio}>{c.register}</button>
      </div>
    </header>

    <main>
      <section className="marketing-hero" id="top">
        <img className="marketing-hero-media" src="/marketing/hero-nupicai-dubbing-studio.webp" alt={c.altHero} width={1672} height={941} fetchPriority="high" />
        <div className="marketing-hero-shade" />
        <div className="marketing-hero-content">
          <span className="marketing-kicker"><Sparkles size={15} /> {c.kicker}</span>
          <h1>NupicAI</h1>
          <p className="hero-lead">{c.lead}</p>
          <p className="hero-support">{c.support}</p>
          <div className="hero-actions">
            <button className="button button-primary button-large" onClick={openStudio}>{c.start} <ArrowRight size={17} /></button>
            <a className="button button-glass button-large" href="#jak-dziala"><Play size={16} /> {c.see}</a>
          </div>
          <div className="hero-proof">
            {c.proof.map(item => <span key={item}><Check size={14} /> {item}</span>)}
          </div>
        </div>
      </section>

      <section className="proof-band" aria-label={locale === 'pl' ? 'Możliwości NupicAI' : 'NupicAI capabilities'}>
        {[FileText, Languages, Mic2, Video].map((Icon, index) => <div key={c.capabilities[index][0]}><Icon size={21} /><strong>{c.capabilities[index][0]}</strong><span>{c.capabilities[index][1]}</span></div>)}
      </section>

      <section className="marketing-section workflow-section" id="jak-dziala">
        <div className="marketing-section-heading">
          <span className="section-label">{c.workflowLabel}</span>
          <h2>{c.workflowTitle}</h2>
          <p>{c.workflowIntro}</p>
        </div>
        <div className="workflow-layout">
          <img src="/marketing/workflow-audio-to-dubbing.webp" alt={c.altWorkflow} width={1672} height={941} loading="lazy" />
          <ol className="workflow-steps">
            {c.workflow.map((item, index) => <li key={item[0]}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{item[0]}</strong><p>{item[1]}</p></div></li>)}
          </ol>
        </div>
      </section>

      <section className="marketing-section audience-section" id="dla-kogo">
        <div className="audience-copy">
          <span className="section-label">{c.audienceLabel}</span>
          <h2>{c.audienceTitle}</h2>
          <p>{c.audienceIntro}</p>
          <div className="audience-list">
            {[Video, Users, AudioLines, FileText].map((Icon, index) => <span key={c.audiences[index]}><Icon size={17} /> {c.audiences[index]}</span>)}
          </div>
        </div>
        <img src="/marketing/creators-localization.webp" alt={c.altAudience} width={1672} height={941} loading="lazy" />
      </section>

      <section className="privacy-band" id="prywatnosc">
        <div className="privacy-heading"><ShieldCheck size={28} /><div><span className="section-label">{c.privacyLabel}</span><h2>{c.privacyTitle}</h2></div></div>
        <div className="privacy-grid">
          {[LockKeyhole, ShieldCheck, FileText].map((Icon, index) => <div key={c.privacy[index][0]}><Icon size={20} /><strong>{c.privacy[index][0]}</strong><p>{c.privacy[index][1]}</p></div>)}
        </div>
      </section>

      <section className="marketing-section pricing-section" id="cennik">
        <div className="marketing-section-heading">
          <span className="section-label">{c.pricingLabel}</span>
          <h2>{c.pricingTitle}</h2>
          <p>{c.pricingIntro}</p>
        </div>
        <div className="pricing-grid">
          {c.plans.map((plan, index) => <PriceCard key={plan.name} {...plan} popular={c.popular} accountAction={c.accountAction} featured={index === 1} action={() => setAuthMode('register')} />)}
        </div>
      </section>

      <section className="marketing-section faq-section" id="faq">
        <div className="marketing-section-heading"><span className="section-label">FAQ</span><h2>{c.faqTitle}</h2></div>
        <div className="faq-list">
          {c.faqs.map(item => <Faq key={item[0]} question={item[0]}>{item[1]}</Faq>)}
        </div>
      </section>

      <section className="marketing-cta">
        <img src="/brand/mark.png" alt="" width={1254} height={1254} loading="lazy" />
        <div><h2>{c.ctaTitle}</h2><p>{c.ctaText}</p></div>
        <button className="button button-primary button-large" onClick={openStudio}>{c.cta} <ArrowRight size={17} /></button>
      </section>
    </main>

    <footer className="marketing-footer">
      <img src="/brand/logo.png" alt="NupicAI" width={2172} height={724} loading="lazy" />
      <p>{c.footer}</p>
      <div><a href="#prywatnosc">{c.privacyLink}</a><a href="#faq">FAQ</a><button onClick={() => setAuthMode('login')}>{c.login}</button></div>
    </footer>

    {authMode && <AuthDialog locale={locale} mode={authMode} onMode={setAuthMode} onClose={() => setAuthMode(null)} onAuthenticated={onAuthenticated} />}
  </div>;
}

function PriceCard({ name, price, note, features, featured, popular, accountAction, action }: { name: string; price: string; note: string; features: readonly string[]; featured?: boolean; popular: string; accountAction: string; action: () => void }) {
  return <article className={`price-card ${featured ? 'featured' : ''}`}>
    {featured && <span className="price-badge">{popular}</span>}
    <h3>{name}</h3><strong className="price-value">{price}</strong><p>{note}</p>
    <ul>{features.map(item => <li key={item}><Check size={15} />{item}</li>)}</ul>
    <button className={`button ${featured ? 'button-primary' : 'button-secondary'}`} onClick={action}>{accountAction}</button>
  </article>;
}

function Faq({ question, children }: { question: string; children: React.ReactNode }) {
  return <details><summary>{question}<ChevronDown size={17} /></summary><p>{children}</p></details>;
}

function AuthDialog({ locale, mode, onMode, onClose, onAuthenticated }: {
  locale: Locale; mode: 'login' | 'register'; onMode: (mode: 'login' | 'register') => void;
  onClose: () => void; onAuthenticated: (user: User) => void;
}) {
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [terms, setTerms] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const en = locale === 'en';

  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const user = mode === 'login'
        ? await loginAccount(email, password)
        : await registerAccount({ email, display_name: name, password, terms_accepted: terms });
      onAuthenticated(user);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(false); }
  };

  return <div className="auth-backdrop" role="presentation" onMouseDown={e => e.target === e.currentTarget && onClose()}>
    <section className="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <button className="icon-button auth-close" title={en ? 'Close' : 'Zamknij'} onClick={onClose}><X size={18} /></button>
      <img src="/brand/mark.png" alt="" className="auth-mark" width={1254} height={1254} />
      <h2 id="auth-title">{mode === 'login' ? (en ? 'Welcome back' : 'Witaj ponownie') : (en ? 'Create a NupicAI account' : 'Załóż konto NupicAI')}</h2>
      <p>{mode === 'login' ? (en ? 'Log in to open your workspace.' : 'Zaloguj się, aby przejść do swoich narzędzi.') : (en ? 'Your first project is free during the pilot.' : 'Pierwszy projekt uruchomisz bezpłatnie.')}</p>
      <div className="auth-tabs"><button className={mode === 'login' ? 'active' : ''} onClick={() => onMode('login')}>{en ? 'Log in' : 'Logowanie'}</button><button className={mode === 'register' ? 'active' : ''} onClick={() => onMode('register')}>{en ? 'Register' : 'Rejestracja'}</button></div>
      <form onSubmit={submit}>
        {mode === 'register' && <label><span className="field-label">{en ? 'Name or company' : 'Imię lub nazwa'}</span><input autoComplete="name" value={name} onChange={e => setName(e.target.value)} required minLength={2} /></label>}
        <label><span className="field-label">E-mail</span><input type="email" autoComplete="email" value={email} onChange={e => setEmail(e.target.value)} required /></label>
        <label><span className="field-label">{en ? 'Password' : 'Hasło'}</span><input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} onChange={e => setPassword(e.target.value)} required minLength={10} /></label>
        {mode === 'register' && <label className="auth-consent"><input type="checkbox" checked={terms} onChange={e => setTerms(e.target.checked)} required /><span>{en ? 'I accept the terms and privacy rules, including automatic file deletion after 24 hours.' : 'Akceptuję regulamin i zasady prywatności, w tym automatyczne usuwanie plików po 24 godzinach.'}</span></label>}
        {error && <div className="notice notice-error">{error}</div>}
        <button className="button button-primary button-large" disabled={busy}>{busy ? (en ? 'Please wait…' : 'Proszę czekać…') : mode === 'login' ? (en ? 'Log in' : 'Zaloguj się') : (en ? 'Create account' : 'Utwórz konto')}</button>
      </form>
      <small>{en ? 'Your password is stored only as a secure hash.' : 'Hasło jest przechowywane wyłącznie w postaci bezpiecznego skrótu.'}</small>
    </section>
  </div>;
}
