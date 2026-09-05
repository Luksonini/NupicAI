import Link from 'next/link';

type Locale = 'pl' | 'en';
type DocumentKind = 'terms' | 'privacy';

const VERSION = '29 sierpnia 2026';
const VERSION_EN = '29 August 2026';

const termsPl = [
  ['1. Usługodawca', <>Usługę NupicAI świadczy Tomasz Gasior IT, ul. Spokojna 2, 20-074 Lublin, NIP 9182117616. Kontakt: <a href="mailto:gasior1510@gmail.com">gasior1510@gmail.com</a>.</>],
  ['2. Zakres usługi', <>NupicAI udostępnia konto oraz narzędzia do transkrypcji audio i wideo, tłumaczenia tekstu, syntezy mowy i tworzenia dubbingu. W obecnym pilocie usługa jest bezpłatna w granicach limitu widocznego na koncie. Cennik na stronie ma charakter zapowiedzi i nie umożliwia jeszcze zakupu.</>],
  ['3. Wymagania techniczne', <>Potrzebne są aktualna przeglądarka z obsługą JavaScript, połączenie z Internetem oraz aktywny adres e-mail. Obsługiwane formaty i maksymalny rozmiar pliku wynikają z komunikatów formularza i serwera. Przerwanie połączenia lub zamknięcie przeglądarki może przerwać odbiór postępu zadania.</>],
  ['4. Konto i zawarcie umowy', <>Umowa o świadczenie usług drogą elektroniczną zostaje zawarta po rejestracji i akceptacji niniejszego regulaminu oraz polityki prywatności. Użytkownik podaje prawdziwy adres e-mail, chroni hasło i odpowiada za działania wykonane w swojej sesji. Konto można zakończyć w każdej chwili w sekcji „Moje konto”.</>],
  ['5. Materiały i dozwolone użycie', <>Użytkownik zachowuje prawa do swoich materiałów i wyników, ale musi mieć prawa, licencję lub inną podstawę do przesłania i przetworzenia treści oraz głosu. Zabronione są treści bezprawne, naruszające prawa osób trzecich, oszustwa, podszywanie się, nieuprawnione klonowanie głosu i użycie wyników do wprowadzania odbiorców w błąd.</>],
  ['6. Treści generowane przez AI', <>Transkrypcje, tłumaczenia i nagrania są tworzone automatycznie i mogą zawierać błędy, pominięcia lub inną wymowę nazw własnych. Użytkownik powinien sprawdzić wynik przed publikacją. Pliki syntezy i dubbingu są treściami wygenerowanymi lub zmodyfikowanymi przez AI; użytkownik odpowiada za wymagane prawem oznaczenie ich przy dalszym udostępnianiu.</>],
  ['7. Limity, dostępność i pliki', <>Limit jest rozliczany za ukończone audio. Nieudane zadanie zwalnia rezerwację limitu. Pliki źródłowe, prompty głosowe, wyniki i pliki robocze są standardowo usuwane automatycznie po 24 godzinach i mogą zostać usunięte wcześniej z konta. Usługa pilotażowa może mieć przerwy techniczne; nie gwarantujemy nieprzerwanej dostępności.</>],
  ['8. Reklamacje', <>Reklamację można wysłać na <a href="mailto:gasior1510@gmail.com">gasior1510@gmail.com</a>, podając adres konta, opis problemu i identyfikator zadania, jeśli jest dostępny. Odpowiemy w terminie do 14 dni. Niniejszy regulamin nie ogranicza praw konsumenta wynikających z bezwzględnie obowiązujących przepisów.</>],
  ['9. Zmiany regulaminu', <>Aktualna wersja jest dostępna stale pod tym adresem. Istotne zmiany dotyczące istniejących kont będą komunikowane przed ich wejściem w życie. Dalsze korzystanie może wymagać zaakceptowania nowej wersji.</>],
];

const privacyPl = [
  ['1. Administrator danych', <>Administratorem danych jest Tomasz Gasior IT, ul. Spokojna 2, 20-074 Lublin, NIP 9182117616. W sprawach prywatności napisz na <a href="mailto:gasior1510@gmail.com">gasior1510@gmail.com</a>.</>],
  ['2. Jakie dane przetwarzamy', <>Przetwarzamy dane konta (adres e-mail, nazwa, bezpieczny skrót hasła, sesje i wykorzystanie limitu), przesłane pliki audio lub wideo, tekst, ustawienia projektu, prompty głosowe, wyniki oraz podstawowe dane techniczne niezbędne do bezpieczeństwa i działania usługi.</>],
  ['3. Cele i podstawy prawne', <>Dane przetwarzamy, aby utworzyć i obsługiwać konto, wykonać zamówione operacje, rozliczyć limit, zapewnić bezpieczeństwo i obsłużyć zgłoszenia. Podstawą jest wykonanie umowy o usługę elektroniczną, obowiązki prawne administratora oraz prawnie uzasadniony interes polegający na ochronie usługi i dochodzeniu roszczeń. Nie używamy przesłanych materiałów do trenowania modeli bez osobnej, dobrowolnej zgody.</>],
  ['4. Gdzie trafiają dane', <>Transkrypcja i synteza mowy są wykonywane lokalnie na infrastrukturze administratora. Gdy wybrany jest zdalny tryb tłumaczenia, tekst segmentów jest przesyłany do skonfigurowanego serwera tłumaczeniowego. Dostawcy hostingu i infrastruktury mogą przetwarzać dane wyłącznie w zakresie potrzebnym do świadczenia usługi. Nie sprzedajemy danych użytkowników.</>],
  ['5. Okres przechowywania', <>Pliki źródłowe, prompty głosowe, wyniki i pliki robocze są standardowo usuwane po 24 godzinach. Dane konta, saldo i historia rozliczenia limitu są przechowywane do usunięcia konta, a później tylko wtedy i tak długo, jak wymagają tego przepisy lub obrona przed roszczeniami. Sesje wygasają standardowo po 30 dniach.</>],
  ['6. Twoje prawa', <>Możesz żądać dostępu, sprostowania, usunięcia, ograniczenia przetwarzania i przeniesienia danych oraz wnieść sprzeciw, gdy podstawą jest prawnie uzasadniony interes. Masz też prawo złożyć skargę do Prezesa Urzędu Ochrony Danych Osobowych. Pliki możesz usunąć od razu, a konto zamknąć po potwierdzeniu hasłem.</>],
  ['7. Cookies i pamięć przeglądarki', <>Używamy wyłącznie niezbędnego ciasteczka sesyjnego HttpOnly. Wybór języka może być zapisany lokalnie w przeglądarce. Obecnie nie używamy reklamowych ani analitycznych plików cookie, dlatego nie wyświetlamy zgody marketingowej.</>],
  ['8. Automatyczne przetwarzanie i AI', <>Modele automatycznie przetwarzają treść w celu zwrócenia transkrypcji, tłumaczenia lub audio. Nie podejmują wobec użytkownika decyzji wywołujących skutki prawne. Wyniki audio są syntetyczne lub zmodyfikowane przez AI i powinny być ocenione przed publikacją.</>],
  ['9. Bezpieczeństwo i zmiany', <>Stosujemy odseparowane katalogi użytkowników, szyfrowane połączenie na produkcji, hasła przechowywane jako skróty oraz ograniczony czas retencji. Aktualna wersja polityki jest dostępna pod tym adresem; o istotnych zmianach dotyczących kont poinformujemy przed ich wejściem w życie.</>],
];

const termsEn = [
  ['1. Service provider', <>NupicAI is provided by Tomasz Gasior IT, ul. Spokojna 2, 20-074 Lublin, Poland, VAT ID PL9182117616. Contact: <a href="mailto:gasior1510@gmail.com">gasior1510@gmail.com</a>.</>],
  ['2. Service scope', <>NupicAI provides an account and tools for audio/video transcription, text translation, speech synthesis and dubbing. The current pilot is free within the allowance shown in the account. Pricing displayed on the website is a preview and purchases are not yet available.</>],
  ['3. Technical requirements', <>An up-to-date JavaScript-enabled browser, an Internet connection and an active email address are required. Supported formats and upload limits are communicated by the form and server. A connection interruption or closing the browser may interrupt progress updates.</>],
  ['4. Account and agreement', <>The electronic-services agreement is concluded when you register and accept these terms and the privacy policy. You must provide a valid email, protect your password and remain responsible for activity in your session. You can terminate the agreement at any time by deleting the account.</>],
  ['5. Content and permitted use', <>You retain rights to your materials and outputs, but must have the rights, licence or another legal basis to upload and process content and voices. Illegal content, infringement, fraud, impersonation, unauthorised voice cloning and deceptive use of outputs are prohibited.</>],
  ['6. AI-generated content', <>Transcripts, translations and recordings are generated automatically and may contain errors, omissions or altered pronunciation. Review every result before publication. Synthesised and dubbed files are AI-generated or AI-modified content; you are responsible for disclosures required when distributing them.</>],
  ['7. Allowance, availability and files', <>Completed audio consumes the allowance; failed jobs release reservations. Source files, voice prompts, outputs and working files are normally deleted automatically after 24 hours and can be deleted sooner. The pilot may have maintenance interruptions and uninterrupted availability is not guaranteed.</>],
  ['8. Complaints', <>Send complaints to <a href="mailto:gasior1510@gmail.com">gasior1510@gmail.com</a> with your account email, a problem description and job ID where available. We respond within 14 days. These terms do not limit mandatory consumer rights.</>],
  ['9. Changes', <>The current terms remain available at this URL. Material changes affecting existing accounts will be communicated before taking effect and may require acceptance of the updated version.</>],
];

const privacyEn = [
  ['1. Controller', <>The data controller is Tomasz Gasior IT, ul. Spokojna 2, 20-074 Lublin, Poland, VAT ID PL9182117616. Privacy contact: <a href="mailto:gasior1510@gmail.com">gasior1510@gmail.com</a>.</>],
  ['2. Data processed', <>We process account data (email, name, secure password hash, sessions and allowance usage), uploaded audio/video, text, project settings, voice prompts, results and basic technical data needed to secure and operate the service.</>],
  ['3. Purposes and legal bases', <>Data is processed to maintain the account, perform requested operations, account for usage, secure the service and handle support. The legal bases are performance of the electronic-services agreement, legal obligations and legitimate interests in service security and legal claims. Uploaded materials are not used to train models without separate voluntary consent.</>],
  ['4. Recipients', <>Transcription and speech synthesis run locally on the controller’s infrastructure. In remote translation mode, segment text is sent to the configured translation server. Hosting and infrastructure providers may process data only as required to provide the service. We do not sell user data.</>],
  ['5. Retention', <>Source files, voice prompts, outputs and working files are normally deleted after 24 hours. Account, balance and allowance-ledger data is retained until account deletion and afterwards only where and for as long as law or legal claims require. Sessions normally expire after 30 days.</>],
  ['6. Your rights', <>You may request access, rectification, erasure, restriction and portability, and object where processing relies on legitimate interests. You may lodge a complaint with the Polish Personal Data Protection Office or your local EEA supervisory authority. Files can be removed immediately and the account can be deleted after password confirmation.</>],
  ['7. Cookies and local storage', <>We use only an essential HttpOnly session cookie. Your language choice may be stored locally in the browser. We currently use no advertising or analytics cookies, so no marketing-cookie consent is displayed.</>],
  ['8. Automated processing and AI', <>Models process content automatically to return transcription, translation or audio. They do not make decisions about users that produce legal effects. Audio outputs are synthetic or AI-modified and should be reviewed before publication.</>],
  ['9. Security and changes', <>We use isolated user directories, encrypted transport in production, password hashing and limited file retention. The current policy is available at this URL; material changes affecting accounts will be communicated before taking effect.</>],
];

export default function LegalPage({ locale, kind }: { locale: Locale; kind: DocumentKind }) {
  const en = locale === 'en';
  const sections = kind === 'terms' ? (en ? termsEn : termsPl) : (en ? privacyEn : privacyPl);
  const title = kind === 'terms' ? (en ? 'Terms of service' : 'Regulamin NupicAI') : (en ? 'Privacy policy' : 'Polityka prywatności');
  return <main className="legal-page">
    <header className="legal-header">
      <Link href={en ? '/en' : '/'} aria-label={en ? 'Back to NupicAI' : 'Wróć do NupicAI'}><img src="/brand/logo.png" alt="NupicAI" /></Link>
      <nav><Link href={en ? '/en' : '/'}>{en ? 'Home' : 'Strona główna'}</Link><Link href={en ? '/en/terms' : '/regulamin'}>{en ? 'Terms' : 'Regulamin'}</Link><Link href={en ? '/en/privacy' : '/privacy'}>{en ? 'Privacy' : 'Prywatność'}</Link></nav>
    </header>
    <article className="legal-document">
      <p className="section-label">NupicAI</p><h1>{title}</h1>
      <p className="legal-version">{en ? 'Effective version' : 'Wersja obowiązująca'}: {en ? VERSION_EN : VERSION}</p>
      <div className="legal-intro">{kind === 'terms' ? (en ? 'Rules for the free NupicAI pilot and electronic services.' : 'Zasady bezpłatnego pilota NupicAI i świadczenia usług drogą elektroniczną.') : (en ? 'How we process personal data and working media.' : 'Jak przetwarzamy dane osobowe i materiały robocze.')}</div>
      {sections.map(([heading, body]) => <section key={String(heading)}><h2>{heading}</h2><p>{body}</p></section>)}
    </article>
  </main>;
}
