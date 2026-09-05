# NupicAI - plan ulepszeń produkcyjnych

Plan został przeniesiony z `parakeet-ui/public`, ponieważ pliki z tego katalogu są
publikowane bezpośrednio przez frontend. Dane logowania nie mogą znajdować się w
notatkach ani w repozytorium.

## Zrealizowane

- Połączenie marki z serwisem [ZróbEbooka](https://zrobebooka.pl/) i wyjaśnienie
  różnicy między dubbingiem NupicAI a modelem do polskich audiobooków.
- Opis własnej części TTS, polskiej specjalizacji, równoległej syntezy i pełnego
  przepływu od transkrypcji do miksu.
- Role administratora przypisane do konta. Panel nie wymaga drugiego tokenu i
  jest niewidoczny dla zwykłych użytkowników.
- Konto właściciela wskazane w `.env` ma uprawnienia administratora oraz
  nielimitowane renderowanie.
- Ostrożny opis syntetycznych presetów głosu i odpowiedzialności za prawa do
  materiału źródłowego.
- Powrót ze studia do publicznej strony bez wylogowania.
- Stopka firmy, dane kontaktowe i informacja o budowie projektu w Polsce.
- Osobny stan wyczerpanego limitu z przejściem do konta.
- Tłumaczenie tekstu wklejonego bez pliku audio.
- Jednorazowy reset hasła przez Resend z hashowanym tokenem, terminem ważności i
  unieważnieniem wcześniejszych sesji po zmianie hasła.
- Cennik z rzeczywistym przeliczeniem kosztu pakietu, bez niezweryfikowanych
  porównań cen konkurencji.
- Uczciwa sekcja ograniczeń modelu i kontakt do zgłaszania problemów.
- Akronim NUPIC pozostaje rozwijany jako *Neural Unified Platform for Intelligent
  Communication*, ponieważ ta wersja obejmuje literę U.
- Polskie i angielskie treści, metadane, canonical/hreflang, Open Graph, dane
  strukturalne WebApplication i dane organizacji.
- Historia projektu dwóch braci i treningu na prywatnym sprzęcie z RTX 3090.
- Regulamin i polityka prywatności pilota po polsku i angielsku, podlinkowane
  przed rejestracją; baza zapisuje wersję dokumentów i czas akceptacji.
- Samodzielne usuwanie konta po ponownym podaniu hasła.
- Limit rozmiaru i allowlista rozszerzeń uploadu, ograniczenie prób logowania i
  rejestracji oraz podstawowe nagłówki bezpieczeństwa.
- Przykładowe wdrożenie systemd/Nginx i spójny backup SQLite w `deploy/`.
- Tryb `check_production.py --strict` kontrolujący HTTPS, cookies, hosty, CORS,
  administratora, pocztę, prawa pliku `.env` i produkcyjny build frontendu.
- Osobny endpoint `/ready`, który zwraca 503 do czasu załadowania ASR i TTS.
- Historia 100 zmian edytora dubbingu z cofaniem/ponawianiem tekstu, głosu,
  dzielenia i łączenia segmentów oraz stabilnymi identyfikatorami wierszy.
- Usunięcie pierwotnych notatek z katalogu publicznego; zawierały dane logowania
  i nie mogą trafić do statycznego eksportu.

## Do decyzji biznesowej

- Podpięcie operatora płatności i faktycznego zakupu pakietów. Obecny cennik jest
  opisany jako planowany.
- Adres domeny produkcyjnej dla `NEXT_PUBLIC_SITE_URL` oraz wpis `Sitemap` w
  `robots.txt` po ustaleniu domeny NupicAI.
- Przegląd prawny obecnego regulaminu i polityki prywatności oraz rozszerzenie
  ich o płatności, odstąpienie i zwroty przed uruchomieniem sprzedaży.
- Potwierdzanie adresu e-mail. Dostawca Resend i odzyskiwanie hasła są już
  obsłużone w kodzie, ale wymagają ustawienia zweryfikowanego nadawcy.
- Monitoring, alerty, zewnętrzna kopia bazy i decyzja o trwałym standardzie
  maszynowego znakowania audio wygenerowanego przez AI.

## Audyt produkcyjny 2026-09-02

Kod i zasoby lokalne są kompletne dla wdrożenia pilota na jednym serwerze i
jednej karcie GPU. Test kompletności obejmuje ASR, dwa checkpointy TTS, lokalny
translator, vocoder, bank 40 wybranych głosów, mapę learned voice, Deno,
`ffmpeg`, zależności Pythona oraz gotowy statyczny frontend.

Przed uruchomieniem płatnej usługi, w tej kolejności:

1. Ustawić domenę, TLS, secure cookies, dozwolone hosty/CORS i Resend, a następnie
   uzyskać czysty wynik `python check_production.py --strict`.
2. Dodać potwierdzanie adresu e-mail oraz podłączyć płatności z serwerowym
   katalogiem produktów, podpisanym i idempotentnym webhookiem oraz historią
   transakcji. Dokumenty prawne muszą wtedy objąć zakup, odstąpienie i zwroty.
3. Dodać trwałą historię projektów i odzyskiwanie zadań po restarcie procesu.
   Obecna kolejka i stan aktywnych zadań są pamięciowe, dlatego wdrożenie musi
   pozostać jednoprocesowe.
4. Dodać anulowanie oczekujących zadań i kooperatywne zatrzymanie renderingu
   między segmentami, aby nie zajmować GPU po zamknięciu projektu w przeglądarce.
5. Włączyć monitoring błędów, czasu kolejki i VRAM, alert gotowości `/ready`,
   zewnętrzną kopię bazy oraz okresowy test odtworzenia backupu.

Ulepszenia edytora po starcie pilota:

- regulacja granicy dwóch sąsiednich segmentów na osi czasu, z walidacją braku
  luk i nakładania;
- automatyczna diarization jako propozycja obsady dialogów, zawsze z ręczną
  korektą użytkownika;
- zapis szkicu projektu i ponowne otwarcie go z konta;
- testy dostępności klawiatury, czytnika ekranu i kontrastu oraz pomiary Core
  Web Vitals na prawdziwej domenie.
