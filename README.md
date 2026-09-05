# NupicAI Studio

Hermetyczny folder aplikacji do transkrypcji, tlumaczenia, dubbingu i syntezy glosu z lokalnym ASR Parakeet, lokalnym TTS Wegorz, lokalnym vocoderem Vocos, lokalnym bankiem glosow oraz lokalnym modelem tlumaczenia Wegorz.

NupicAI jest marka aplikacji. Nazwa Wegorz pozostaje nazwa lokalnego silnika TTS i translatora.

## Logo i identyfikacja

Glowne logo umiesc jako:

```text
parakeet-ui/public/brand/logo.png
```

Logo moze byc szerokim, przezroczystym PNG albo WebP. Interfejs miesci je w obszarze `196x58 px`; gdy pliku nie ma, automatycznie pokazuje ikone audio i tekst NupicAI. Po zmianie logo wykonaj `npm run build` w katalogu `parakeet-ui`.

Interfejs rozdziela cztery uslugi, ale zachowuje jeden wspolny projekt:

```text
Transkrypcja -> Tlumaczenie -> Dubbing
                         +-> Studio glosu
```

Wynik poprzedniego etapu pozostaje w pamieci projektu, dlatego przejscie do kolejnej uslugi nie uruchamia ponownie ASR ani tlumaczenia.

Edytor dubbingu pozwala przypisac inny glos do pojedynczej sceny, polaczyc segment z nastepnym, a potem podzielic polaczony tekst w miejscu kursora. Przycisk ponowienia przy segmencie zmienia jego seed i generuje tylko ten fragment. Backend kopiuje niezmienione WAV-y segmentow z poprzedniego renderu, sklada nowa os czasu i nalicza limit za faktycznie wygenerowane ponownie audio, a nie za cala produkcje.

Zmiany tekstu, glosu, podzialu i laczenia segmentow maja historie 100 krokow.
Przyciski cofania i ponawiania sa w naglowku edytora; poza polami tekstowymi
dzialaja tez standardowe skroty `Ctrl+Z`, `Ctrl+Shift+Z` i `Ctrl+Y`.

## Co jest w tym folderze

```text
dubbing/
  server.py
  start.sh
  check_production.py
  requirements.txt
  models/
    asr/
      parakeet-tdt-0.6b-v3.nemo
    translate/
      wegorz_translator_32k_best.pt
    tts/
      checkpoints/
        styleenc128_lstm.pt
        mini_dualpath_learnedvoice.pt
        minidualpath_bins_maskgit_continuity_ep742.pt
      vocos-mel-24khz/
      voice_banks/
        selected_top_voices_current.pt
  translate/
    parakeet_translation_core.py
    translate_wegorz_sentence_split.py
    model_dualpath_v3.py
    model_shared.py
    wegorz.model
  tts/
    tts_daemon.py
    wegorz_tts_model.py
    tts_helpers.py
    inference_helpers.py
    vocab_pl_orth_en_ipa_bridge.json
    learned_voice_speaker_map.json
    wegorz_normalizer/
  parakeet-ui/out/          # gotowy frontend; Node.js nie jest potrzebny na produkcji
```

Najwazniejsza zasada: runtime nie powinien importowac modeli ani modulow dubbingu z zewnetrznych folderow projektu. Modele potrzebne do dubbingu sa w `models/`, a kod TTS i tlumacza jest w `tts/` oraz `translate/`.

## Wymagania

Folder zawiera modele i kod aplikacji, ale nie zawiera calego srodowiska Pythona. Dokladne zaleznosci sa w `requirements.txt`. Zalecane wymagania:

- Python 3.11
- NVIDIA CUDA 12.8 lub zgodny sterownik dla dolaczonej wersji PyTorch
- co najmniej okolo 12 GB VRAM dla pojedynczego profilu TTS; oba profile trzymane jednoczesnie potrzebuja wiecej
- `ffmpeg` w `PATH`
- pobieranie z YouTube korzysta z `yt-dlp[default]` oraz dolaczonego `tools/deno/deno` (Linux x86_64)

Tryb zdalnego tlumaczenia Qwen/API wymaga sieci i klucza API. Tryb lokalnego tlumacza Wegorz uzywa lokalnego checkpointu z `models/translate/`.

## Instalacja i uruchamianie

```bash
cd /sciezka/do/dubbing
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# Edytuj .env: ustaw NUPIC_API_KEY i NUPICAI_ADMIN_EMAILS.
python check_production.py
./start.sh
```

Potem otworz:

```text
http://127.0.0.1:8765
```

Jesli aplikacja ma byc dostepna z innych komputerow w sieci:

```bash
# W .env ustaw HOST=0.0.0.0 oraz dopisz publiczny origin do WEGORZ_CORS_ORIGINS.
./start.sh
```

`start.sh` automatycznie laduje lokalny plik `.env`. Konta wymienione w `NUPICAI_ADMIN_EMAILS` otrzymuja dostep do panelu Administrator oraz nielimitowany rendering. Panel korzysta z tej samej bezpiecznej sesji co pozostala czesc aplikacji. `WEGORZ_CORS_ORIGINS` powinno zawierac wylacznie adresy frontendu, ktore maja korzystac z API.

## Konta i retencja danych

Publiczna strona startowa prowadzi do rejestracji lub logowania. Konta sa przechowywane w lokalnej bazie `runtime/nupicai.sqlite3` (ignorowanej przez Git). Hasla sa zapisywane jako PBKDF2-SHA256 z osobna losowa sola, a token sesji w bazie wystepuje tylko jako SHA-256. Przegladarka otrzymuje sesje w ciasteczku `HttpOnly` z `SameSite=Lax`.

Odzyskiwanie hasla korzysta z jednorazowego linku wysylanego przez Resend. W bazie znajduje sie tylko SHA-256 tokenu, link domyslnie wygasa po godzinie, a udana zmiana hasla uniewaznia wszystkie dotychczasowe sesje. Skonfiguruj zweryfikowanego nadawce:

```env
NUPICAI_PUBLIC_URL=https://nupicai.example
NUPICAI_PASSWORD_RESET_TTL_SECONDS=3600
RESEND_API_KEY=re_...
NUPICAI_EMAIL_FROM=NupicAI <noreply@nupicai.example>
```

Kazde zadanie ma wlasciciela. Pliki trafiaja do osobnej przestrzeni:

```text
/tmp/parakeet_server/users/<user_id>/jobs/<job_id>/
```

Backend sprawdza wlasciciela rowniez przy bezposrednim pobieraniu audio, wideo, zrodla i strumienia SSE. Zakonczone pliki oraz prompty glosowe sa automatycznie usuwane po czasie ustawionym przez `NUPICAI_DATA_RETENTION_HOURS` (domyslnie 24 h). Uzytkownik moze tez natychmiast usunac pliki albo trwale zamknac konto po ponownym podaniu hasla w zakladce `Moje konto`.

Na serwerze publicznym wymagane jest HTTPS. Za reverse proxy ustaw:

```bash
NUPICAI_SECURE_COOKIES=1
```

`NUPICAI_SESSION_DAYS` steruje czasem sesji (domyslnie 30 dni). Czyszczenie uruchamia sie przy starcie i nastepnie raz na godzine; aktywne zadania nie sa usuwane.

## Limity generowania

Nowe konto otrzymuje limit ustawiony przez `NUPICAI_FREE_SECONDS` (domyslnie 300 sekund). Limit jest rozliczany za zakonczone audio z dubbingu i studia glosu. Transkrypcja oraz tlumaczenie sa obecnie wlaczone w usluge i nie zuzywaja osobnego salda.

Przed rozpoczeciem renderingu backend atomowo rezerwuje szacowany czas. Chroni to przed przekroczeniem salda przez kilka rownoleglych zadan. Po sukcesie pobierany jest rzeczywisty czas pliku, a po bledzie rezerwacja jest zwalniana. Rezerwacje pozostawione przez przerwany proces sa zwalniane przy ponownym uruchomieniu serwera. Stan konta jest dostepny przez `GET /account/usage` i widoczny w naglowku oraz zakladce `Moje konto`.

Saldo jest przechowywane w sekundach w `runtime/nupicai.sqlite3`. Metoda `AuthStore.add_credits(user_id, seconds)` stanowi punkt integracji dla panelu operatora lub webhooka platnosci. Przed sprzedaza pakietow nalezy podlaczyc dostawce platnosci z idempotentnym identyfikatorem transakcji; sam interfejs platnosci nie jest jeszcze wdrozony.

## Checklista przed platna produkcja

- uruchomic aplikacje za HTTPS i ustawic `NUPICAI_SECURE_COOKIES=1`;
- podlaczyc platnosci do `AuthStore.add_credits` i zapisywac unikalny identyfikator transakcji;
- skonfigurowac Resend dla odzyskiwania hasla i dodac potwierdzanie adresu e-mail;
- zastapic pamieciowy rate limiter wspolnym magazynem (np. Redis), jezeli aplikacja bedzie uruchamiana w wielu procesach;
- wlaczyc skanowanie antymalware multimediow, jezeli model zagrozen uzasadnia koszt;
- wdrozyc monitoring bledow, metryki kolejki GPU, alerty oraz kopie bazy poza serwerem;
- dla wielu procesow lub wielu serwerow przeniesc konta, ledger i kolejke z SQLite/pamieci do PostgreSQL oraz wspolnej kolejki zadan;
- poddac regulamin i polityke prywatnosci przegladowi prawnemu oraz dodac zasady zakupu, odstapienia i zwrotow przed przyjmowaniem platnosci.

Backend ogranicza liczbe prob logowania i rejestracji, rozmiar uploadu oraz rozszerzenia multimediow. Dodaje CSP, HSTS przy bezpiecznych cookies, `nosniff`, ochrone przed osadzaniem w ramkach i polityke uprawnien. Limity aplikacji nie zastepuja limitow i TLS na reverse proxy.

## Regulamin, prywatnosc i tresci AI

Dokumenty pilota sa dostepne pod `/regulamin`, `/privacy`, `/en/terms` i `/en/privacy`. Rejestracja zapisuje wersje obu dokumentow i czas akceptacji. Przed wlaczeniem platnosci trzeba uzupelnic zasady zakupu, odstapienia i zwrotow oraz zlecic przeglad prawny.

Pobrane audio i wideo syntetyczne ma nazwe wskazujaca na AI i naglowek `X-AI-Generated-Content: true`. Jest to pomocnicze oznaczenie, a nie gwarancja pelnej zgodnosci z wymogiem trwalego, maszynowo czytelnego znakowania. Przed publicznym wdrozeniem trzeba wybrac standard metadanych/proweniencji odpowiedni dla audio.

## Wdrozenie serwerowe

Przyklady dla systemd, Nginx i kopii SQLite sa w `deploy/`. Przed startem uzupelnij `.env`, domene i certyfikat, potem uruchom `python check_production.py` oraz testy.

Na publicznym serwerze ustaw `NUPICAI_PRODUCTION=1` i użyj
`python check_production.py --strict`. Tryb rygorystyczny kończy się błędem, gdy
pozostają adresy localhost, niezabezpieczone cookies, luźne hosty/CORS,
nieskonfigurowana poczta albo frontend zbudowany dla innej domeny. Endpoint
`/ready` zwraca HTTP 503 do czasu pełnego załadowania ASR i TTS; `/health`
pozostaje endpointem diagnostycznym.

## Jezyki i SEO

Publiczna strona ma indeksowalne wersje polska (`/`) i angielska (`/en`) z osobnymi tytulami, opisami, linkami `hreflang`, adresami kanonicznymi i poprawnym atrybutem `lang`. Dla osoby bez zapisanego wyboru interfejs wybiera polski, gdy jezyk przegladarki zaczyna sie od `pl`, a w pozostalych przypadkach angielski. Reczny wybor PL/EN jest zapisywany w przegladarce. Jezyk docelowy tlumaczenia moze byc ustawiony niezaleznie na polski albo angielski.

Przed publicznym buildem ustaw prawdziwy adres wdrozenia, a dopiero potem przebuduj frontend:

```bash
NEXT_PUBLIC_SITE_URL=https://twoja-domena.example npm run build
```

FastAPI wystawia dynamiczne `/robots.txt` i `/sitemap.xml`, dlatego uwzglednia faktyczny host wdrozenia. Landing zawiera widoczny opis produktu, semantyczne naglowki, WebApplication JSON-LD oraz opisowe teksty alternatywne. Strona FAQ pozostaje zwykla trescia HTML; nie udaje danych strukturalnych FAQ przeznaczonych dla serwisow medycznych lub rzadowych.

Audio dla ASR i TTS jest przetwarzane lokalnie. Przy zdalnym trybie tlumaczenia tekst segmentow jest wysylany do endpointu skonfigurowanego w panelu administratora. Ta informacja musi pozostac w polityce prywatnosci wdrozenia.

## YouTube i aktualizacja yt-dlp

Downloader wykonuje kontrolowana probe standardowego formatu, a po bledzie ponawia pobieranie zgodnym strumieniem MP4. Rozpoznaje filmy prywatne, ograniczenia geograficzne, wymaganie cookies, problemy sieciowe oraz bledy 403/PO Token. Szczegoly techniczne sa zapisywane w `yt_dlp_attempts.log` w katalogu zadania, bez pokazywania surowego logu uzytkownikowi.

Nie aktualizuj pakietow automatycznie w trakcie zadania HTTP. Przy powtarzalnych bledach YouTube wykonaj kontrolowana aktualizacje w srodowisku aplikacji, test gotowosci i restart:

```bash
source /srv/nupicai/.venv/bin/activate
python -m pip install --upgrade "yt-dlp[default]"
python check_production.py
sudo systemctl restart nupicai
```

Dolaczony Deno musi miec wersje co najmniej 2.3. Filmy wymagajace konta konfiguruje sie przez `WEGORZ_YTDLP_COOKIES_FILE` albo `WEGORZ_YTDLP_COOKIES_FROM_BROWSER`. Przy aktywnym wymogu Proof of Origin nalezy wdrozyc utrzymywanego dostawce PO Token i przekazac jego ustawienia przez `WEGORZ_YTDLP_EXTRACTOR_ARGS`; nie nalezy wpisywac recznie tokenu zwiazanego z pojedynczym filmem. Zewnetrzne serwisy typu „YouTube downloader” nie sa fallbackiem produkcyjnym: ujawnialyby adres materialu i nie zapewniaja stabilnego API ani kontroli nad plikiem.

## Płatności ze ZróbEbooka

ZróbEbooka zawiera dzialajace klocki PayU, BTCPay, historie transakcji oraz naliczanie sekund. NupicAI moze wykorzystac te same konta operatorow i podobny interfejs, ale kodu nie nalezy kopiowac bez zmian, ponieważ aplikacje maja inny backend, a stara sciezka tworzenia zamowienia przyjmuje cene i wielkosc pakietu z przegladarki.

Bezpieczna integracja NupicAI musi:

- przyjmowac z frontendu wyłącznie identyfikator pakietu, a cene, walute i liczbe sekund pobierac z serwerowego katalogu;
- zapisac zamowienie `pending` przed przekierowaniem do operatora;
- zweryfikowac podpis webhooka na surowym body i dodatkowo porownac kwote, walute oraz identyfikator sprzedawcy;
- w jednej transakcji SQL zmienic status zamowienia i naliczyc sekundy tylko raz;
- posiadac unikalny identyfikator operatora, historie platnosci, zwroty i proces uzgadniania brakujacych webhookow;
- nigdy nie pomijac weryfikacji podpisu BTCPay tylko dlatego, ze sekret nie zostal ustawiony.

Istniejace `AuthStore.add_credits` pozostaje punktem administracyjnym. Webhook platnosci powinien korzystac z osobnej atomowej metody rozliczenia zamowienia, a nie wywolywac jej bezposrednio.

## Widoki uzytkownika i administratora

Zwykly uzytkownik ma dostep do materialu, tekstu, tlumaczenia, glosu, tempa, miksu i eksportu. Parametry checkpointow, flow, duration, klucz API oraz diagnostyka zadan nie sa wyswietlane w standardowym workflow.

Zakladka Administrator zawiera:

- endpoint, model i tryb tlumaczenia,
- zapis lub usuniecie klucza API,
- stan ASR i TTS,
- aktywny profil TTS oraz modele zaladowane do pamieci,
- ostatnie zadania i sciezki logow diagnostycznych.

Klucz API jest zapisywany lokalnie w `runtime/admin_config.json` z uprawnieniami `0600`. Starszy `admin_config.json` jest jednokrotnie migrowany do tego katalogu. Plik jest ignorowany przez Git, a frontend otrzymuje tylko zamaskowana koncowke klucza. Zmienna `NUPIC_API_KEY` ma pierwszenstwo przy starcie serwera.

## Miks dubbingu

Dubbing tworzy dwa pliki:

- `dubbed.wav` - sam wygenerowany glos,
- `mixed.wav` - glos polaczony z oryginalna sciezka.

Panel miksu steruje poziomem oryginalu, poziomem dubbingu i sidechain duckingiem. Ducking automatycznie scisza oryginal podczas wypowiedzi lektora, z lagodnym attack/release i limiterem na wyjsciu. Bez separacji stemow suwak oryginalu obejmuje jednoczesnie mowe, muzyke i efekty z materialu zrodlowego.

## Modele lokalne

### ASR

```text
models/asr/parakeet-tdt-0.6b-v3.nemo
```

ASR jest ladowany lokalnie przez `EncDecRNNTBPEModel.restore_from(...)`. To oznacza, ze aplikacja nie musi pobierac modelu Parakeet z HuggingFace, o ile plik `.nemo` istnieje.

### TTS

Strona zawiera dwa kompletne profile zarzadzane przez backend:

```text
models/tts/checkpoints/styleenc128_lstm.pt
models/tts/checkpoints/mini_dualpath_learnedvoice.pt
models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt
```

Domyslny profil wybiera administrator. `maskgit_continuity` zawiera TDA-MaskGIT duration, stan rytmu i pamiec akustyczna poprzedniego chunku. Checkpointy zawieraja potrzebne wagi enkodera lub tablic learned voice; runtime nie pobiera ich z katalogu treningowego.
Mapa `tts/learned_voice_speaker_map.json` odwzorowuje surowe ID z banku glosow na wiersze tablicy learned voice, dlatego manifest treningowy nie jest potrzebny na produkcji.

Architektura modelu jest w:

```text
tts/wegorz_tts_model.py
```

Vocoder:

```text
models/tts/vocos-mel-24khz/
```

Bank glosow:

```text
models/tts/voice_banks/selected_top_voices_current.pt
```

### Lokalny tlumacz Wegorz

```text
models/translate/wegorz_translator_32k_best.pt
translate/wegorz.model
```

Pierwsze uruchomienie lokalnego tlumacza trwa dluzej, bo model musi zostac zaladowany. Kolejne tlumaczenia sa szybsze, bo model jest trzymany w pamieci procesu.

## TTS jako osobny modul do innej strony

TTS jest wydzielony tak, zeby mozna bylo go wziac do innej aplikacji bez calego systemu dubbingu.

Minimalny zestaw do przeniesienia:

```text
tts/
models/tts/checkpoints/styleenc128_lstm.pt
models/tts/checkpoints/mini_dualpath_learnedvoice.pt
models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt
models/tts/vocos-mel-24khz/
models/tts/voice_banks/selected_top_voices_current.pt
```

Najprostsza integracja to uruchomienie:

```bash
python tts/tts_daemon.py \
  --resume models/tts/checkpoints/styleenc128_lstm.pt \
  --dataset-json tts/manifest_runtime_refs.json \
  --vocab tts/vocab_pl_orth_en_ipa_bridge.json \
  --device cuda
```

Daemon laduje model raz, a potem czyta pojedyncze zapytania JSON ze `stdin` i zwraca odpowiedzi JSON na `stdout`.

Przyklad zapytania:

```json
{
  "text": "To jest test syntezy mowy.",
  "voice_emb": "/tmp/parakeet_server/voice_bank_embs/voice_108080.pt",
  "speed": 1.0,
  "mel_steps_first": 8,
  "mel_steps_second": 3,
  "mel_twopass_t_noise": 0.12,
  "seed": 1234,
  "out_dir": "/tmp/wegorz_tts",
  "tag": "test",
  "lang": "pl",
  "digital_silence": true,
  "pause_edge_frames": 10,
  "short_continuity_ms": 128.0,
  "emotion_group": "neutral",
  "emotion_strength": 0.0
}
```

Przyklad odpowiedzi:

```json
{"wav": "/tmp/wegorz_tts/test.wav"}
```

W praktyce `server.py` robi dodatkowy wrapper:

- wybiera glos z voice banku,
- dzieli dluzszy tekst na segmenty,
- utrzymuje stan continuation,
- sklada wynikowe WAV-y,
- zapisuje log debug duration/tokenow.

Najwazniejsze funkcje integracyjne w `server.py`:

- `_start_daemon_locked()`
- `_daemon_call()`
- `_daemon_synth_chunked_response()`
- `_speaker_condition_payload()`

## Budowanie nowej bazy glosow

W folderze TTS jest skrypt:

```text
tts/build_voice_database.py
```

Przyklad:

```bash
cd /sciezka/do/dubbing

python tts/build_voice_database.py \
  --checkpoint models/tts/checkpoints/styleenc128_lstm.pt \
  --dataset-json /path/to/manifest.json \
  --out models/tts/voice_banks/selected_top_voices.pt \
  --device cuda \
  --max-mels-per-speaker 8
```

Manifest musi zawierac sciezki do mel/audio w formacie obslugiwanym przez obecny loader danych. Skrypt bierze kilka probek danego speakera, przepuszcza je przez `speaker_encoder` z checkpointu i usrednia wektor glosu.

## Lokalnosc i hermetycznosc

Sprawdzone elementy dzialaja z lokalnych plikow w `dubbing/`:

- ASR Parakeet `.nemo`
- TTS checkpoint
- TTS daemon
- Vocos
- voice bank
- lokalny Wegorz translator
- tokenizer lokalnego tlumacza

Uwaga: w niektorych duzych plikach modelu moga istniec stare sciezki w komentarzach, argumentach CLI treningu albo nieuzywanych fallbackach. Nie sa one czescia runtime serwera. Sciezka wykonania strony uzywa lokalnych modeli z `models/`.

## Integralnosc przeplywu

- ASR uzywa okien 180 s z overlappem 2 s. Granica odpowiedzialnosci lezy w polowie overlapu, dlatego slowo przeciete na 180 s trafia do nastepnego, kompletnego okna, ale nie jest dublowane.
- Koncowe okno dostaje 1.2 s ciszy, aby RNNT wypchnal ostatnie tokeny.
- Segmentacja zachowuje wszystkie slowa i ich kolejnosc.
- Tlumacz musi zwrocic dokladnie jeden niepusty wynik na kazdy segment; brakujacy fragment zatrzymuje zadanie.
- Dubbing odrzuca pusty segment zamiast go cicho pomijac.

Parametry okien mozna zmienic przez `PARAKEET_ASR_WINDOW_SEC`, `PARAKEET_ASR_OVERLAP_SEC` i `PARAKEET_ASR_FINAL_TAIL_PAD_SEC`.

## Szybkie testy techniczne

Kompilacja plikow:

```bash
python -m py_compile \
  server.py \
  tts/tts_daemon.py \
  translate/parakeet_translation_core.py \
  translate/translate_wegorz_sentence_split.py
```

Test integralnosci chunkowania i segmentow:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Przebudowanie frontendu nie jest potrzebne przy zwyklym uruchomieniu. Do pracy nad UI uzyj Node.js 18+:

```bash
cd parakeet-ui
npm install
npm run build
```

Wynik trafia do `parakeet-ui/out/` i jest serwowany bezposrednio przez FastAPI.

## Znane ograniczenia

- Folder nie zawiera kompletnego virtualenv/conda env; odtwarza je `requirements.txt`.
- `ffmpeg` musi byc zainstalowany systemowo.
- YouTube wymaga aktualnego `yt-dlp[default]`; zgodny Deno 2.9.5 jest dolaczony w `tools/deno/`.
- Jezeli YouTube wymaga zalogowania, ustaw przed startem serwera `WEGORZ_YTDLP_COOKIES_FROM_BROWSER=firefox` albo `WEGORZ_YTDLP_COOKIES_FILE=/sciezka/cookies.txt`.
- Zdalny tryb Qwen/API wymaga sieci i klucza API.
- Pierwsze ladowanie ASR/TTS moze trwac kilkadziesiat sekund, zalezne od GPU/CPU.
- Na maszynach bez CUDA aplikacja moze dzialac na CPU, ale TTS/ASR beda znacznie wolniejsze.
