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
# Edytuj .env: ustaw WEGORZ_ADMIN_TOKEN i NUPIC_API_KEY.
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

`start.sh` automatycznie laduje lokalny plik `.env`. `WEGORZ_ADMIN_TOKEN` chroni zakladke Administrator. Bez tej zmiennej uslugi uzytkownika nadal dzialaja, ale backend administracyjny pozostaje zablokowany. `WEGORZ_CORS_ORIGINS` powinno zawierac wylacznie adresy frontendu, ktore maja korzystac z API.

## Widoki uzytkownika i administratora

Zwykly uzytkownik ma dostep do materialu, tekstu, tlumaczenia, glosu, tempa, miksu i eksportu. Parametry checkpointow, flow, duration, klucz API oraz diagnostyka zadan nie sa wyswietlane w standardowym workflow.

Zakladka Administrator zawiera:

- endpoint, model i tryb tlumaczenia,
- zapis lub usuniecie klucza API,
- stan ASR i TTS,
- aktywny profil TTS oraz modele zaladowane do pamieci,
- ostatnie zadania i sciezki logow diagnostycznych.

Klucz API jest zapisywany lokalnie w `admin_config.json` z uprawnieniami `0600`. Plik jest ignorowany przez Git, a frontend otrzymuje tylko zamaskowana koncowke klucza. Zmienna `NUPIC_API_KEY` ma pierwszenstwo przy starcie serwera.

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
```

`styleenc128_lstm` jest profilem domyslnym. Wybor technicznego profilu nie jest pokazywany zwyklemu uzytkownikowi. Oba checkpointy zawieraja potrzebne wagi enkodera lub tablic learned voice; runtime nie pobiera ich z katalogu treningowego.
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
