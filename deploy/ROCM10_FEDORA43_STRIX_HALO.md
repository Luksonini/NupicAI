# NupicAI: plan wdrozenia na Fedora 43 + ROCm 10 + Strix Halo

Stan badan: 2026-09-03. Dokument jest planem wykonawczym dla kolejnego agenta
oraz checklista czynnosci manualnych dla wlasciciela serwera. Celem jest jedno
repozytorium i jeden plik `compose.yaml`, uruchamiajacy cala aplikacje NupicAI.

## 1. Cel i ograniczenia

Docelowy host:

- GMKtec z AMD Ryzen AI Max / Max+ (Strix Halo, `gfx1151`);
- Fedora 43 z aktualnym kernelem i hostowym sterownikiem `amdgpu`;
- Docker Engine z pluginem Compose;
- ROCm 10 i PyTorch ROCm wewnatrz kontenera Ubuntu;
- jeden proces FastAPI/Uvicorn oraz jego procesy TTS daemon;
- opcjonalny Caddy jako drugi serwis w tym samym `compose.yaml`;
- lokalne modele ASR, translatora, TTS i Vocos montowane tylko do odczytu;
- SQLite i dane uzytkownikow na trwalych wolumenach.

Nie uruchamiac wielu workerow Uvicorn. Kolejka GPU, cache modeli i komunikacja z
daemonami TTS sa lokalne dla jednego procesu. Skalowanie poziome wymaga osobnego
projektu z PostgreSQL, Redisem i wspolna kolejka zadan.

## 2. Aktualny stan zgodnosci

1. ROCm 10 ma natywne artefakty dla `gfx1151` i oficjalne obrazy PyTorch, m.in.
   `rocm/pytorch:rocm10.0_ubuntu24.04_py3.11_pytorch_release_2.12.0`.
2. Fedora 43 jest wspierana przez Docker Engine. Nie jest jednak wymieniona jako
   oficjalny system Ryzen APU w macierzy ROCm 10; dlatego Fedora jest tylko
   hostem, a userspace ROCm pochodzi z oficjalnego obrazu Ubuntu.
3. Strix Halo wymaga poprawek KFD obecnych w kernelu `6.18.4+`. Dokumentacja AMD
   wymienia Fedore 43 jako dystrybucje zawierajaca wymagane poprawki, ale agent ma
   sprawdzic faktycznie uruchomiona wersje kernela, nie tylko nazwe dystrybucji.
4. PyTorch ROCm celowo uzywa API `torch.cuda`; obecne `device="cuda"` w aplikacji
   nie oznacza zaleznosci od karty NVIDIA. Poprawna identyfikacja to jednoczesnie
   `torch.cuda.is_available() == True`, `torch.version.hip != None` i
   `torch.version.cuda == None`.
5. Parakeet TDT nadal ma otwarty problem w NeMo: label-looping decoder moze
   sprobowac zaladowac `libcuda.so.1`, gdy zainstalowany jest `cuda-python`.
   Sam Parakeet dziala na ROCm po wylaczeniu CUDA Graph decoder. Jest raport
   potwierdzajacy ok. 650 godzin transkrypcji na ROCm z tym obejsciem.
6. Obecny `requirements.txt` jest wariantem CUDA (`cu128`) i nie moze byc uzyty
   bez zmian w obrazie ROCm.

Zrodla:

- [AMD ROCm 10 compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
- [AMD: instalacja PyTorch dla ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
- [AMD: optymalizacja Strix Halo](https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/strixhalo.html)
- [AMD: uruchamianie ROCm w Dockerze](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.3/how-to/docker.html)
- [PyTorch: semantyka HIP/ROCm](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [NeMo issue #15905: Parakeet TDT na ROCm](https://github.com/NVIDIA-NeMo/Speech/issues/15905)
- [Docker Engine na Fedora](https://docs.docker.com/engine/install/fedora/)

## 3. Docelowe artefakty do wykonania przez agenta

Agent ma przygotowac male, latwe do wycofania zmiany:

```text
dubbing/
  compose.yaml
  .dockerignore
  deploy/rocm10/
    Dockerfile
    requirements-rocm.txt
    entrypoint.sh
    rocm_smoke.py
    parakeet_smoke.py
  deploy/ROCM10_FEDORA43_STRIX_HALO.md
```

Nie modyfikowac ani nie konwertowac checkpointow. Nie kopiowac `.env`, SQLite,
plikow zadan, cache ani sekretow do obrazu.

## 4. Etapy implementacji dla agenta

### Etap A: inwentaryzacja i reprodukowalnosc

1. Zapisac wynik `git status`; nie usuwac zmian uzytkownika.
2. Uruchomic obecne `python check_production.py` i testy na maszynie zrodlowej.
3. Zapisac SHA-256 wszystkich wymaganych modeli oraz ich rozmiary do manifestu
   deploymentu. Manifest nie moze zawierac sekretow.
4. Potwierdzic, ze frontend `parakeet-ui/out` jest aktualny albo budowac go w
   osobnym etapie Dockerfile z ustawionym `NEXT_PUBLIC_SITE_URL`.

### Etap B: rozdzielenie zaleznosci CUDA i ROCm

1. Pozostawic obecny `requirements.txt` dla NVIDIA.
2. Utworzyc `deploy/rocm10/requirements-rocm.txt` bez `--extra-index-url cu128`,
   `torch` i `torchaudio`. Te dwa pakiety maja pochodzic wylacznie z obrazu AMD.
3. Zachowac pozostale wersje aplikacji jako punkt startowy, ale zweryfikowac
   resolverem zgodnosc `nemo_toolkit[asr]==2.7.3` z PyTorch 2.12/Python 3.11.
4. Instalowac zaleznosci z constraints/lockiem i zapisywac pelne `pip freeze` w
   obrazie. Nie uzywac nieprzypietego `latest`.
5. Jezeli NeMo instaluje `cuda-python`, preferowac jawne wylaczenie CUDA Graph w
   kodzie. Usuniecie `cuda-python` pozostawic jako dodatkowy bezpiecznik po
   potwierdzeniu, ze nic innego go nie potrzebuje.

Brama: importy `torch`, `torchaudio`, `nemo.collections.asr`, `vocos`, FastAPI i
lokalnych modulow TTS musza przejsc w czystym kontenerze.

### Etap C: poprawki przenosnosci kodu

1. W `ParakeetTranscriber`, po zaladowaniu modelu i przed pierwszym
   `transcribe()`, wykryc ROCm przez `torch.version.hip` i ustawic w konfiguracji
   dekodera TDT `greedy.use_cuda_graph_decoder=False`, po czym wywolac
   `change_decoding_strategy`. Zalogowac wybrany decoder raz przy starcie.
2. Dodac probe regresyjna, ktora zaklada obecny pakiet `cuda-python` i dowodzi,
   ze Parakeet nie probuje otwierac `libcuda.so.1`.
3. W implementacjach SDPA nie wymuszac tylko backendow Flash/Efficient. Gdy
   backend ROCm/AOTriton nie obsluzy maski lub ksztaltu, kod ma przejsc do SDPA
   math albo istniejacego jawnego attention. Fallback musi byc testowany.
4. Nie zamieniac globalnie napisow `cuda` na `rocm` lub `hip`; PyTorch ROCm
   wymaga urzadzenia `cuda`.
5. Rozszerzyc `check_production.py` o tryb `--accelerator rocm`, ktory wymaga:
   `torch.version.hip`, `gfx1151`, rzeczywistej alokacji tensora na GPU oraz
   poprawnego przejscia Conv1d, LSTM, SDPA i `torchaudio`.
6. Dodac log startowy: wersja Torch, HIP, nazwa GPU, architektura, widoczna
   pamiec i wybrany backend attention. Brak HIP ma zatrzymac produkcyjny start,
   zamiast uruchamiac ASR/TTS na CPU.

### Etap D: Dockerfile

1. Bazowac na przypietym oficjalnym obrazie:

   ```text
   rocm/pytorch:rocm10.0_ubuntu24.04_py3.11_pytorch_release_2.12.0
   ```

   Przed finalizacja przypiac digest obrazu.
2. Zainstalowac tylko pakiety systemowe potrzebne w runtime: `ffmpeg`,
   `libsndfile1`, certyfikaty, `curl` oraz biblioteki wymagane przez NeMo/Vocos.
3. Uzyc wieloetapowego buildu frontendu albo kopiowac sprawdzone
   `parakeet-ui/out`. Node nie moze pozostac w obrazie runtime.
4. Uruchamiac aplikacje jako nie-root. Uzytkownik kontenera musi otrzymac grupy
   hostowych urzadzen `render` i `video` przez Compose.
5. `entrypoint.sh` ma najpierw wykonac szybki `rocm_smoke.py`, potem
   `check_production.py --strict --accelerator rocm`, a na koncu jeden proces
   `server.py` przez `exec`.
6. Dodac `HEALTHCHECK` oparty na `/ready`, nie tylko `/health`.

### Etap E: jeden compose

Jeden `compose.yaml` ma zawierac:

- `nupicai`: pojedynczy kontener aplikacji z `/dev/kfd` i `/dev/dri`,
  `security_opt: [seccomp=unconfined]`, `init: true`, `restart: unless-stopped`,
  odpowiednio duze `shm_size`, `env_file: .env` i healthcheck `/ready`;
- `caddy`: reverse proxy i TLS, zalezne od zdrowego `nupicai`; mozna je wylaczyc
  profilem, jezeli TLS zapewnia zewnetrzny proxy;
- trwale wolumeny dla `runtime/`, katalogu zadan, logow i danych Caddy;
- bind mount `./models:/app/models:ro` oraz potrzebnych lokalnych danych TTS;
- limity logow Docker (`max-size`, `max-file`).

Na Fedora stosowac oznaczenia SELinux `:Z`/`:z` dla zapisywalnych mountow.
Nie wylaczac SELinux globalnie. `label=disable` wolno rozwazyc dopiero po analizie
konkretnego AVC i udokumentowaniu powodu.

Nie ustawiac starego `HSA_OVERRIDE_GFX_VERSION`. ROCm 10 ma natywny target
`gfx1151`; override moglby zaladowac niewlasciwe kernele. Nie wlaczac od razu
eksperymentalnych flag AOTriton. Najpierw zgodnosc, potem osobny benchmark.

### Etap F: testy akceptacyjne na prawdziwym sprzecie

Testy maja zatrzymywac wdrozenie przy dowolnym fallbacku CPU:

1. `rocminfo` widzi `gfx1151`; kontener widzi `/dev/kfd` i render node.
2. `torch.cuda.is_available()` jest prawda, `torch.version.hip` nie jest puste,
   `torch.version.cuda` jest puste, tensor i wynik operacji sa na GPU.
3. Przejscie FP32 i FP16 dla matmul, Conv1d, LSTM i SDPA. BF16 nie jest wymagane
   w pierwszym wdrozeniu; dla Ryzen APU AMD walidowalo przede wszystkim FP16.
4. Zaladowanie lokalnego `parakeet-tdt-0.6b-v3.nemo` oraz transkrypcja stalego
   WAV. Wynik nie moze byc pusty, nie moze wystapic `libcuda.so.1`, a ostatnie
   zdanie/tail musi pozostac w transkrypcji.
5. Zaladowanie obu profili TTS, synteza tych samych tekstow PL i EN, zapis WAV,
   brak NaN/Inf i brak przelaczenia na CPU.
6. Vocos dekoduje mel obu profili. Lokalny translator laduje checkpoint; zdalny
   Qwen przechodzi kontrolowany test dopiero po podaniu sekretu.
7. Pelny E2E: upload, YouTube, transkrypcja, tlumaczenie, dubbing, ponowienie
   segmentu, zmiana glosu, eksport, limit sekund, retencja i usuniecie konta.
8. Restart kontenera zachowuje SQLite i konta, ale nie pozostawia osieroconych
   rezerwacji salda. Backup i odtworzenie bazy musza zostac sprawdzone praktycznie.
9. Test obciazenia jednego zadania i kolejki kilku zadan. Raport ma zawierac
   real-time factor ASR/TTS, szczyt pamieci, RAM, temperature i throttling.

Brama produkcyjna: co najmniej godzina materialu o roznych dlugosciach bez OOM,
GPU resetu, pustych transkrypcji i zgubionych koncowek. Jakosc wynikow porownac
bitowo lub metrycznie z zatwierdzonym zestawem referencyjnym z NVIDIA; drobne
roznice numeryczne sa dopuszczalne, roznice tekstu/audio wymagaja odsluchu.

## 5. Kolejnosc wdrozenia i rollback

1. Zbudowac obraz i wykonac smoke bez wystawiania portow publicznych.
2. Uruchomic E2E na LAN, zapisac wersje obrazu, digest, `pip freeze`, hash modeli
   i wyniki benchmarku.
3. Dopiero po akceptacji wlaczyc Caddy, DNS i certyfikat.
4. Wdrozenie oznaczyc tagiem, np. `nupicai-rocm10-YYYYMMDD`.
5. Zachowac poprzedni digest obrazu oraz backup `runtime/`. Rollback polega na
   zmianie jednego taga/digestu w `compose.yaml` i `docker compose up -d`.
6. Migracje SQLite musza byc wstecznie zgodne albo przed migracja wymagaja kopii
   i osobnej procedury odtworzenia.

## 6. Kroki manualne dla wlasciciela

### Sprzet i Fedora

1. Zaktualizuj BIOS/UEFI GMKtec i Fedore, potem wykonaj restart.
2. Sprawdz:

   ```bash
   uname -r
   lspci -nnk | grep -A3 -E 'VGA|Display'
   ls -l /dev/kfd /dev/dri/renderD*
   ```

   Kernel ma byc co najmniej `6.18.4`, sterownik ma byc `amdgpu`.
3. Ustaw rozsadny limit pamieci GPUVM/TTM zgodnie z iloscia RAM i dokumentacja
   AMD. Dla maszyny 128 GB punktem startowym moze byc 64-96 GB, ale pozostaw RAM
   dla CPU, ffmpeg, ASR, dwoch TTS i cache. Po zmianie wykonaj restart i sprawdz
   pamiec widziana przez ROCm. Nie wpisuj starych parametrow GTT z poradnikow dla
   ROCm 6.x bez potwierdzenia, ze ROCm 10 ich wymaga.

### Docker

1. Zainstaluj Docker Engine i plugin Compose z oficjalnego repozytorium Docker
   dla Fedory 43, wlacz usluge i sprawdz `docker run hello-world`.
2. Dodaj operatora do grupy `docker` tylko jezeli akceptujesz, ze daje ona
   uprawnienia zblizone do roota. Po zmianie wyloguj i zaloguj sesje.
3. Sprawdz dostep GPU najpierw w prostym kontenerze ROCm, zanim zbudujesz NupicAI.

### Konfiguracja aplikacji

1. Skopiuj caly katalog `dubbing/` wraz z `models/`, ale bez starego `.env`,
   SQLite i danych uzytkownikow, chyba ze jest to kontrolowana migracja.
2. Utworz `.env` z prawdziwymi: domena, CORS, allowed hosts, admin e-mail,
   bezpieczne cookies, Resend i klucz zdalnego translatora. Ustaw `chmod 600 .env`.
3. Ustaw DNS domeny na serwer i otworz w `firewalld` tylko 80/443. Port FastAPI
   nie powinien byc publiczny.
4. Zapewnij miejsce na dane robocze, modele, backupy i logi. Modele zajmuja teraz
   ok. 8 GB, a pliki uzytkownikow wymagaja osobnego zapasu.
5. Skonfiguruj codzienny backup `runtime/` poza tym komputerem i okresowo testuj
   odtworzenie, nie tylko samo tworzenie kopii.
6. Po otrzymaniu obrazu od agenta wykonaj kolejno:

   ```bash
   docker compose build --pull
   docker compose run --rm nupicai python deploy/rocm10/rocm_smoke.py
   docker compose run --rm nupicai python deploy/rocm10/parakeet_smoke.py
   docker compose up -d
   docker compose ps
   docker compose logs -f nupicai
   ```

7. Nie publikuj uslugi, dopoki `/ready` nie jest zdrowe i caly test E2E nie
   przejdzie na GPU.

## 7. Decyzje po pierwszym benchmarku

- Jezeli Parakeet dziala po wylaczeniu CUDA Graph, zachowac model i NeMo.
- Jezeli NeMo 2.7.3 nie wspolpracuje z PyTorch 2.12, najpierw przetestowac nowsze
  NeMo z tym samym `.nemo`. Nie obnizac losowo pakietow w obrazie produkcyjnym.
- Jezeli problem dotyczy tylko TDT decoder, dopuszczalny jest osobny proces ASR
  CPU jako awaryjny tryb administracyjny, ale nie jako niewidoczny fallback.
- Jezeli wymuszony backend SDPA zawodzi, wlaczyc backend math/manual i zmierzyc
  predkosc. Dopiero potem testowac CK/AOTriton.
- Jezeli ROCm 10 powoduje regresje nieusuwalna w aplikacji, rollbackowac obraz i
  wykonac kontrolny test na oficjalnym ROCm 7.2/PyTorch 2.9. Nie mieszac bibliotek
  ROCm 7 i ROCm 10 w jednym obrazie.

## 8. Kryterium zakonczenia pracy agenta

Praca jest zakonczona dopiero, gdy:

- `docker compose up -d` uruchamia calosc na czystym Fedora 43;
- jeden `compose.yaml` obejmuje aplikacje, TLS i trwale dane;
- wszystkie modele sa lokalne i znalezione przez `check_production.py`;
- ASR i oba TTS faktycznie wykonuja obliczenia na `gfx1151`;
- workaround Parakeet ma automatyczny test regresyjny;
- testy integralnosci pipeline i pelny E2E przechodza;
- istnieje raport wydajnosci, backup/restore i sprawdzony rollback;
- dokumentacja nie zawiera sekretow ani kluczy API.
