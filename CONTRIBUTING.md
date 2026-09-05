# Wspolna praca nad NupicAI

## Podzial odpowiedzialnosci

Repozytorium przechowuje wspolny kod produktu:

- backend FastAPI, autoryzacje, limity i platnosci;
- frontend i jego gotowy eksport `parakeet-ui/out`;
- runtime ASR/TTS oraz kompatybilne formaty checkpointow;
- testy, migracje i ogolna dokumentacje wdrozenia;
- bazowy `compose.yaml` i Dockerfile, gdy zostana dodane.

Poza repozytorium pozostaja elementy konkretnej maszyny:

- `.env`, klucze API i sekrety operatora platnosci;
- `compose.override.yaml` z urzadzeniami ROCm, mountami i lokalnymi sciezkami;
- `deploy/local/` z ustawieniami hosta;
- `runtime/`, SQLite, pliki uzytkownikow i backupy;
- modele, checkpointy i duze pliki binarne.

Kod nie powinien zawierac na stale sciezek `/home/...`, `/srv/...`, adresow hosta
ani ustawien konkretnego GPU. Takie wartosci trafiaja do zmiennych srodowiskowych
lub ignorowanego pliku override.

## Codzienny workflow

Nie rozwijamy funkcji bezposrednio na branchu produkcyjnym. Przyklad dla
platnosci:

```bash
git switch master
git pull --ff-only origin master
git switch -c feature/payments

# zmiany i testy
git add server.py auth_store.py parakeet-ui/src tests
git commit -m "feat: add payment checkout and webhooks"
git push -u origin feature/payments
```

Nastepnie brat otwiera Pull Request do `master`. Po przegladzie i testach zmiana
jest scalana. Twoja maszyna pobiera ja przez:

```bash
git switch master
git pull --ff-only origin master
```

Na serwerze produkcyjnym nalezy tylko wdrazac zatwierdzony `master`. Dobrze miec
drugi katalog roboczy do programowania, zamiast edytowac `/srv/nupicai` podczas
dzialania uslugi.

## Wymagania przed Pull Request

```bash
python -m pytest tests/test_pipeline_integrity.py -q
python -m py_compile server.py tts/tts_daemon.py tts/wegorz_tts_model.py
cd parakeet-ui
npm run build
```

Po zmianie frontendu commitujemy zarowno `parakeet-ui/src`, jak i wygenerowany
`parakeet-ui/out`, poniewaz backend serwuje statyczny eksport.

Zmiana schematu SQLite musi byc migracja zgodna wstecznie i miec test. Przed
wdrozeniem migracji wykonujemy `deploy/backup_runtime.py`.

## Platnosci i webhooki

Do wspolnego repo trafiaja adapter operatora, endpoint webhooka, idempotencja,
migracja bazy, UI oraz testy. Sekrety (`STRIPE_SECRET_KEY`, secret webhooka itp.)
sa ustawiane osobno w `.env` kazdego srodowiska. Testy korzystaja z atrap lub
trybu sandbox i nigdy nie wymagaja produkcyjnego klucza.

## Aktualizacja produkcji

```bash
cd /srv/nupicai
python deploy/backup_runtime.py --out-dir /srv/nupicai-backups
git pull --ff-only origin master
python check_production.py --strict
docker compose build nupicai
docker compose up -d nupicai
docker compose logs --tail=200 nupicai
```

Nie uzywamy `git reset --hard` do aktualizacji serwera i nie wykonujemy
`docker compose down -v`. Gdy `git status --short` na produkcji nie jest pusty,
najpierw wyjasniamy lokalne zmiany.
