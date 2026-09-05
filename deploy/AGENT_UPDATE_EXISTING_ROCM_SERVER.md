# Runbook dla agenta: migracja istniejacego NupicAI do Git

## Cel

Zaktualizuj dzialajacy serwer NupicAI z prywatnego repozytorium GitHub bez
ponownej konfiguracji Fedory, ROCm, sterownikow, Dockera, urzadzen GPU, TLS,
modeli i danych produkcyjnych.

Repozytorium zrodlowe:

```text
git@github.com:Luksonini/NupicAI.git
```

Zakladana sciezka produkcyjna to `/srv/nupicai`, ale agent ma najpierw ustalic
faktyczna sciezke. Nie wolno przyjmowac jej bez weryfikacji.

## Twarde ograniczenia

1. Nie wykonuj `git reset --hard`, `git clean -fd`, `docker compose down -v` ani
   kasowania starego katalogu.
2. Nie kopiuj sekretow do Git i nie wypisuj ich w logu lub odpowiedzi.
3. Nie przebudowuj konfiguracji ROCm, jezeli obecna przechodzi smoke test.
4. Nie zmieniaj hostowych sterownikow, kernela, BIOS ani podzialu pamieci GPU.
5. Nie modyfikuj checkpointow i nie konwertuj ich automatycznie.
6. Nie uruchamiaj migracji bazy bez poprawnego backupu SQLite.
7. Zachowaj jeden worker aplikacji, jezeli obecny runtime opiera kolejke GPU i
   cache modeli na pamieci procesu.

## Faza 1: inwentaryzacja tylko do odczytu

Zbierz i zapisz bez sekretow:

```bash
pwd
git status --short 2>/dev/null || true
git remote -v 2>/dev/null || true
docker compose ps
docker compose config > /tmp/nupicai-compose-before-update.yaml
find . -maxdepth 3 -type f \
  \( -name '.env*' -o -name 'compose*.yml' -o -name 'compose*.yaml' \
     -o -name 'admin_config.json' -o -name 'parakeet_config.json' \) -print
du -sh models runtime 2>/dev/null || true
```

Sprawdz wszystkie bind mounty i named volumes. Zidentyfikuj w szczegolnosci:

- `.env` i lokalne pliki Compose;
- `deploy/local/`;
- `models/`;
- `runtime/` i baze SQLite;
- dane Caddy/reverse proxy;
- cookies lub konfiguracje yt-dlp;
- lokalny plik `tools/deno/deno`;
- dodatkowe katalogi wskazane przez bezwzgledne sciezki w `.env`/Compose.

Nie odczytuj wartosci sekretow do odpowiedzi. Wystarczy potwierdzic obecnosc,
uprawnienia i miejsce montowania.

## Faza 2: decyzja o metodzie

### A. Katalog jest juz klonem `Luksonini/NupicAI`

Jezeli `git remote get-url origin` wskazuje wlasciwe repo, worktree nie ma zmian
w kodzie, a konfiguracja jest ignorowana, wykonaj aktualizacje w miejscu:

```bash
git fetch origin
git diff --stat HEAD..origin/master
git merge --ff-only origin/master
```

### B. Stary katalog nie jest tym repozytorium

Preferuj swiezy klon obok starego katalogu zgodnie z
`deploy/MIGRATE_EXISTING_ROCM_SERVER.md`. Nie inicjalizuj Git na slepo w starym
katalogu i nie nadpisuj go checkoutem.

## Faza 3: backup i przeniesienie lokalnego stanu

Przed zapisem zatrzymaj aplikacje i wykonaj backup narzedziem projektu. Zachowaj
stary katalog przez zmiane nazwy. Do nowego klonu przenies tylko lokalny stan:

```text
.env*
compose.yaml
compose.override.yaml lub compose.override.yml
admin_config.json
parakeet_config.json
deploy/local/
models/
runtime/
tools/deno/deno
```

Uwaga: w obecnym repo nie ma sledzonego bazowego `compose.yaml`. Dlatego dzialajacy
plik Compose serwera jest wymaganym lokalnym artefaktem i musi zostac zachowany.

Uzywaj `cp -a` dla malych plikow i `rsync -a` dla katalogow. Nie kopiuj `.git`
ze starego wdrozenia do nowego klonu.

## Faza 4: walidacja bez uruchamiania publicznego ruchu

Wykonaj:

```bash
git status --short
docker compose config --quiet
python check_production.py --strict
```

Nastepnie porownaj znormalizowany Compose sprzed i po migracji. Agent ma jawnie
potwierdzic, ze zachowano:

- `/dev/kfd` oraz odpowiednie `/dev/dri`;
- grupy/uprawnienia GPU i ustawienia SELinux;
- mount `models` tylko do odczytu, jezeli tak bylo;
- trwaly `runtime` i baze;
- porty, proxy, certyfikaty i healthcheck;
- wszystkie zmienne srodowiskowe wymagane przez `check_production.py`.

Nie pokazuj wartosci `.env`.

## Faza 5: start i bramy akceptacyjne

Uruchom kontenery bez kasowania wolumenow. Sprawdz logi i testy E2E wymienione w
instrukcji manualnej. Aktualizacja jest udana dopiero, gdy:

1. kontenery sa zdrowe;
2. ASR i TTS faktycznie korzystaja z ROCm;
3. obecne sa wszystkie trzy checkpointy TTS oraz wymagany bank glosow;
4. transkrypcja zachowuje koniec materialu;
5. PL i EN przechodza przez tlumaczenie oraz dubbing;
6. konta, saldo i stare zadania pozostaly dostepne;
7. restart nie usuwa danych;
8. agent podaje commit wdrozonego `master`, ale nie podaje sekretow.

## Faza 6: rollback

Przy bledzie zatrzymaj nowa wersje i przywroc stary katalog pod ta sama sciezka.
Nie probuj naprawiac dzialajacego starego wdrozenia przez kopiowanie pojedynczych
plikow z nowej wersji. Zachowaj logi nowej wersji do diagnozy.

## Raport koncowy agenta

Raport ma zawierac:

- wdrozony commit i branch;
- sciezke wdrozenia;
- liste zachowanych kategorii konfiguracji bez wartosci sekretow;
- wynik `check_production.py --strict`;
- status kontenerow i testu ROCm;
- wyniki smoke ASR/TTS/PL/EN;
- wykonany backup i instrukcje rollbacku;
- wykryte roznice Compose wymagajace decyzji operatora.

