# Aktualizacja istniejacego serwera NupicAI

Ta procedura aktualizuje aplikacje bez ponownej instalacji Fedory, ROCm, Dockera,
sterownikow ani srodowiska. Nowa wersja nie dodaje zaleznosci Python wymagajacych
przebudowy calego hosta, ale obraz aplikacji powinien zostac przebudowany.

## Pliki transportowe

Na serwer trzeba dostarczyc:

```text
nupicai-update.bundle
minidualpath_bins_maskgit_continuity_ep742.pt
```

Bundle zawiera kod, frontend i historie Git. Checkpoint jest osobny, poniewaz
modele sa celowo ignorowane przez Git.

## 1. Backup przed aktualizacja

Na serwerze:

```bash
cd /srv/nupicai
docker compose stop nupicai
python deploy/backup_runtime.py --out-dir /srv/nupicai-backups
cp -a .env /srv/nupicai-backups/env-before-6bef7fc
cp -a admin_config.json /srv/nupicai-backups/admin-config-before-6bef7fc.json 2>/dev/null || true
```

Nie kasowac ani nie nadpisywac:

```text
.env
admin_config.json
runtime/
models/
```

## 2A. Serwer ma repozytorium Git

Skopiuj bundle na serwer, a nastepnie:

```bash
cd /srv/nupicai
git status --short
git fetch /mnt/pendrive/nupicai-update.bundle master
git merge --ff-only FETCH_HEAD
```

Jezeli `git status` pokazuje lokalne zmiany w kodzie, nie wykonuj merge na sile.
Zapisz `git diff` i porownaj zmiany z konfiguracja serwera.

## 2B. Starszy katalog nie ma `.git`

Nie trzeba przenosic danych do nowego wdrozenia. Sklonuj bundle tymczasowo i
zsynchronizuj tylko kod:

```bash
rm -rf /tmp/nupicai-update
git clone /mnt/pendrive/nupicai-update.bundle /tmp/nupicai-update
rsync -a \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='admin_config.json' \
  --exclude='runtime/' \
  --exclude='models/' \
  /tmp/nupicai-update/ /srv/nupicai/
```

Po udanej aktualizacji mozna wlaczyc Git bez ruszania danych produkcyjnych:

```bash
cp -a /tmp/nupicai-update/.git /srv/nupicai/.git
cd /srv/nupicai
git status --short
```

Oczekiwany status jest pusty. Ignorowane `.env`, `runtime/`, `models/` i
`admin_config.json` pozostaja na miejscu.

## 3. Dostarczenie checkpointu

```bash
install -m 0644 \
  /mnt/pendrive/minidualpath_bins_maskgit_continuity_ep742.pt \
  /srv/nupicai/models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt
```

## 4. Kontrola i restart

```bash
cd /srv/nupicai
python check_production.py --strict
docker compose build nupicai
docker compose up -d nupicai
docker compose ps
docker compose logs -f --tail=200 nupicai
```

Nie wykonywac `docker compose down -v`, poniewaz `-v` moze usunac wolumeny z
danymi. Jezeli Compose ma inna nazwe uslugi, zastap `nupicai` jej nazwa.

Po starcie zaloguj sie jako administrator. Ustaw profil
`TDA-MaskGIT continuity` oraz poczatkowo `first=8`, `second=3`, `t_noise=0.12`.

## 5. Szybki rollback

Kod w repozytorium Git:

```bash
cd /srv/nupicai
git switch --detach nupicai-before-production-update
docker compose build nupicai
docker compose up -d nupicai
```

Sam model mozna wycofac bez przebudowy: w panelu administratora wybierz
`StyleEnc128 LSTM` albo `MiniDualPath learned voice`.

Przy problemie z baza zatrzymaj aplikacje i odtworz najnowszy plik SQLite z
`/srv/nupicai-backups`. Nie kopiuj bazy podczas aktywnych zapisow zwyklym `cp`;
do backupu uzywaj `backup_runtime.py`.
