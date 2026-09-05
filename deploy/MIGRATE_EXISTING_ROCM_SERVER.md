# Aktualizacja istniejacego serwera ROCm z GitHuba

Ta instrukcja jest dla operatora serwera, na ktorym NupicAI juz dziala. Aktualizacja
nie instaluje ponownie Fedory, ROCm, Dockera ani sterownikow. Zachowuje lokalne
modele, konta, pliki uzytkownikow i konfiguracje konkretnej maszyny.

Repozytorium:

```text
git@github.com:Luksonini/NupicAI.git
```

## Co pozostaje lokalne

Git celowo nie zawiera ponizszych elementow. Nie wolno ich usuwac ani nadpisywac
wersja z innego serwera:

```text
.env
.env.local
.env.production.local
compose.yaml                    # obecnie lokalny plik wdrozenia ROCm
compose.override.yaml
admin_config.json               # tylko jesli istnieje w starszej wersji
parakeet_config.json
deploy/local/
models/
runtime/
tools/deno/deno                 # lokalny plik wykonywalny, jesli jest uzywany
```

Wolumeny Dockera, dane Caddy i zewnetrzne katalogi wskazane w Compose pozostaja
na swoim miejscu. Nigdy nie wykonuj `docker compose down -v` podczas aktualizacji.

## 1. Kontrola starego wdrozenia

Ponizsze polecenia zakladaja katalog `/srv/nupicai`. Jezeli wdrozenie jest w
innym miejscu, zmien tylko te sciezke.

```bash
cd /srv/nupicai
docker compose ps
git status --short 2>/dev/null || true
find . -maxdepth 2 -type f \
  \( -name '.env*' -o -name 'compose*.yml' -o -name 'compose*.yaml' \
     -o -name 'admin_config.json' -o -name 'parakeet_config.json' \) -print
```

Zapisz takze liste mountow i wolumenow, zanim zatrzymasz aplikacje:

```bash
docker compose config > /tmp/nupicai-compose-before-update.yaml
docker compose ps -q | xargs -r docker inspect \
  --format '{{.Name}} {{range .Mounts}}{{.Type}}:{{.Source}}->{{.Destination}} {{end}}'
```

## 2. Backup

```bash
sudo mkdir -p /srv/nupicai-backups
cd /srv/nupicai
docker compose stop
python deploy/backup_runtime.py --out-dir /srv/nupicai-backups || true
sudo tar --xattrs --acls -cpf \
  /srv/nupicai-backups/local-config-before-git.tar \
  .env compose.yaml compose.override.yaml admin_config.json \
  parakeet_config.json deploy/local 2>/dev/null || true
```

Nie pakujemy tutaj `models/` i `runtime/`, poniewaz moga byc duze. Zostana
zachowane przez zmiane nazwy calego starego katalogu. Baza SQLite powinna miec
dodatkowo kopie wykonana przez `backup_runtime.py`.

## 3. Klon obok starej wersji

```bash
cd /srv
sudo mv nupicai nupicai-before-git
sudo git clone git@github.com:Luksonini/NupicAI.git nupicai
sudo chown -R "$(id -u):$(id -g)" /srv/nupicai
cd /srv/nupicai
```

Katalog docelowy nadal nazywa sie `/srv/nupicai`. Dzieki temu lokalne sciezki i
nazwa projektu Compose nie powinny sie zmienic.

## 4. Przywrocenie konfiguracji i danych

Najpierw przywroc pliki konfiguracyjne. `cp -a` zachowuje uprawnienia:

```bash
cd /srv/nupicai
for path in \
  .env .env.local .env.production.local \
  compose.yaml compose.override.yaml compose.override.yml \
  admin_config.json parakeet_config.json
do
  if [ -e "/srv/nupicai-before-git/$path" ]; then
    cp -a "/srv/nupicai-before-git/$path" "$path"
  fi
done

for dir in deploy/local models runtime; do
  if [ -d "/srv/nupicai-before-git/$dir" ]; then
    mkdir -p "$dir"
    rsync -a "/srv/nupicai-before-git/$dir/" "$dir/"
  fi
done

if [ -f /srv/nupicai-before-git/tools/deno/deno ]; then
  install -Dm755 /srv/nupicai-before-git/tools/deno/deno tools/deno/deno
fi
chmod 600 .env 2>/dev/null || true
```

Jezeli stary Compose montuje dodatkowe katalogi, certyfikaty, cookies YouTube lub
pliki spoza `/srv/nupicai`, pozostaw je w oryginalnych lokalizacjach. Sprawdz ich
sciezki w `/tmp/nupicai-compose-before-update.yaml` i `.env`.

## 5. Kontrola przed startem

```bash
cd /srv/nupicai
git status --short
git branch -vv
docker compose config --quiet
python check_production.py --strict
```

`git status --short` powinien byc pusty. Lokalne pliki sa ignorowane, wiec nie
powinny pojawic sie na liscie zmian.

Porownaj Compose przed uruchomieniem:

```bash
docker compose config > /tmp/nupicai-compose-after-update.yaml
diff -u /tmp/nupicai-compose-before-update.yaml \
  /tmp/nupicai-compose-after-update.yaml || true
```

Roznice w obrazie lub kodzie sa oczekiwane, ale urzadzenia ROCm, mounty modeli,
`runtime`, porty i wolumeny musza pozostac obecne.

## 6. Start i test

```bash
cd /srv/nupicai
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=200
```

Po starcie sprawdz kolejno:

1. `/health` i `/ready`;
2. logowanie administratora;
3. transkrypcje krotkiego pliku wraz z ostatnim zdaniem;
4. tlumaczenie PL i EN;
5. jeden dubbing kazdym dostepnym profilem TTS;
6. ponowienie segmentu i finalny eksport;
7. `torch.version.hip` oraz brak niewidocznego fallbacku na CPU.

## 7. Kolejne aktualizacje

Po tej jednorazowej migracji aktualizacja nie wymaga kopiowania katalogow:

```bash
cd /srv/nupicai
python deploy/backup_runtime.py --out-dir /srv/nupicai-backups
git pull --ff-only origin master
docker compose config --quiet
python check_production.py --strict
docker compose up -d --build
docker compose logs --tail=200
```

## 8. Rollback

Najprostszy rollback pierwszej migracji:

```bash
cd /srv
sudo mv nupicai nupicai-failed-update
sudo mv nupicai-before-git nupicai
cd /srv/nupicai
docker compose up -d
```

Nie kasuj `nupicai-before-git`, dopoki nowa wersja nie przejdzie pelnego testu.

