# NupicAI: dostep do repo i aktualizacja serwera brata

Ta instrukcja laczy istniejacy serwer Fedora 43/ROCm z prywatnym repozytorium
NupicAI bez ponownej instalacji sterownikow, ROCm ani modeli. Wykonuj polecenia
jako zwykly uzytkownik, z wyjatkiem polecen wyraznie poprzedzonych `sudo`.

## 1. Nadanie dostepu przez wlasciciela repo

Wlasciciel repozytorium otwiera:

1. `https://github.com/Luksonini/NupicAI/settings/access`
2. `Collaborators` -> `Add people`.
3. Wpisuje konto GitHub brata i wysyla zaproszenie.

Nie wysylaj bratu swojego klucza prywatnego SSH ani zawartosci pliku `.env`.

## 2. Przyjecie zaproszenia

Brat loguje sie na swoje konto GitHub i przyjmuje zaproszenie:

- z wiadomosci e-mail od GitHuba; albo
- na `https://github.com/notifications`.

Po zaakceptowaniu powinien widziec prywatne repozytorium pod adresem:

`https://github.com/Luksonini/NupicAI`

## 3. Wlasny klucz SSH brata

Najpierw sprawdz, czy komputer ma juz klucz:

```bash
ls -la ~/.ssh
```

Jesli nie ma plikow `id_ed25519` i `id_ed25519.pub`, utworz nowy klucz:

```bash
ssh-keygen -t ed25519 -C "ADRES_EMAIL_KONTA_GITHUB"
```

Zaakceptuj domyslna sciezke `~/.ssh/id_ed25519`. Haslo do klucza jest zalecane.
Nastepnie uruchom agenta i dodaj klucz:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Skopiuj wynik ostatniego polecenia, zaczynajacy sie od `ssh-ed25519`. Na koncie
GitHub brata otworz `Settings` -> `SSH and GPG keys` -> `New SSH key`, wpisz np.
`Nupic AMD ROCm` i wklej klucz publiczny.

Nigdy nie kopiuj ani nie wysylaj pliku `~/.ssh/id_ed25519`. Do GitHuba trafia
wylacznie plik z koncowka `.pub`.

Sprawdz dostep:

```bash
ssh -T git@github.com
```

Przy pierwszym polaczeniu potwierdz host wpisujac `yes`. Prawidlowy wynik zawiera
nazwe konta oraz informacje o udanym uwierzytelnieniu.

## 4. Kopia dzialajacego wdrozenia

Zakladamy, ze obecna instalacja jest w `/srv/nupicai`:

```bash
cd /srv/nupicai
docker compose ps
docker compose stop
cd /srv
sudo mv nupicai nupicai-before-git
sudo mkdir nupicai
sudo chown "$(id -u):$(id -g)" nupicai
```

Nie uzywaj `docker compose down -v`, bo moze usunac wolumeny z danymi.

## 5. Sklonowanie repozytorium

Klonuj jako zwykly uzytkownik, aby Git korzystal z jego klucza SSH:

```bash
cd /srv
git clone git@github.com:Luksonini/NupicAI.git nupicai
cd /srv/nupicai
git status
```

Nie uzywaj `sudo git clone`.

## 6. Przeniesienie lokalnej konfiguracji ROCm

Repo nie przechowuje sekretow, danych runtime ani ustawien konkretnego serwera.
Przenies tylko lokalne pliki konfiguracyjne, ktore faktycznie istnieja:

```bash
cd /srv/nupicai

for file in \
  .env .env.local .env.production.local \
  compose.yaml compose.override.yaml compose.override.yml \
  admin_config.json parakeet_config.json
do
  if [ -e "/srv/nupicai-before-git/$file" ]; then
    cp -a "/srv/nupicai-before-git/$file" "$file"
  fi
done

for dir in runtime deploy/local
do
  if [ -d "/srv/nupicai-before-git/$dir" ]; then
    mkdir -p "$(dirname "$dir")"
    cp -a "/srv/nupicai-before-git/$dir" "$(dirname "$dir")/"
  fi
done
```

Jesli lokalna konfiguracja ROCm zawiera dodatkowe pliki, najpierw porownaj je z
nowym repo. Nie kopiuj starego `server.py`, katalogow `tts/`, `translate/`,
`parakeet-ui/` ani innych zrodel, bo nadpisalyby aktualna wersje aplikacji.

Sprawdz, czy sekrety nadal nie sa sledzone przez Git:

```bash
git status --short
git check-ignore -v .env runtime 2>/dev/null || true
```

## 7. Modele z pendrive'a

Modele nie sa przechowywane w GitHubie. Najbezpieczniej przekazac bratu aktualny
caly katalog `models/` z komputera, na ktorym dziala zatwierdzona wersja. Nie
zakladaj, ze jego stare checkpointy sa zgodne z aktualnym kodem.

Pendrive powinien miec exFAT, NTFS albo ext4. FAT32 nie obsluguje plikow powyzej
4 GB. Na komputerze wlasciciela najpierw znajdz punkt montowania pendrive'a i
skopiuj aktualny katalog modeli:

```bash
findmnt -t exfat,ntfs,ntfs3,ext4
USB="/media/$USER/NUPICAI"  # zastap rzeczywistym punktem montowania
cd /home/rizos/Downloads/SalmonTTS2/test_paraqueet/wegorz_dubbingTTS/stronka/dubbing
mkdir -p "$USB/models"
rsync -avP --partial models/ "$USB/models/"
(cd "$USB" && sha256sum -c models/MODEL_MANIFEST.sha256)
sync
```

W chwili przygotowania tej instrukcji manifest obejmuje Parakeet, translator,
Vocos, aktualny bank glosow oraz checkpointy `mini_dualpath_learnedvoice.pt` i
`minidualpath_bins_maskgit_continuity_ep742.pt`.

Na serwerze brata, po podlaczeniu pendrive'a przykladowo pod
`/run/media/$USER/NUPICAI`, wykonaj:

```bash
cd /srv/nupicai
mkdir -p models
rsync -avP --partial "/run/media/$USER/NUPICAI/models/" models/
```

Jesli na pendrivie jest manifest sum kontrolnych, zweryfikuj modele:

```bash
sha256sum -c models/MODEL_MANIFEST.sha256
```

Alternatywnie mozna przekazac archiwum `nupicai-models-ep742.tar.zst` razem z
plikiem `.sha256`; szczegoly zawiera `deploy/MODEL_DISTRIBUTION.md`.

Przed wyjazdem warto porownac, ktory checkpoint jest obecnie produkcyjny, i
wygenerowac nowy manifest. Sama nazwa starego pliku nie gwarantuje zgodnosci.

## 8. Kontrola i uruchomienie

```bash
cd /srv/nupicai
docker compose config --quiet
python check_production.py --strict
docker compose up -d --build
docker compose ps
docker compose logs --tail=200
```

Po starcie sprawdz:

- `/health` i `/ready`;
- logowanie uzytkownika i administratora;
- transkrypcje, zwlaszcza ostatnie zdanie nagrania;
- tlumaczenie PL/EN;
- Voice Studio i dubbing;
- w logach Pythona `torch.version.hip` oraz brak niezamierzonego CPU fallback.

Jesli migracja sie nie powiedzie, zatrzymaj nowy projekt i przywroc katalog
`/srv/nupicai-before-git`. Pelny runbook znajduje sie w
`deploy/MIGRATE_EXISTING_ROCM_SERVER.md`.

## 9. NupicAI Flow

Klient desktopowy nie potrzebuje lokalnych modeli ASR, ale musi laczyc sie z
dzialajacym backendem NupicAI. Ze zrodel uruchamia sie go w sesji graficznej:

```bash
cd /srv/nupicai/nupic-flow-tauri
./run.sh
```

Adres backendu ustawia sie w ustawieniach Flow. Domyslnie jest to
`http://127.0.0.1:8765`; na innym komputerze powinien to byc publiczny adres HTTPS
serwera. Budowanie instalatorow opisuje `nupic-flow-tauri/README.md`.

## 10. Codzienna praca obu osob

Przed rozpoczeciem zmian:

```bash
git switch master
git pull --ff-only
git switch -c nazwa-zmiany
```

Po zmianach:

```bash
git status
git add PLIKI_KTORE_MAJA_TRAFIC_DO_REPO
git commit -m "krotki opis zmiany"
git push -u origin nazwa-zmiany
```

Nastepnie utworz Pull Request na GitHubie. Nie zatwierdzaj `.env`, sekretow,
modeli, nagran uzytkownikow, baz danych ani lokalnych override'ow ROCm.

Aktualizacja serwera po polaczeniu zmian do `master`:

```bash
cd /srv/nupicai
git pull --ff-only
docker compose config --quiet
python check_production.py --strict
docker compose up -d --build
```
