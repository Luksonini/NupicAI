# Dystrybucja modeli NupicAI

Modele nie sa przechowywane w Git. Aktualny katalog `models/` ma okolo 9 GB, a
same checkpointy TTS maja rozmiar nieodpowiedni dla zwyklego repozytorium kodu.
Git powinien przechowywac kod, format modelu i manifest sum kontrolnych, a pliki
binarne powinny byc transportowane osobnym kanalem.

## Zalecana metoda: SSH, Tailscale i rsync

Jezeli oba komputery moga polaczyc sie przez SSH (bezposrednio albo przez
Tailscale), jest to najprostsza i najbardziej odporna metoda. `rsync` wznawia
przerwany transfer i wysyla ponownie tylko zmienione pliki.

Na komputerze zrodlowym:

```bash
cd /sciezka/do/NupicAI
rsync -avP --partial --append-verify \
  models/ brat@ADRES_SERWERA:/srv/nupicai/models/
```

Do wyslania tylko nowego TTS:

```bash
rsync -avP --partial --append-verify \
  models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt \
  brat@ADRES_SERWERA:/srv/nupicai/models/tts/checkpoints/
```

Po transferze na komputerze zrodlowym i docelowym:

```bash
cd /srv/nupicai
sha256sum -c models/MODEL_MANIFEST.sha256
python check_production.py --strict
```

## Alternatywa bez dostepu SSH

Utworz wersjonowane archiwum, policz hash i wyslij je prywatnym magazynem
obiektowym lub zaszyfrowanym dyskiem. Nazwa powinna zawierac wersje zgodna z
tagiem kodu, na przyklad:

```bash
tar -C . -I 'zstd -T0 -8' -cf nupicai-models-ep742.tar.zst models
sha256sum nupicai-models-ep742.tar.zst \
  > nupicai-models-ep742.tar.zst.sha256
```

Na serwerze:

```bash
sha256sum -c nupicai-models-ep742.tar.zst.sha256
tar -I zstd -xf nupicai-models-ep742.tar.zst -C /srv/nupicai
cd /srv/nupicai
sha256sum -c models/MODEL_MANIFEST.sha256
```

Do transferu internetowego preferuj prywatny bucket S3/R2 z krotko waznym,
podpisanym linkiem. Jezeli magazyn nie szyfruje pliku kluczem kontrolowanym przez
was, zaszyfruj archiwum przed wyslaniem, np. narzedziem `age`. Nie publikuj
checkpointow jako publiczne GitHub Releases.

## Wersjonowanie

Kazde wdrozenie modelu powinno miec:

1. tag lub commit kompatybilnego kodu;
2. jednoznaczna nazwe checkpointu;
3. `models/MODEL_MANIFEST.sha256`;
4. krotka notatke o profilu i wymaganych plikach;
5. zachowany poprzedni checkpoint do rollbacku.

Po dodaniu lub wymianie modelu odswiez manifest z katalogu projektu:

```bash
sha256sum \
  models/asr/parakeet-tdt-0.6b-v3.nemo \
  models/translate/wegorz_translator_32k_best.pt \
  models/tts/checkpoints/mini_dualpath_learnedvoice.pt \
  models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt \
  models/tts/checkpoints/styleenc128_lstm.pt \
  models/tts/vocos-mel-24khz/pytorch_model.bin \
  models/tts/voice_banks/selected_top_voices_current.pt \
  > models/MODEL_MANIFEST.sha256
```

Manifest mozna commitowac, poniewaz zawiera tylko nazwy i sumy kontrolne, bez
wag modelu i sekretow.
