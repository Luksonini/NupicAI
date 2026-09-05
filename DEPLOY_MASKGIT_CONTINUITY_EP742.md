# Wdrozenie TDA-MaskGIT continuity (ep0742)

## Co zostalo dodane

- Trzeci profil TTS `maskgit_continuity` oparty o checkpoint `ep0742`.
- Predyktor `MiniDualPathBinsMaskGITDurationPredictor` z dyskretnymi klasami duracji i iteracyjnym dekodowaniem MaskGIT.
- Stan rytmu poprzedniego chunku dla predyktora duracji.
- Dwusekundowa pamiec akustyczna kodowana do tokenu prefix `[1]` przed text encoderem.
- ContextBridge, rytm i pamiec akustyczna sa utrzymywane pomiedzy segmentami tego samego lektora w jednym zadaniu.
- Retry dopasowania czasu cofa wszystkie stany do snapshotu sprzed nieudanej proby.
- Panel administratora zapisuje domyslny model oraz `first pass`, `second pass` i `t_noise`.
- Zwykle ekrany TTS pobieraja te ustawienia z serwera.

Nie jest stosowany akustyczny prefix ramek. Produkcyjne `short_continuity_ms` pozostaje rowne `0`, wiec nie doklejamy audio poprzedniego chunku.

## Pliki do przekazania bratu

Najbezpieczniej przekazac caly katalog `stronka/dubbing`, poniewaz frontend jest statycznym buildem, a runtime zalezy od lokalnych modeli. Przy aktualizacji roznicowej trzeba podmienic:

```text
server.py
check_production.py
README.md
DEPLOY_MASKGIT_CONTINUITY_EP742.md
tts/tts_daemon.py
tts/wegorz_tts_model.py
models/tts/checkpoints/minidualpath_bins_maskgit_continuity_ep742.pt
parakeet-ui/src/components/AdminPanel.tsx
parakeet-ui/src/components/TTSPanel.tsx
parakeet-ui/src/components/TextTTSPanel.tsx
parakeet-ui/src/lib/api.ts
parakeet-ui/src/lib/types.ts
parakeet-ui/out/
```

Plik checkpointu ma okolo 1.9 GB. Nie wystarczy wyslac samego kodu.

## Ustawienie i wycofanie

Po zalogowaniu jako administrator wejdz do panelu administratora i wybierz `TDA-MaskGIT continuity`. Zalecane ustawienia startowe zgodne z dotychczasowym inference:

```text
first pass: 8
second pass: 3
t_noise: 0.12
```

Zmiana jest odwracalna bez kasowania plikow. W panelu wybierz ponownie `StyleEnc128 LSTM` albo `MiniDualPath learned voice`. Ustawienie trafia do prywatnego pliku runtime `admin_config.json` i obowiazuje nowe zadania.

## Zachowanie pamieci

- Stan jest oddzielny dla zadania, profilu modelu i lektora.
- Pierwszy segment resetuje stan; kolejne segmenty tego samego lektora go kontynuuja.
- Zmiana lektora nie miesza stanow.
- Nowe zadanie zaczyna od pustej pamieci.
- Dla profilu continuity nie uzywamy cache gotowych segmentow, poniewaz sam WAV nie odtwarza wewnetrznego stanu potrzebnego kolejnemu segmentowi.
- Dwa starsze profile zachowuja dotychczasowe, niezalezne segmenty.

## Kontrola przed uruchomieniem

```bash
cd /sciezka/do/dubbing
python check_production.py
python -m py_compile server.py tts/tts_daemon.py tts/wegorz_tts_model.py
cd parakeet-ui && npm run build
```

Test lokalny wykonany podczas wdrozenia potwierdzil scisle wczytanie wag oraz synteze dwoch chunkow. Pierwszy chunk nie mial pamieci, a drugi raportowal aktywny stan rytmu i pamiec akustyczna.
