# NupicAI Flow for desktop

Native Tauri 2 dictation client for the NupicAI server.

## Included in the first native build

- compact NupicAI recording window with a live microphone meter,
- global `Ctrl+Alt+Space` push-to-talk shortcut,
- hold-to-talk, shortcut toggle and continuous Silero VAD activation modes,
- local speech endpointing with pre-roll, silence skipping and phrase-by-phrase paste,
- native microphone capture through CPAL,
- system-audio capture through PipeWire/PulseAudio on Linux and WASAPI loopback on Windows,
- selectable input device,
- NupicAI login with the session token stored in the operating-system credential store,
- direct upload to `/dictation/transcribe`,
- optional `/dictation/polish` pass,
- clipboard delivery and optional paste into the previously focused application,
- system tray and single-instance behavior.

## Activation modes

- `Hold`: recording lasts while the global shortcut is held.
- `Toggle`: one press starts recording and the next press transcribes it.
- `Auto VAD`: one press starts continuous listening. A bundled Silero VAD keeps 320 ms of pre-roll, ignores silence, and submits each completed phrase after roughly 600 ms of silence. Press the shortcut again to stop.

`Auto VAD` performs phrase-level near-real-time transcription. The current Parakeet TDT server model is full-context, so partial words are not emitted while a phrase is still being spoken.

The account password is never saved. The session token is stored by Secret Service on
Linux, Windows Credential Manager on Windows and Keychain on macOS.

## Ubuntu 24.04 build

Install the native development packages once:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config libglib2.0-dev libgtk-3-dev \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
  libasound2-dev libxdo-dev libdbus-1-dev pulseaudio-utils
```

Run the development build:

```bash
cd nupic-flow-tauri
cargo run -p nupic-flow
```

On an already configured development machine, the shortest command is:

```bash
./run.sh
```

Build a release binary:

```bash
cd nupic-flow-tauri
cargo build --release -p nupic-flow
```

The binary is written to `target/release/nupic-flow`. Packaging as AppImage, DEB, RPM,
MSI or NSIS is the next release step after behavior is validated on Linux and Windows.

## Fedora build dependencies

```bash
sudo dnf install -y \
  gcc gcc-c++ make pkgconf-pkg-config glib2-devel gtk3-devel \
  webkit2gtk4.1-devel libappindicator-gtk3-devel librsvg2-devel \
  alsa-lib-devel libxdo-devel dbus-devel
sudo dnf install -y pulseaudio-utils
```

## Runtime model

The desktop package contains no ASR or language-model weights. It records mono speech,
converts it to 16 kHz PCM WAV and sends it over HTTPS to the configured NupicAI server.
Python, CUDA and Parakeet remain server-side dependencies only.

The system-audio mode records the default output mix, so it captures audio played by browsers,
media players and meeting applications. Linux uses the default PipeWire/PulseAudio sink monitor;
Windows uses CPAL's native WASAPI loopback path and does not require a Stereo Mix device.

On Wayland, synthetic keyboard input may be rejected by compositor policy. The transcript
is always copied to the clipboard, so delivery still works; production Wayland integration
should use the desktop portal where supported.
