# NupicAI Flow for desktop

Native Tauri 2 dictation client for the NupicAI server.

## Packaging status

The application source and reproducible packaging workflow are ready. Installer
binaries are not stored in Git and are not present in a fresh clone. A release
build must be given the public HTTPS URL of the NupicAI backend.

Expected artifacts:

```text
nupicai-flow-linux-x86_64.AppImage
nupicai-flow-linux-amd64.deb
nupicai-flow-windows-x86_64.exe
```

The website serves files copied to the ignored `../runtime/downloads/` directory.
Until an artifact exists there, its download button is shown as unavailable.

## Included in the first native build

- compact NupicAI recording window with a live microphone meter,
- independent global shortcuts for hold-to-talk, toggle and continuous Silero VAD,
- local speech endpointing with pre-roll, silence skipping and phrase-by-phrase paste,
- native microphone capture through CPAL,
- system-audio capture through PipeWire/PulseAudio on Linux and WASAPI loopback on Windows,
- selectable input device,
- live input diagnostics and a one-click microphone/system-audio switch,
- Polish and English UI selected automatically from the operating-system locale,
- NupicAI login with the session token stored in the operating-system credential store,
- direct upload to `/dictation/transcribe`,
- optional `/dictation/polish` pass,
- clipboard delivery and optional paste into the previously focused application,
- system tray and single-instance behavior.

## Activation modes

- `Hold` (`Ctrl+Alt+Space` by default): recording lasts while its shortcut is held.
- `Toggle` (`Ctrl+Alt+D`): one press starts recording and the next press transcribes it.
- `Auto VAD` (`Ctrl+Alt+V`): one press starts continuous listening. A bundled Silero VAD keeps roughly 500 ms of pre-roll, ignores silence, and submits each completed phrase after roughly 600 ms of silence. Continuous speech is flushed every six seconds with an overlap that the UI reconciles. Press the shortcut again to stop.

All three shortcuts can be changed independently. Existing installations automatically
migrate their old single shortcut to the hold-to-talk action.

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

Install the Tauri packaging CLI once:

```bash
cargo install tauri-cli --version '^2' --locked
```

Build Linux installers for a production server:

```bash
cd nupic-flow-tauri
./build-linux.sh https://nupicai.example.com
```

This creates an AppImage and DEB. The AppImage is copied to the website's ignored
`runtime/downloads` directory and immediately becomes available through the Linux
download button.

Build the Windows installer on a Windows machine with Rust, Microsoft C++ Build Tools,
WebView2 and `cargo-tauri` installed:

```powershell
.\build-windows.ps1 https://nupicai.example.com
```

The NSIS `.exe` is copied to `runtime\downloads` and activates the Windows button.
Do not publish packages built with the default `127.0.0.1` development address.

## GitHub build and release

Set the repository Actions variable `NUPICAI_SERVER_URL` to the public HTTPS
backend address. A manual run of `Desktop installers` produces downloadable CI
artifacts. Pushing a version tag builds both platforms and creates a release:

```bash
git tag flow-v0.1.0
git push origin flow-v0.1.0
```

The repository is private, so GitHub release links are not suitable as public
website links without authentication. Download the three release artifacts and
place them in the production server's `runtime/downloads/` directory. The backend
then exposes stable `/downloads/linux` and `/downloads/windows` URLs.

Before publishing, verify the embedded default URL, log in using a test account,
test microphone and system-audio capture, and confirm paste behavior on each OS.

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
