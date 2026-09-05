# NupicAI Flow desktop prototype

Isolated push-to-talk client for NupicAI. It does not contain ASR model weights. Audio is
sent to the configured NupicAI server and removed from the local temporary directory after
the request finishes.

## Linux prototype

The first prototype targets X11. Hold `Ctrl+Alt+Space`, speak, then release any shortcut
key. The recognized text is pasted into the previously focused application. Under Wayland,
the result is copied to the clipboard because arbitrary synthetic keyboard input is blocked
by design.

```bash
cd nupic-flow-desktop
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

Linux requires `parec` (`pulseaudio-utils` on Fedora and Debian/Ubuntu). It records through
the active PipeWire/PulseAudio compatibility server and does not require PortAudio.

The server URL should include the scheme, for example `https://nupic.example.com` or
`http://127.0.0.1:8765`. Use the same account as on the NupicAI website.

AI text cleanup is optional and disabled by default. With it disabled, no translation-model
request is made: microphone audio goes directly through Parakeet and the transcript is
pasted.

Configuration is stored with mode `0600` in
`~/.config/nupicai-flow/config.json`. The prototype stores an opaque NupicAI session token,
never the account password. A production package should move that token to Secret Service
(Linux) or Windows Credential Manager.

## Packaging roadmap

1. Validate latency, microphone selection and paste reliability on Fedora/X11.
2. Add a signed Linux AppImage and autostart `.desktop` entry.
3. Add a Wayland GlobalShortcuts portal and explicit paste action.
4. Build the Windows variant using the same Python code and WASAPI input.
5. Move credentials to the OS keyring and add signed automatic updates.
