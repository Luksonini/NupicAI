# NupicAI Flow desktop releases

Desktop installers are deployment artifacts and stay outside Git. The website
detects them in `runtime/downloads` and enables the matching download button.

Supported filenames:

- Linux: `nupicai-flow-linux-x86_64.AppImage` or `nupicai-flow-linux-x86_64.tar.gz`
- Windows: `nupicai-flow-windows-x86_64.exe` or `nupicai-flow-windows-x86_64.msi`

Use the platform release scripts from `nupic-flow-tauri`:

```bash
./build-linux.sh https://nupicai.example.com
```

```powershell
.\build-windows.ps1 https://nupicai.example.com
```

On the production host:

```bash
mkdir -p runtime/downloads
cp /path/to/package runtime/downloads/nupicai-flow-linux-x86_64.AppImage
```

No server restart is required. `GET /api/desktop-downloads` reports availability,
and `GET /downloads/linux` or `GET /downloads/windows` serves the first matching
artifact.

Before publishing a desktop build, set its default API server to the production
NupicAI URL and test login, `/auth/me`, and `/dictation/transcribe` against that
deployment. The application uses the same account as the web studio and stores
its session token in the operating system keyring.

Silero VAD runs locally through ONNX Runtime. Its bundled model is about 2.3 MB;
the server receives selected speech fragments for transcription, not continuous
silence captured between utterances.

## GitHub Actions

`.github/workflows/desktop-installers.yml` builds both platforms without requiring
a local Windows development machine.

1. In GitHub, open **Settings -> Secrets and variables -> Actions -> Variables**.
2. Add `NUPICAI_SERVER_URL` with the public HTTPS address of the backend.
3. For a test build, open **Actions -> Desktop installers -> Run workflow** and
   provide the same URL. Download both artifacts from the completed run.
4. For a permanent GitHub Release, increment the application version, commit it,
   and push a tag such as `flow-v0.1.0`.

```bash
git tag flow-v0.1.0
git push origin flow-v0.1.0
```

Git tracks application source and shared documentation. `.env`, runtime state,
model checkpoints, local Docker overrides and machine-specific RTX/ROCm settings
remain ignored and must not be committed.
