#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_URL="${1:-${NUPICAI_SERVER_URL:-}}"

if [[ -z "$SERVER_URL" ]]; then
  echo "Usage: $0 https://your-nupicai-server.example" >&2
  exit 2
fi
if [[ "$SERVER_URL" != https://* && "${ALLOW_INSECURE_BUILD:-0}" != "1" ]]; then
  echo "Production builds require an HTTPS server URL." >&2
  echo "For a local test only, set ALLOW_INSECURE_BUILD=1." >&2
  exit 2
fi
if ! cargo tauri --version >/dev/null 2>&1; then
  echo "Missing cargo-tauri. Install it with:" >&2
  echo "  cargo install tauri-cli --version '^2' --locked" >&2
  exit 3
fi

cd "$HERE"
NUPICAI_SERVER_URL="${SERVER_URL%/}" cargo tauri build --bundles appimage,deb

APPIMAGE="$(find target/release/bundle/appimage -maxdepth 1 -type f -name '*.AppImage' -print -quit)"
DEB="$(find target/release/bundle/deb -maxdepth 1 -type f -name '*.deb' -print -quit)"
if [[ -z "$APPIMAGE" || -z "$DEB" ]]; then
  echo "Tauri finished, but an AppImage or DEB artifact is missing." >&2
  exit 4
fi

PUBLISH_DIR="$HERE/../runtime/downloads"
mkdir -p "$PUBLISH_DIR"
install -m 0755 "$APPIMAGE" "$PUBLISH_DIR/nupicai-flow-linux-x86_64.AppImage"
install -m 0644 "$DEB" "$PUBLISH_DIR/nupicai-flow-linux-amd64.deb"

echo "Linux packages:"
echo "  $PUBLISH_DIR/nupicai-flow-linux-x86_64.AppImage"
echo "  $PUBLISH_DIR/nupicai-flow-linux-amd64.deb"
