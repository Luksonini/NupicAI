#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Keep JIT and plotting caches out of site-packages and service home directories.
CACHE_DIR="${NUPICAI_CACHE_DIR:-$APP_DIR/runtime/cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$CACHE_DIR/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

PYTHON_BIN="${WEGORZ_PYTHON:-python}"
"$PYTHON_BIN" check_production.py
exec "$PYTHON_BIN" server.py
