#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

PYTHON_BIN="${WEGORZ_PYTHON:-python}"
"$PYTHON_BIN" check_production.py
exec "$PYTHON_BIN" server.py
