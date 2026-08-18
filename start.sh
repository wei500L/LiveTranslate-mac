#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
READY_MARKER="$VENV_DIR/.livetranslate-ready"
if [[ ! -f "$READY_MARKER" ]]; then
  echo "Setup is incomplete. Run ./install.sh first." >&2
  exit 1
fi
exec "$VENV_DIR/bin/python" "$ROOT_DIR/main.py" "$@"
