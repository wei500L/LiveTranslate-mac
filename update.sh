#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
READY_MARKER="$VENV_DIR/.livetranslate-ready"
cd "$ROOT_DIR"
git pull
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  exec "$ROOT_DIR/install.sh"
fi
rm -f "$READY_MARKER"
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-mac.txt"
"$VENV_DIR/bin/python" -m pip install "yasbd-lib>=0.15,<1.0"
"$VENV_DIR/bin/python" -m pip check
printf 'ready\n' > "$READY_MARKER"
