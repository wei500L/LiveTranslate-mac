#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
READY_MARKER="$VENV_DIR/.livetranslate-ready"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "install.sh is for macOS; use install.ps1 on Windows." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "LiveTranslate macOS builds require native arm64 Python (Rosetta is unsupported)." >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import platform, sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 12) and platform.machine() == "arm64"))'; then
  echo "Python 3.10-3.12 arm64 is required. Install a native build and set PYTHON_BIN." >&2
  exit 1
fi

rm -f "$READY_MARKER"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-mac.txt"
"$VENV_DIR/bin/python" -m pip install "yasbd-lib>=0.15,<1.0"
"$VENV_DIR/bin/python" -m pip check
printf 'ready\n' > "$READY_MARKER"
echo "Environment ready: $VENV_DIR"
