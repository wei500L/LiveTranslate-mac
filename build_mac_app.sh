#!/usr/bin/env bash
# Build a double-clickable LiveTranslate.app for macOS.
#
#   ./build_mac_app.sh             build dist/LiveTranslate.app
#   ./build_mac_app.sh --install   build it and copy into /Applications
#
# The bundle's launcher points at THIS project directory (baked in below).
# If you move or rename the project folder, rerun this script.
#
# The app is unsigned. That is fine for local use — Gatekeeper only blocks
# quarantined downloads, and this bundle never leaves your machine. The first
# launch may re-prompt for Microphone / Screen Recording permission because
# TCC attributes those to the .app instead of the terminal you used before.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="LiveTranslate"
BUNDLE_ID="com.livetranslate.app"
VENV_PY="$ROOT_DIR/.venv/bin/python"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
APP_DIR="$OUT_DIR/$APP_NAME.app"

if [[ ! -x "$VENV_PY" ]]; then
  echo "error: $VENV_PY not found. Run ./install.sh first." >&2
  exit 1
fi
for tool in iconutil cc; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: $tool not found (needs macOS with Xcode Command Line Tools)." >&2
    exit 1
  fi
done

# `|| true`: with set -e a missing changelog (grep exit 1) would abort the
# script here, making the date fallback below dead code
VERSION_DATE="$(grep -m1 '^## ' "$ROOT_DIR/i18n/CHANGELOG_en.md" 2>/dev/null \
  | sed 's/^## //' | tr -d '-' || true)"
if [[ -z "$VERSION_DATE" ]]; then
  VERSION_DATE="$(date +%Y%m%d)"
fi
SHORT_VERSION="${VERSION_DATE:0:4}.${VERSION_DATE:4:2}.${VERSION_DATE:6:2}"

echo "Building $APP_DIR (version $SHORT_VERSION)..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

# --- Icon: same design as create_app_icon() in main.py, rendered at icns sizes
ICONSET_DIR="$APP_DIR/Contents/Resources/AppIcon.iconset"
mkdir -p "$ICONSET_DIR"
QT_QPA_PLATFORM=offscreen "$VENV_PY" - "$ICONSET_DIR" <<'PY'
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QGuiApplication, QColor, QFont, QPainter, QPixmap

from platform_fonts import default_mono_font_family

# QPixmap needs a QGuiApplication to exist first
_qapp = QGuiApplication([])
iconset = Path(sys.argv[1])

# Mirrors create_app_icon() in main.py: blue rounded square, white bold "LT".
# main.py itself cannot be imported here — it loads torch and starts the app —
# so keep the two designs in sync manually.
def render(size: int) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(60, 130, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(size / 16, size / 16, size * 14 / 16, size * 14 / 16),
                      size * 12 / 64, size * 12 / 64)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont(default_mono_font_family(), int(size * 28 / 64),
                    QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    return pix

for size in (16, 32, 64, 128, 256, 512, 1024):
    name = f"icon_{size}x{size}.png"
    pix = render(size)
    if size > 512:  # 1024 exists only as @2x
        name = "icon_512x512@2x.png"
    pix.save(str(iconset / name))
    if size <= 512:
        pix.save(str(iconset / f"icon_{size // 2}x{size // 2}@2x.png"))
PY
iconutil -c icns -o "$APP_DIR/Contents/Resources/AppIcon.icns" "$ICONSET_DIR"
rm -rf "$ICONSET_DIR"

# --- Info.plist: written via plistlib so escaping is never a guessing game
"$VENV_PY" - "$APP_DIR/Contents/Info.plist" "$APP_NAME" "$BUNDLE_ID" \
  "$SHORT_VERSION" "$VERSION_DATE" <<'PY'
import plistlib
import sys

path, app_name, bundle_id, short_version, version = sys.argv[1:6]
with open(path, "wb") as f:
    # LSUIElement is deliberately absent: the app manages its own Dock icon
    # via platform_app.set_dock_visible() and the user's "Dock icon" setting.
    plistlib.dump({
        "CFBundleDevelopmentRegion": "zh_CN",
        "CFBundleExecutable": app_name,
        "CFBundleIconFile": "AppIcon",
        "CFBundleIdentifier": bundle_id,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": app_name,
        "CFBundleDisplayName": app_name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": short_version,
        "CFBundleVersion": version,
        "NSHighResolutionCapable": True,
        # TCC prompts shown on first launch if the terminal grant doesn't carry over
        "NSMicrophoneUsageDescription":
            "LiveTranslate captures the microphone for real-time translation.",
        "NSScreenCaptureDescription":
            "LiveTranslate captures system audio for real-time translation.",
    }, f)
PY

# --- Launcher: Finder launches executables with cwd=/, so everything is
# resolved against the project dir baked in at build time.
# LaunchServices (macOS 13+) refuses shell scripts as CFBundleExecutable
# (-10669), so the bundle executable is a tiny Mach-O launcher that execs
# the venv python directly; the bash script is kept for debugging.
LAUNCHER_SH="$APP_DIR/Contents/MacOS/$APP_NAME.sh"
LAUNCHER="$APP_DIR/Contents/MacOS/$APP_NAME"
cat > "$LAUNCHER_SH" <<'EOF'
#!/bin/bash
# Generated by build_mac_app.sh — rerun it after moving the project folder.
PROJECT_DIR="__PROJECT_DIR__"
mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"
LOG="$PROJECT_DIR/logs/app_bundle.log"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/main.py" >>"$LOG" 2>&1
fi
# venv missing or broken: start.sh repairs the environment first
exec "$PROJECT_DIR/start.sh" >>"$LOG" 2>&1
EOF
sed -i '' "s|__PROJECT_DIR__|$ROOT_DIR|g" "$LAUNCHER_SH"
chmod +x "$LAUNCHER_SH"

cat > "$APP_DIR/Contents/MacOS/launcher.c" <<'EOF'
/* Generated by build_mac_app.sh. Execs the venv python with cwd and stdio
 * pointed at the project directory; falls back to start.sh (which repairs
 * the environment) when the venv is missing. */
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void) {
    const char *project_dir = "__PROJECT_DIR__";
    const char *log_path = "__PROJECT_DIR__/logs/app_bundle.log";
    const char *python = "__PROJECT_DIR__/.venv/bin/python";
    const char *repair = "__PROJECT_DIR__/start.sh";

    if (chdir(project_dir) != 0) {
        /* Project folder moved since build_mac_app.sh ran. Fail loudly —
         * a silent exit 127 from a double-click looks like nothing happened. */
        execl("/usr/bin/osascript", "osascript", "-e",
              "display dialog \"LiveTranslate: project folder not found at "
              "__PROJECT_DIR__\" with title \"LiveTranslate\" "
              "buttons {\"OK\"} default button \"OK\" "
              "with icon caution", (char *)0);
        return 127;
    }
    mkdir("__PROJECT_DIR__/logs", 0755);
    int log_fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (log_fd >= 0) {
        dup2(log_fd, 1);
        dup2(log_fd, 2);
        close(log_fd);
    }
    execl(python, "python", "__PROJECT_DIR__/main.py", (char *)0);
    /* exec failed: let start.sh repair the environment */
    execl("/bin/bash", "bash", repair, (char *)0);
    return 127;
}
EOF
sed -i '' "s|__PROJECT_DIR__|$ROOT_DIR|g" "$APP_DIR/Contents/MacOS/launcher.c"
cc "$APP_DIR/Contents/MacOS/launcher.c" -o "$LAUNCHER"
rm "$APP_DIR/Contents/MacOS/launcher.c"

echo "Built $APP_DIR"

if [[ "${1:-}" == "--install" ]]; then
  DEST="/Applications/$APP_NAME.app"
  if pgrep -f "$ROOT_DIR/main.py" >/dev/null 2>&1; then
    echo "warning: LiveTranslate is currently running; the new bundle takes" >&2
    echo "         effect on the next launch (quit the app to replace it now)." >&2
  fi
  echo "Installing to $DEST..."
  rm -rf "$DEST"
  cp -R "$APP_DIR" "$DEST"
  # Refresh LaunchServices so Launchpad/Spotlight don't show a stale entry
  # after the bundle was deleted and recreated
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$DEST" >/dev/null 2>&1 || true
  echo "Installed. Launch with:  open -a $APP_NAME   (or double-click in Launchpad)"
else
  echo "To install into /Applications:  ./build_mac_app.sh --install"
fi
