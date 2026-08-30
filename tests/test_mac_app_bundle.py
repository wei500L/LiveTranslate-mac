"""build_mac_app.sh must produce a launchable bundle (macOS packaging).

LaunchServices rejects shell-script executables with error -10669, so the
bundle's CFBundleExecutable is a compiled Mach-O launcher that execs the
venv python. If someone "simplifies" the script back to a bash launcher, the
app becomes un-launchable from Launchpad — this test pins the requirements
the bundle must meet: a Mach-O executable, a parseable Info.plist with the
TCC usage strings, a real .icns, and a project path baked into the launcher.
"""

import os
import plistlib
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="build_mac_app.sh produces a macOS .app bundle",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build(tmp_path):
    pytest.importorskip("PyQt6", reason="icon generation needs Qt")
    venv_python = os.path.join(ROOT, ".venv", "bin", "python")
    if not os.path.isfile(venv_python):
        pytest.skip("project venv not present")
    for tool in ("cc", "iconutil"):
        if shutil.which(tool) is None:
            pytest.skip(f"{tool} not available")

    env = dict(os.environ, OUT_DIR=str(tmp_path))
    result = subprocess.run(
        ["bash", os.path.join(ROOT, "build_mac_app.sh")],
        cwd=ROOT,  # the icon step imports platform_fonts from the project dir
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return tmp_path / "LiveTranslate.app"


def test_bundle_is_launchable(tmp_path):
    app = _build(tmp_path)
    assert app.is_dir()

    with open(app / "Contents" / "Info.plist", "rb") as f:
        plist = plistlib.load(f)
    assert plist["CFBundleExecutable"] == "LiveTranslate"
    assert plist["CFBundlePackageType"] == "APPL"
    # Dock visibility is the app's own runtime decision (set_dock_visible)
    assert "LSUIElement" not in plist
    # Without these, a first-launch TCC prompt cannot explain itself
    assert "NSMicrophoneUsageDescription" in plist

    # LaunchServices refuses non-Mach-O executables with error -10669
    exe = app / "Contents" / "MacOS" / "LiveTranslate"
    assert os.access(exe, os.X_OK)
    with open(exe, "rb") as f:
        assert f.read(4) == b"\xcf\xfa\xed\xfe", \
            "CFBundleExecutable must be a Mach-O binary, not a script"

    icns = app / "Contents" / "Resources" / "AppIcon.icns"
    with open(icns, "rb") as f:
        assert f.read(4) == b"icns"


def test_launchers_point_at_the_project(tmp_path):
    app = _build(tmp_path)
    # The debug shell launcher carries the baked project dir
    sh = app / "Contents" / "MacOS" / "LiveTranslate.sh"
    assert ROOT in sh.read_text(encoding="utf-8")


def test_rebuild_is_deterministic(tmp_path):
    """A rebuilt bundle must keep the same executable bytes, or macOS TCC
    (microphone permission) re-prompts after every rebuild."""
    exe1 = _build(tmp_path / "a") / "Contents" / "MacOS" / "LiveTranslate"
    exe2 = _build(tmp_path / "b") / "Contents" / "MacOS" / "LiveTranslate"
    assert exe1.read_bytes() == exe2.read_bytes()
