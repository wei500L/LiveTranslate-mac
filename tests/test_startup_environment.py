from pathlib import Path


READY_MARKER = ".venv\\.livetranslate-ready"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_source_launcher_rejects_an_incomplete_environment():
    launcher = _read("start.bat")

    assert READY_MARKER in launcher
    assert "setup is incomplete" in launcher


def test_install_and_update_only_mark_verified_environments_ready():
    installer = _read("install.ps1")
    updater = _read("update.bat")

    assert READY_MARKER in installer
    assert installer.rindex("set-content -literalpath $readymarker") > installer.rindex(
        "pip check"
    )
    assert updater.rindex(f'> "{READY_MARKER}" echo') > updater.rindex("pip check")


def test_portable_launcher_repairs_interrupted_bootstraps():
    builder = _read("build_release.ps1")

    assert READY_MARKER in builder
    assert "--allow-existing" in builder
    assert "pip check --python $py" in builder
    assert builder.rindex("set-content -literalpath $ready") > builder.rindex(
        "pip check --python $py"
    )


def test_mac_install_and_update_only_mark_verified_environments_ready():
    installer = _read("install.sh")
    updater = _read("update.sh")

    assert ".livetranslate-ready" in installer
    assert installer.index('rm -f "$ready_marker"') < installer.index("pip install")
    assert installer.rindex("printf 'ready") > installer.rindex("pip check")
    assert updater.index('rm -f "$ready_marker"') < updater.index("pip install")
    assert updater.rindex("printf 'ready") > updater.rindex("pip check")


def test_mac_source_launcher_rejects_an_incomplete_environment():
    launcher = _read("start.sh")
    assert ".livetranslate-ready" in launcher
    assert "setup is incomplete" in launcher


def test_mac_installer_rejects_rosetta_and_unsupported_python():
    installer = _read("install.sh")
    assert "uname -m" in installer
    assert "platform.machine()" in installer
    assert "(3, 10)" in installer
    assert "(3, 12)" in installer
