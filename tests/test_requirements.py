import re
from pathlib import Path

import pytest


INSTALL_ENTRYPOINTS = ("install.ps1", "update.bat", "build_release.ps1")


def _requirement_lines() -> set[str]:
    return {
        line.split("#", 1)[0].strip().lower()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }


def test_funasr_uses_published_dependency_metadata():
    """Install a current FunASR normally instead of copying its dependencies."""
    requirements = _requirement_lines()

    assert {
        "funasr>=1.3.28",
        "hydra-core>=1.3.2",
        "soundfile>=0.12.1",
    } <= requirements
    assert "editdistance-s>=1.0.0" not in requirements


def test_numpy_numba_versions_support_python_312_resolution():
    requirements = _requirement_lines()

    assert "numpy>=1.24.0,<2.3" in requirements
    assert "numba>=0.59.0" in requirements


@pytest.mark.parametrize("path", INSTALL_ENTRYPOINTS)
def test_install_entrypoint_resolves_funasr_dependencies(path: str):
    installer = Path(path).read_text(encoding="utf-8").lower()

    assert "-r requirements.txt" in installer
    assert "--no-deps" not in installer


def test_update_stops_when_dependency_installation_fails():
    updater = Path("update.bat").read_text(encoding="utf-8").lower()
    dependency_block = updater.split("install -r requirements.txt", 1)[1].split(
        'install "yasbd-lib', 1
    )[0]

    assert "exit /b 1" in dependency_block


@pytest.mark.parametrize("path", INSTALL_ENTRYPOINTS)
def test_install_entrypoints_pin_yasbd_and_drop_pysbd(path: str):
    installer = Path(path).read_text(encoding="utf-8").lower()

    assert 'yasbd-lib>=0.15,<1.0' in installer
    assert "install pysbd" not in installer


def test_readmes_do_not_describe_the_removed_editdistance_workaround():
    for path in (Path("README.md"), Path("README_zh.md")):
        text = path.read_text(encoding="utf-8").lower()
        assert "--no-deps" not in text
        assert "editdistance-s" not in text


def test_mac_requirements_use_coreaudio_without_windows_patch():
    text = Path("requirements-mac.txt").read_text(encoding="utf-8").lower()
    assert "pyaudio>=0.2.14" in text
    assert "pyobjc-framework-screencapturekit" in text
    assert "pyobjc-framework-coreaudio" in text
    assert "pyobjc-framework-libdispatch" in text
    assert "pyobjc-framework-cocoa" in text
    assert "pyobjc-framework-appkit" not in text
    assert "pyaudiowpatch" not in text


def test_mac_requirements_pin_gigaam_compatible_stack():
    text = Path("requirements-mac.txt").read_text(encoding="utf-8").lower()
    assert "torch==2.8.0" in text
    assert "torchaudio==2.8.0" in text
    assert "transformers==4.57.1" in text
    assert "pyannote-audio>=4.0,<5" in text
    assert "torchcodec>=0.7" in text
    assert "socksio>=1.0.0" in text


def test_mac_launchers_require_the_verified_environment_marker():
    for path in ("install.sh", "start.sh", "update.sh"):
        text = Path(path).read_text(encoding="utf-8").lower()
        assert ".livetranslate-ready" in text
    assert "setup is incomplete" in Path("start.sh").read_text(encoding="utf-8").lower()


def test_mac_installers_pin_yasbd_and_resolve_dependencies_normally():
    for path in ("install.sh", "update.sh"):
        text = Path(path).read_text(encoding="utf-8").lower()
        assert 'yasbd-lib>=0.15,<1.0' in text
        assert "-r \"$root_dir/requirements-mac.txt\"" in text
        assert "--no-deps" not in text


def test_mac_requirements_contain_every_cross_platform_dependency():
    windows = _requirement_lines()
    mac = {
        line.split("#", 1)[0].strip().lower()
        for line in Path("requirements-mac.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    }
    common = {line for line in windows if not line.startswith("pyaudiowpatch")}

    # macOS may pin a stricter compatible version of a shared dependency.
    # Compare requirement names so `transformers==4.57.1` satisfies the
    # cross-platform baseline `transformers>=4.40.0`.
    def package_name(requirement: str) -> str:
        return requirement.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip()

    mac_names = {package_name(line) for line in mac}
    assert {package_name(line) for line in common} <= mac_names


def test_release_workflow_has_arm64_test_and_distinct_macos_artifact():
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8").lower()
    assert "test-macos-arm64" in workflow
    assert "runs-on: macos-14" in workflow
    assert "livetranslate-macos-arm64" in workflow
    assert "pytest -q" in workflow


def test_every_release_producing_job_gates_on_the_test_job():
    """A job that publishes an artifact must not run when the tests are red.

    The Windows `build` job used to omit `needs:` while the macOS package job
    declared it, so a tagged push published a Windows zip from a failing tree.
    """
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    jobs = re.split(r"^  (?=\S)", workflow.split("jobs:", 1)[1], flags=re.M)
    producing = {}
    for block in jobs:
        name = block.split(":", 1)[0].strip()
        if not name or name == "test-macos-arm64":
            continue
        if "upload-artifact" in block or "action-gh-release" in block:
            producing[name] = "needs: test-macos-arm64" in block

    assert producing, "no release-producing job found; update this test"
    ungated = sorted(name for name, gated in producing.items() if not gated)
    assert not ungated, f"release-producing jobs without `needs`: {ungated}"


def test_translation_layer_imports_without_the_network_stack():
    """translator must stay importable without openai/httpx so the offline
    test job can collect its pure-logic tests."""
    source = Path("translator.py").read_text(encoding="utf-8")
    header = source.split("def make_openai_client", 1)[0]

    assert "\nimport httpx" not in header
    assert "\nfrom openai import" not in header


def test_i18n_locales_define_the_same_keys():
    import yaml

    zh = yaml.safe_load(Path("i18n/zh.yaml").read_text(encoding="utf-8"))
    en = yaml.safe_load(Path("i18n/en.yaml").read_text(encoding="utf-8"))

    assert set(zh) == set(en), (
        f"only zh: {sorted(set(zh) - set(en))}; only en: {sorted(set(en) - set(zh))}"
    )
