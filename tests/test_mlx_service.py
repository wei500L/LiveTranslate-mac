from mlx_service import (
    HY_MT_MODEL_ID,
    MLX_BASE_URL,
    MLXServiceManager,
    ensure_hy_mt_model,
    hy_mt_model_config,
    is_hy_mt_model,
)
import translator as translator_module
from translator import Translator


def test_hy_mt_model_config_is_openai_compatible_and_local():
    model = hy_mt_model_config()

    assert model["model"] == HY_MT_MODEL_ID
    assert model["api_base"] == MLX_BASE_URL
    assert model["no_system_role"] is True
    assert model["thinking_style"] == "off"
    assert model["overrides"] == {
        "temperature": 0.7,
        "top_p": 0.6,
        "max_tokens": 128,
    }
    assert model["extra_body"] == {"top_k": 20, "repetition_penalty": 1.05}
    assert is_hy_mt_model(model)


def test_ensure_hy_mt_model_preserves_existing_models():
    settings = {
        "models": [{"name": "cloud", "model": "cloud-model"}],
        "active_model": 0,
    }

    changed = ensure_hy_mt_model(settings)

    assert changed is True
    assert [m["name"] for m in settings["models"]] == [
        "cloud",
        "HY-MT1.5-7B (MLX 4-bit)",
    ]
    assert settings["active_model"] == 0


def test_ensure_hy_mt_model_is_idempotent():
    settings = {"models": []}

    assert ensure_hy_mt_model(settings) is True
    assert ensure_hy_mt_model(settings) is False
    assert len(settings["models"]) == 1


def test_service_readiness_requires_model_files(tmp_path):
    manager = MLXServiceManager(tmp_path)
    assert manager.is_model_ready() is False

    model_dir = tmp_path / "models" / "hy-mt1.5-7b-mlx-4bit"
    model_dir.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors",
    ):
        (model_dir / name).write_text("{}", encoding="utf-8")

    assert manager.is_model_ready() is True


def test_hy_mt_request_uses_user_role_and_official_sampling(monkeypatch):
    monkeypatch.setattr(
        translator_module,
        "make_openai_client",
        lambda *args, **kwargs: object(),
    )
    config = hy_mt_model_config()
    translator = Translator(
        api_base=config["api_base"],
        api_key=config["api_key"],
        model=config["model"],
        target_language="zh",
        no_system_role=config["no_system_role"],
        thinking_style=config["thinking_style"],
        overrides=config["overrides"],
        extra_body=config["extra_body"],
        system_prompt=config["system_prompt"],
    )

    request = translator._build_request_kwargs(
        translator._build_system_prompt("en"), "It's on the house.", stream=True
    )

    assert request["messages"][0]["role"] == "user"
    prompt = request["messages"][0]["content"]
    assert prompt.endswith("\nIt's on the house.")
    assert "俄语课堂的实时翻译助手" in prompt
    assert "课程术语" in prompt
    assert "近期课堂上下文" in prompt
    assert request["temperature"] == 0.7
    assert request["top_p"] == 0.6
    assert request["max_tokens"] == 128
    assert request["extra_body"] == {
        "top_k": 20,
        "repetition_penalty": 1.05,
    }
    assert request["stream"] is True


def test_pid_liveness_uses_signal_zero(tmp_path, monkeypatch):
    manager = MLXServiceManager(tmp_path)
    calls = []
    monkeypatch.setattr("mlx_service.os.kill", lambda pid, sig: calls.append((pid, sig)))

    assert manager._pid_is_alive(1234) is True
    assert calls == [(1234, 0)]


def test_model_preparation_is_in_app_and_cleans_temporary_source(tmp_path, monkeypatch):
    manager = MLXServiceManager(tmp_path)
    source = tmp_path / "models" / ".hy-mt1.5-7b-bf16.tmp"
    source.mkdir(parents=True)
    source.joinpath("leftover").write_text("temporary", encoding="utf-8")
    monkeypatch.setattr(manager, "is_supported_platform", lambda: True)
    monkeypatch.setattr(manager, "is_environment_ready", lambda: True)
    monkeypatch.setattr(manager, "is_model_ready", lambda: True)
    manager.prepare_model()
    assert not source.exists()


# --- A8: lifecycle, cancellation and platform safety -----------------------


def test_version_probe_is_cached_against_the_venv_mtime(tmp_path, monkeypatch):
    """It spawns an interpreter, and _update_mlx_controls used to call it on
    every model-list selection change."""
    manager = MLXServiceManager(tmp_path)
    python = tmp_path / ".mlx-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(manager, "_probe_versions", lambda: calls.append(1) or True)

    assert manager._versions_are_compatible() is True
    assert manager._versions_are_compatible() is True
    assert len(calls) == 1

    # Touching the venv invalidates the cache.
    import os

    os.utime(manager.env_dir, (1, 1))
    assert manager._versions_are_compatible() is True
    assert len(calls) == 2


def test_prepare_model_refuses_to_replace_a_directory_the_server_holds(
    tmp_path, monkeypatch
):
    manager = MLXServiceManager(tmp_path)
    monkeypatch.setattr(manager, "is_supported_platform", lambda: True)
    monkeypatch.setattr(manager, "is_running", lambda: True)
    stopped = []
    monkeypatch.setattr(manager, "stop", lambda: stopped.append(1))

    import pytest

    from mlx_service import MLXServiceError

    with pytest.raises(MLXServiceError):
        manager.prepare_model()
    assert stopped == [1]  # it tries a stop before refusing


def test_prepare_model_stops_a_running_service_then_proceeds(tmp_path, monkeypatch):
    manager = MLXServiceManager(tmp_path)
    running = [True]
    monkeypatch.setattr(manager, "is_supported_platform", lambda: True)
    monkeypatch.setattr(manager, "is_running", lambda: running[0])
    monkeypatch.setattr(manager, "stop", lambda: running.__setitem__(0, False))
    monkeypatch.setattr(manager, "is_environment_ready", lambda: True)
    monkeypatch.setattr(manager, "is_model_ready", lambda: True)

    manager.prepare_model()  # must not raise
    assert running[0] is False


def test_run_logged_cancels_a_silent_child(tmp_path):
    """The cancel check used to live inside `for line in stdout`, so a child
    that printed nothing was uncancellable."""
    import threading
    import time

    import pytest

    from mlx_service import MLXServiceError

    manager = MLXServiceManager(tmp_path)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()

    started = time.monotonic()
    with pytest.raises(MLXServiceError):
        manager._run_logged(
            [sys_executable(), "-c", "import time; time.sleep(30)"],
            cancel_event=cancel,
        )
    assert time.monotonic() - started < 15


def sys_executable():
    import sys

    return sys.executable


def test_stop_survives_a_platform_without_killpg(tmp_path, monkeypatch):
    """stop() hangs off aboutToQuit on every platform; os.killpg is POSIX-only
    and an AttributeError there escapes into Qt's shutdown."""
    manager = MLXServiceManager(tmp_path)
    manager.log_dir.mkdir(parents=True, exist_ok=True)
    manager.pid_file.write_text("4321", encoding="ascii")
    monkeypatch.setattr(manager, "_pid_is_owned", lambda pid: True)
    alive = [True]
    monkeypatch.setattr(manager, "_pid_is_alive", lambda pid: alive[0])
    monkeypatch.delattr("mlx_service.os.killpg", raising=False)

    def fake_kill(pid, sig):
        alive[0] = False

    monkeypatch.setattr("mlx_service.os.kill", fake_kill)

    manager.stop()  # must not raise AttributeError
    assert not manager.pid_file.exists()


def test_progress_text_is_localized_through_the_injected_translator(tmp_path):
    manager = MLXServiceManager(tmp_path)
    manager.translate = lambda key: {"mlx_downloading": "Fetching {repo}"}.get(key, key)
    assert manager._text("mlx_downloading", repo="acme/model") == "Fetching acme/model"
    # Unknown keys fall back to the key itself rather than raising.
    assert manager._text("no_such_key") == "no_such_key"
