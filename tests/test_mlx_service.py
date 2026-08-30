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
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 128,
    }
    assert model["extra_body"] == {"repetition_penalty": 1.05}
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
    # One directive line, no rule list and no context block: this model
    # continues those instead of following them (see the preset's comment).
    assert "Output only the translation." in prompt
    assert "Chinese" in prompt
    assert "\n-" not in prompt
    assert request["temperature"] == 0.0
    assert request["top_p"] == 1.0
    assert request["max_tokens"] == 128
    assert request["extra_body"] == {"repetition_penalty": 1.05}
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


# --- The preset prompt this model can actually follow -----------------------


def test_the_managed_preset_avoids_context_and_long_instructions():
    """HY-MT is translation-specialized, not instruction-following: measured on
    the 4-bit build, a multi-line instruction block failed 3/3 (regurgitated the
    prompt, burned the whole 128-token budget) and adding a context block failed
    0/3. A one-line directive with no context succeeded 4/4 at ~0.3s."""
    config = hy_mt_model_config()

    prompt = config["system_prompt"]
    assert prompt.count("\n\n") <= 1 and "\n-" not in prompt, (
        "a multi-line rule list makes this model continue the prompt"
    )
    assert "{context}" not in prompt
    # Context is carried as real conversation turns, never pasted into the
    # prompt: the text-block form is what makes this model continue the block.
    assert config["context_turns"] == 2
    # The placeholders the prompt builder substitutes must still be there.
    assert "{source_lang}" in prompt and "{target_lang}" in prompt


def test_an_unedited_superseded_prompt_is_migrated():
    from mlx_service import _is_superseded_hy_mt_prompt

    old = (
        "你是俄语课堂的实时翻译助手。请将课堂中的{source_lang}内容翻译成{target_lang}。\n"
        "规则：\n- 只输出一条译文。\n近期课堂上下文：\n{context}"
    )
    settings = {"models": [{**hy_mt_model_config(), "system_prompt": old}]}
    assert _is_superseded_hy_mt_prompt(old)
    assert ensure_hy_mt_model(settings) is True
    migrated = settings["models"][0]
    assert migrated["system_prompt"] == hy_mt_model_config()["system_prompt"]


def test_every_prompt_this_project_shipped_is_migrated():
    """Two generations have been superseded; a config on either must converge
    on the current preset rather than being stranded."""
    from mlx_service import _SUPERSEDED_HY_MT_PROMPTS, _is_superseded_hy_mt_prompt

    assert len(_SUPERSEDED_HY_MT_PROMPTS) >= 2
    samples = [
        "你是俄语课堂的实时翻译助手。\n近期课堂上下文：\n{context}",
        "把大学课堂上的{source_lang}内容翻译成{target_lang}，只输出译文。",
    ]
    for sample in samples:
        assert _is_superseded_hy_mt_prompt(sample), sample[:20]
        settings = {"models": [{**hy_mt_model_config(), "system_prompt": sample}]}
        ensure_hy_mt_model(settings)
        assert settings["models"][0]["system_prompt"] == (
            hy_mt_model_config()["system_prompt"]
        )
    # The current preset must not match itself, or it would migrate every load.
    assert not _is_superseded_hy_mt_prompt(hy_mt_model_config()["system_prompt"])


def test_a_user_edited_prompt_is_left_alone():
    """Only the prompt we shipped is replaced; a customized one is theirs."""
    from mlx_service import _is_superseded_hy_mt_prompt

    custom = "Translate {source_lang} to {target_lang}. Keep it terse."
    assert not _is_superseded_hy_mt_prompt(custom)
    settings = {"models": [{**hy_mt_model_config(), "system_prompt": custom}]}
    ensure_hy_mt_model(settings)
    assert settings["models"][0]["system_prompt"] == custom


def test_the_prompt_builder_produces_a_single_directive_line(monkeypatch):
    monkeypatch.setattr(
        translator_module, "make_openai_client", lambda *a, **k: object()
    )
    config = hy_mt_model_config()
    translator = Translator(
        api_base=config["api_base"], api_key=config["api_key"],
        model=config["model"], target_language="zh",
        no_system_role=config["no_system_role"],
        thinking_style=config["thinking_style"],
        system_prompt=config["system_prompt"],
    )
    request = translator._build_request_kwargs(
        translator._build_system_prompt("ru"), "Как вас зовут?", stream=False
    )
    content = request["messages"][0]["content"]
    assert request["messages"][0]["role"] == "user"
    assert content.endswith("Как вас зовут?")
    assert "Russian" in content and "Chinese" in content
    # Everything before the source text is one instruction line.
    instruction = content[: content.index("Как вас зовут?")].strip()
    assert "\n" not in instruction


# --- Multi-turn context for providers that reject a system role -------------


def _translator(monkeypatch, **kwargs):
    monkeypatch.setattr(
        translator_module, "make_openai_client", lambda *a, **k: object()
    )
    defaults = dict(
        api_base="http://127.0.0.1:8080/v1", api_key="local", model="m",
        target_language="zh", system_prompt="INSTRUCTION for {target_lang}.",
    )
    defaults.update(kwargs)
    return Translator(**defaults)


def test_no_system_role_now_carries_conversation_context(monkeypatch):
    """context_turns used to be silently ignored for these providers: the
    branch built a single merged user message and never looked at history."""
    tr = _translator(monkeypatch, no_system_role=True)
    tr.set_context_turns(2)
    tr._history = [("src1", "dst1"), ("src2", "dst2")]

    msgs = tr._build_messages(tr._build_system_prompt("ru"), "src3")
    assert [m["role"] for m in msgs] == [
        "user", "assistant", "user", "assistant", "user"
    ]
    # The instruction rides on the first message only.
    assert msgs[0]["content"].startswith("INSTRUCTION")
    assert msgs[0]["content"].endswith("src1")
    assert msgs[1]["content"] == "dst1"
    assert msgs[2]["content"] == "src2"
    assert msgs[3]["content"] == "dst2"
    assert msgs[4]["content"] == "src3"
    assert "INSTRUCTION" not in "".join(m["content"] for m in msgs[1:])


def test_no_system_role_without_history_is_unchanged(monkeypatch):
    tr = _translator(monkeypatch, no_system_role=True)
    tr.set_context_turns(2)
    msgs = tr._build_messages(tr._build_system_prompt("ru"), "only")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].endswith("only")


def test_context_turns_zero_sends_no_history(monkeypatch):
    tr = _translator(monkeypatch, no_system_role=True)
    tr.set_context_turns(0)
    tr._history = [("src1", "dst1")]
    assert len(tr._build_messages(tr._build_system_prompt("ru"), "x")) == 1


def test_only_the_configured_number_of_turns_is_sent(monkeypatch):
    tr = _translator(monkeypatch, no_system_role=True)
    tr.set_context_turns(2)
    tr._history = [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]
    msgs = tr._build_messages(tr._build_system_prompt("ru"), "e")
    assert len(msgs) == 5  # 2 pairs + the new source
    assert msgs[0]["content"].endswith("c")
    assert msgs[2]["content"] == "d"


def test_a_context_placeholder_prompt_does_not_also_get_turns(monkeypatch):
    """That template embeds the history itself; sending both duplicates it."""
    tr = _translator(
        monkeypatch, no_system_role=True,
        system_prompt="Instruction. Recent:\n{context}",
    )
    tr.set_context_turns(2)
    tr._history = [("src1", "dst1")]
    msgs = tr._build_messages(tr._build_system_prompt("ru"), "src2")
    assert len(msgs) == 1


def test_the_system_role_path_is_unaffected(monkeypatch):
    tr = _translator(monkeypatch, no_system_role=False)
    tr.set_context_turns(1)
    tr._history = [("src1", "dst1")]
    msgs = tr._build_messages(tr._build_system_prompt("ru"), "src2")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[0]["content"].startswith("INSTRUCTION")


def test_the_managed_preset_asks_for_greedy_decoding():
    """Translation wants the single best rendering, not a sample: measured
    4/4 identical outputs versus 2/4 at temperature 0.7, same quality, lower
    latency. A subtitle should not change between identical utterances."""
    config = hy_mt_model_config()
    assert config["overrides"]["temperature"] == 0.0
    assert config["context_turns"] == 2
    # The runaway guard stays; top_k is meaningless under greedy decoding.
    assert config["extra_body"] == {"repetition_penalty": 1.05}


def test_stale_operational_params_are_migrated_away():
    settings = {"models": [{
        **hy_mt_model_config(),
        "context_turns": 0,
        "overrides": {"temperature": 0.7, "top_p": 0.6, "max_tokens": 128},
        "extra_body": {"top_k": 20, "repetition_penalty": 1.05},
    }]}
    assert ensure_hy_mt_model(settings) is True
    model = settings["models"][0]
    assert model["overrides"]["temperature"] == 0.0
    assert model["context_turns"] == 2
    assert "top_k" not in model["extra_body"]
