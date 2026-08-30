"""Benchmark and runtime must issue the same request (B2).

The benchmark is the tool users reach for when a model behaves oddly, so a
crash in it (issue #38's content=None, a usage-only stream frame) costs exactly
the diagnosis it exists to provide.
"""

import pytest

import benchmark
import translator as translator_module
from mlx_service import hy_mt_model_config
from translator import Translator


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(
        translator_module, "make_openai_client", lambda *a, **k: object()
    )


MODEL = {
    "name": "test",
    "api_base": "http://127.0.0.1:1234/v1",
    "api_key": "k",
    "model": "m",
    "no_system_role": True,
    "thinking_style": "qwen",
    "overrides": {"temperature": 0.15, "max_tokens": 77, "top_p": 0.4},
    "extra_body": {"top_k": 11},
    "system_prompt": "Translate {source_lang} to {target_lang}.",
}


def test_benchmark_request_matches_the_runtime_request():
    bench = benchmark.build_bench_translator(MODEL, "fallback prompt", "zh", 5)
    runtime = Translator(
        api_base=MODEL["api_base"],
        api_key=MODEL["api_key"],
        model=MODEL["model"],
        target_language="zh",
        max_tokens=MODEL["overrides"]["max_tokens"],
        temperature=MODEL["overrides"]["temperature"],
        no_system_role=MODEL["no_system_role"],
        thinking_style=MODEL["thinking_style"],
        overrides=MODEL["overrides"],
        extra_body=MODEL["extra_body"],
        system_prompt=MODEL["system_prompt"],
    )

    bench_kwargs = bench._build_request_kwargs(
        bench._build_system_prompt("en"), "hello", stream=True
    )
    runtime_kwargs = runtime._build_request_kwargs(
        runtime._build_system_prompt("en"), "hello", stream=True
    )
    assert bench_kwargs == runtime_kwargs


def test_benchmark_carries_every_per_model_flag():
    bench = benchmark.build_bench_translator(MODEL, "fallback", "ja", 5)
    kwargs = bench._build_request_kwargs(
        bench._build_system_prompt("en"), "hello", stream=True
    )
    assert kwargs["messages"][0]["role"] == "user"  # no_system_role
    assert kwargs["temperature"] == 0.15
    assert kwargs["max_tokens"] == 77
    assert kwargs["top_p"] == 0.4
    assert kwargs["extra_body"]["top_k"] == 11
    assert kwargs["extra_body"]["enable_thinking"] is False  # qwen style


def test_a_model_without_its_own_prompt_uses_the_benchmark_prompt():
    model = dict(MODEL)
    model.pop("system_prompt")
    bench = benchmark.build_bench_translator(model, "shared prompt", "zh", 5)
    assert "shared prompt" in bench._build_system_prompt("en")


def test_hy_mt_benchmarks_with_its_managed_profile():
    config = hy_mt_model_config()
    bench = benchmark.build_bench_translator(config, "unused", "zh", 5)
    kwargs = bench._build_request_kwargs(
        bench._build_system_prompt("ru"), "Привет", stream=True
    )
    assert kwargs["temperature"] == 0.0   # greedy; see the preset's comment
    assert kwargs["max_tokens"] == 128
    assert kwargs["extra_body"]["repetition_penalty"] == 1.05
    assert kwargs["messages"][0]["role"] == "user"


def test_benchmark_translator_keeps_no_history():
    """A benchmark run must not write into the context the pipeline uses."""
    bench = benchmark.build_bench_translator(MODEL, "p", "zh", 5)
    assert bench._context_turns == 0
    bench._append_history("src", "dst")
    assert bench._history == []


def test_only_parameter_rejections_trigger_the_non_streaming_fallback():
    """A connection error retried here doubled the wait for every sentence."""
    errors = benchmark.stream_option_errors()
    assert TypeError in errors
    openai = pytest.importorskip("openai")

    assert openai.BadRequestError in errors
    assert openai.APIConnectionError not in errors
    assert openai.APITimeoutError not in errors


# --- One construction site, so the two cannot drift -------------------------


def test_a_config_without_thinking_style_resolves_the_same_on_both_paths():
    """This is the parameter the benchmark exists to diagnose (issue #38), and
    it silently diverged: Translator's own no_think default was False while
    every caller passed True, so the benchmark resolved "off" (send nothing)
    where the runtime resolved "qwen"."""
    from translator import translator_from_model_config

    model = {
        "name": "cloud",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "k",
        "model": "Qwen/Qwen3-32B",
        # deliberately no thinking_style and no no_think, like a real config
    }
    runtime = translator_from_model_config(model, target_language="zh")
    bench = benchmark.build_bench_translator(model, "prompt", "zh", 5)
    assert bench._thinking_style == runtime._thinking_style == "qwen"


def test_the_documented_auto_default_is_what_you_actually_get():
    """CLAUDE.md documents thinking_style as defaulting to "auto"."""
    from translator import Translator

    tr = Translator(
        api_base="https://api.siliconflow.cn/v1", api_key="k",
        model="Qwen/Qwen3-32B", target_language="zh",
    )
    assert tr._thinking_style == "qwen"  # what "auto" resolves to here


def test_a_legacy_no_think_false_still_means_off():
    from translator import Translator

    tr = Translator(
        api_base="https://api.siliconflow.cn/v1", api_key="k",
        model="Qwen/Qwen3-32B", target_language="zh", no_think=False,
    )
    assert tr._thinking_style == "off"


def test_the_factory_carries_every_per_model_flag():
    from translator import translator_from_model_config

    model = {
        "api_base": "http://127.0.0.1:1234/v1", "api_key": "k", "model": "m",
        "streaming": False, "no_system_role": True, "json_response": True,
        "thinking_style": "vllm", "overrides": {"max_tokens": 64, "top_p": 0.5},
        "extra_body": {"custom": 1},
    }
    tr = translator_from_model_config(model, target_language="ja", timeout=42)
    assert tr._streaming is False
    assert tr._no_system_role is True
    assert tr._json_response is True
    assert tr._thinking_style == "vllm"
    assert tr._max_tokens == 64          # from overrides, not the fallback
    assert tr._extra_body == {"custom": 1}
    assert tr._timeout == 42
    assert tr._target_language == "ja"


def test_environment_overrides_apply_through_the_factory(monkeypatch):
    """LIVETRANSLATE_* used to work only on the app's construction path."""
    from translator import translator_from_model_config

    monkeypatch.setenv("LIVETRANSLATE_MODEL", "from-env")
    tr = translator_from_model_config(
        {"api_base": "http://x/v1", "api_key": "k", "model": "from-config"},
        target_language="zh",
    )
    assert tr._model == "from-env"
