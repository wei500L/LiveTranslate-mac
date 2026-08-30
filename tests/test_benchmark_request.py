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
    assert kwargs["temperature"] == 0.7
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
