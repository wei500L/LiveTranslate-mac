"""Contract test against the installed soniox SDK (offline, no network).

Runs only where the SDK is installed (project venv, CI). Verifies the field
names and session surface soniox_client.py is written against, so an SDK
release that renames anything fails here instead of at runtime in the field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

soniox_types = pytest.importorskip("soniox.types")


def test_realtime_stt_config_accepts_our_field_names():
    from soniox.types import RealtimeSTTConfig, TranslationConfig

    cfg = RealtimeSTTConfig(
        model="stt-rt-v5",
        audio_format="pcm_s16le",
        sample_rate=16000,
        num_channels=1,
        language_hints=["ru"],
        language_hints_strict=True,
        enable_endpoint_detection=True,
        endpoint_latency_adjustment_level=2,
        endpoint_sensitivity=0.3,
        max_endpoint_delay_ms=1500,
        translation=TranslationConfig(type="one_way", target_language="zh"),
    )
    assert cfg.model == "stt-rt-v5"
    assert cfg.language_hints == ["ru"]
    assert cfg.translation.target_language == "zh"


def test_structured_context_accepts_typed_items():
    from soniox.types import (
        StructuredContext,
        StructuredContextGeneralItem,
        StructuredContextTranslationTerm,
    )

    ctx = StructuredContext(
        general=[StructuredContextGeneralItem(key="topic", value="физика")],
        text="физика",
        terms=["термодинамика"],
        translation_terms=[
            StructuredContextTranslationTerm(source="энтропия", target="熵")
        ],
    )
    assert ctx.translation_terms[0].source == "энтропия"


def test_session_surface_has_the_seven_methods():
    from soniox.realtime.stt import RealtimeSTTSession

    for name in (
        "send_byte_chunk",
        "receive_events",
        "finish",
        "finalize",
        "pause",
        "resume",
        "close",
    ):
        assert callable(getattr(RealtimeSTTSession, name, None)), name


def test_token_carries_translation_status():
    from soniox.types.common import Token

    token = Token(text="Привет", is_final=True, translation_status="original")
    assert token.translation_status == "original"


def test_end_token_constant_spelling():
    from soniox.types.realtime import TOKEN_TEXT_END

    assert TOKEN_TEXT_END == "<end>"
