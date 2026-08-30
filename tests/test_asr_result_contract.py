"""The shared ASR result contract (CALL_CHAIN_FIX_TODO 2.1 / 2.2).

Both the local worker path and RemoteASREngine must agree on what a result
looks like, and a transport failure must never be indistinguishable from
silence. These imports are skipped rather than failed when the heavy runtime
deps are absent, so the offline CI job still collects this file.
"""

import numpy as np
import pytest


def _remote():
    pytest.importorskip("httpx")
    import asr_remote

    return asr_remote


def _engine(response):
    """RemoteASREngine with its HTTP client replaced by a canned response."""
    mod = _remote()
    engine = mod.RemoteASREngine.__new__(mod.RemoteASREngine)
    engine._client = type("C", (), {"post": lambda self, *a, **k: response})()
    engine._url = "http://example.invalid/transcribe"
    engine._closed = False
    engine.language = None
    return engine, mod


def _response(status=200, payload=None, raises=None):
    class R:
        status_code = status

        def json(self):
            if raises is not None:
                raise raises
            return payload

    return R()


AUDIO = np.zeros(16, dtype=np.float32)


def test_valid_response_returns_the_contract_shape():
    engine, _ = _engine(_response(payload={"text": "hello", "language": "en"}))
    assert engine.transcribe(AUDIO) == {
        "text": "hello",
        "language": "en",
        "language_name": "en",
    }


def test_empty_text_in_a_well_formed_response_is_silence_not_an_error():
    engine, _ = _engine(_response(payload={"text": "", "language": "en"}))
    assert engine.transcribe(AUDIO) is None


@pytest.mark.parametrize(
    "response",
    [
        _response(status=500, payload={}),
        _response(status=200, raises=ValueError("not json")),
        _response(status=200, payload=["not", "a", "dict"]),
        _response(status=200, payload={"text": 42, "language": "en"}),
        _response(status=200, payload={"text": "hi", "language": 7}),
    ],
    ids=["http-500", "bad-json", "not-a-dict", "text-not-str", "language-not-str"],
)
def test_transport_and_protocol_failures_raise_instead_of_returning_none(response):
    engine, mod = _engine(response)
    with pytest.raises(mod.RemoteASRError):
        engine.transcribe(AUDIO)


def test_connection_failure_raises_rather_than_looking_like_silence():
    mod = _remote()
    engine = mod.RemoteASREngine.__new__(mod.RemoteASREngine)

    def boom(*a, **k):
        raise OSError("connection refused")

    engine._client = type("C", (), {"post": lambda self, *a, **k: boom()})()
    engine._url = "http://example.invalid/transcribe"
    engine._closed = False
    engine.language = None
    with pytest.raises(mod.RemoteASRError):
        engine.transcribe(AUDIO)


# ── Local worker path: same contract, enforced by validate_asr_result ──


def _validate():
    pytest.importorskip("torch")
    pytest.importorskip("PyQt6")
    import main

    return main


def test_validator_accepts_a_well_formed_result():
    main = _validate()
    assert main.validate_asr_result(
        {"text": "  hi  ", "language": "ja"}, "segment"
    ) == ("hi", "ja")


def test_validator_treats_none_and_blank_text_as_silence():
    main = _validate()
    assert main.validate_asr_result(None, "segment") is None
    assert main.validate_asr_result({"text": "   ", "language": "ja"}, "segment") is None


@pytest.mark.parametrize(
    "result",
    [
        "not a dict",
        {"language": "ja"},
        {"text": None, "language": "ja"},
        {"text": "hi"},
        {"text": "hi", "language": ""},
        {"text": "hi", "language": None},
    ],
)
def test_validator_rejects_contract_violations(result):
    main = _validate()
    with pytest.raises(main.ASRProtocolError):
        main.validate_asr_result(result, "segment")
