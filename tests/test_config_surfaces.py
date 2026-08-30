"""Model-list index maintenance and remote-server config surface (C1 / C2)."""

import pytest


# --- C1: active_model must keep pointing at the same model ------------------

control_panel = pytest.importorskip(
    "control_panel", reason="control_panel needs PyQt6"
)
after_removal = control_panel.active_index_after_removal


def test_removing_a_model_before_the_active_one_shifts_the_index():
    """Everything after the deleted entry moves up by one; only clamping the
    tail (the old behaviour) silently switched the active model."""
    # models: [a, b, C, d], active = C at index 2, delete a
    assert after_removal(removed=0, active=2, remaining=3) == 1


def test_removing_a_model_after_the_active_one_leaves_it_alone():
    assert after_removal(removed=3, active=1, remaining=3) == 1


def test_removing_the_active_model_takes_the_neighbour():
    assert after_removal(removed=1, active=1, remaining=3) == 1


def test_removing_the_last_model_when_it_is_active_clamps():
    assert after_removal(removed=2, active=2, remaining=2) == 1


def test_the_result_is_always_a_valid_index():
    for removed in range(5):
        for active in range(5):
            for remaining in range(1, 5):
                index = after_removal(removed, active, remaining)
                assert 0 <= index < remaining


def test_an_empty_list_yields_zero():
    assert after_removal(removed=0, active=0, remaining=0) == 0


def test_the_active_entry_survives_arbitrary_deletions():
    """Property check against a real list: after deleting any index, the
    active slot still holds the same object."""
    for size in range(2, 6):
        for removed in range(size):
            for active in range(size):
                if removed == active:
                    continue  # the active model itself is gone; nothing to keep
                models = [{"name": f"m{i}"} for i in range(size)]
                expected = models[active]
                models.pop(removed)
                index = after_removal(removed, active, len(models))
                assert models[index] is expected


# --- C2: the ASR server configuration surface -------------------------------


def _server():
    """asr_server stays importable without the GPU serving stack, so the
    offline job verifies its configuration surface rather than skipping it."""
    import asr_server

    return asr_server


def test_the_default_bind_address_is_loopback(monkeypatch):
    """An unauthenticated GPU inference endpoint must not default to 0.0.0.0."""
    server = _server()
    monkeypatch.delenv("LIVETRANSLATE_ASR_HOST", raising=False)
    assert server.default_config().host == "127.0.0.1"


def test_both_entry_points_read_the_same_configuration():
    server = _server()
    defaults = server.default_config()
    parsed = server.parse_args([])
    for field in ("host", "port", "model", "device", "compute_type"):
        assert getattr(parsed, field) == getattr(defaults, field)


def test_flags_override_the_defaults():
    server = _server()
    args = server.parse_args(["--host", "0.0.0.0", "--port", "9000"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_environment_variables_configure_the_asgi_entry_point(monkeypatch):
    server = _server()
    monkeypatch.setenv("LIVETRANSLATE_ASR_MODEL", "large-v3")
    monkeypatch.setenv("LIVETRANSLATE_ASR_PORT", "9999")
    config = server.default_config()
    assert config.model == "large-v3"
    assert config.port == 9999


def test_the_module_level_app_carries_its_configuration():
    """`uvicorn asr_server:app` used to start with no app.state.args at all and
    die in the startup hook."""
    server = _server()
    pytest.importorskip("fastapi")
    pytest.importorskip("faster_whisper")
    assert server.app is not None
    assert server.app.state.args is not None
    assert server.app.state.args.model


def test_create_app_says_what_is_missing_when_the_stack_is_absent():
    server = _server()
    if server.FastAPI is not None:
        pytest.skip("the serving stack is installed here")
    with pytest.raises(RuntimeError, match="fastapi"):
        server.create_app()


def test_the_body_limit_matches_the_documented_audio_length():
    server = _server()
    assert server.MAX_AUDIO_SECONDS == 300
    assert server.MAX_BODY_BYTES >= 300 * 16000 * 4
    assert server.MAX_BODY_BYTES < 300 * 16000 * 4 + 10_000


def test_the_wire_format_parser_still_rejects_malformed_bodies():
    import struct

    server = _server()
    with pytest.raises(ValueError):
        server._parse_request(b"ab")
    with pytest.raises(ValueError):
        server._parse_request(struct.pack("<I", 99) + b"en")
    with pytest.raises(ValueError):
        server._parse_request(struct.pack("<I", 0) + b"xyz")  # not a multiple of 4


def test_a_valid_body_round_trips():
    import struct

    import numpy as np

    server = _server()
    audio = np.array([0.1, -0.2], dtype=np.float32)
    body = struct.pack("<I", 2) + b"ru" + audio.tobytes()
    language, decoded = server._parse_request(body)
    assert language == "ru"
    assert np.allclose(decoded, audio)
