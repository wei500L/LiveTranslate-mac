from connection_config import (
    DEFAULT_TRANSLATION_API_BASE,
    normalize_api_base,
    normalize_remote_asr_url,
)


def test_openai_base_defaults_to_v1_and_deduplicates_slashes():
    assert normalize_api_base("127.0.0.1:1234") == DEFAULT_TRANSLATION_API_BASE
    assert normalize_api_base("https://example.test/v1/") == "https://example.test/v1"


def test_remote_asr_url_is_an_origin():
    assert normalize_remote_asr_url("127.0.0.1:8765/") == "http://127.0.0.1:8765"
