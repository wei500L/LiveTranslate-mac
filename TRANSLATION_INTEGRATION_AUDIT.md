# Translation Integration Audit

This is a static integration review; it does not claim that an external model
or server is reachable at review time.

## Result

The translation path is wired end to end:

`audio_capture` -> VAD -> ASR worker -> `Translator` -> OpenAI-compatible
`chat.completions` -> subtitle overlay.

The model adapter is provider-agnostic. `translator.py` owns the shared
OpenAI client, streaming/non-streaming request assembly, context history,
structured output, proxy handling, and provider-specific thinking controls.
The control panel persists model entries and `main.py` rebuilds the translator
when the active entry changes.

## External tool boundaries

| Tool | Protocol | Static status |
| --- | --- | --- |
| OpenAI-compatible translation model | HTTP JSON, `/v1/chat/completions` | Connected through `Translator` |
| Remote Whisper ASR | HTTP, `/health` and `/transcribe` | Compatible client/server pair |
| HY-MT MLX service | Local OpenAI-compatible HTTP | Managed start/probe/stop path |
| ModelScope/Hugging Face | Python download APIs | Centralized in `model_manager.py` |

## Configuration changes

Endpoint defaults and normalization now live in `connection_config.py`.
Existing settings remain compatible. For a one-line setup, the translation
endpoint and key can be supplied with `LIVETRANSLATE_API_BASE` and
`LIVETRANSLATE_API_KEY`; `LIVETRANSLATE_MODEL` is also supported.

The repository config no longer contains a live-looking API key. A real key is
provided by the first-run dialog or the environment.

The model editor now includes a local HY-MT preset. Selecting it fills the
OpenAI-compatible URL, local key, model identifier, managed-service metadata,
sampling defaults, and classroom prompt automatically; existing custom API
entries are unchanged.

## Remaining runtime checks

Static review cannot verify model quality, API authentication, provider-specific
request acceptance, audio permissions, or GPU/model availability. Those require
an actual configured endpoint and a live capture session.
