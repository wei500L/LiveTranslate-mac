"""In-process Soniox streaming engine (the ASRClient-compatible shim).

Mirrors asr_remote.RemoteASREngine: cloud engines run in-process — no worker
subprocess, no GPU model — and present the ASRClient surface so
_switch_asr_engine / _mem_snapshot / _maybe_recycle_asr_worker treat them
uniformly (the recycle logic skips `client.pid is None`).

Unlike every other engine this one is streaming-only: audio flows through
SonioxServiceManager.feed() from the capture loop (bypassing local VAD), and
results arrive as sink callbacks — never through transcribe(). _run_asr is
bypassed entirely in soniox mode.
"""

from __future__ import annotations

import logging

from soniox_client import (
    SonioxMissingKeyError,
    SonioxRuntimeConfig,
    SonioxServiceManager,
    SonioxSink,
    mask_key,
)

log = logging.getLogger("LiveTranslate.ASR.Soniox")


class SonioxASREngine:
    """Soniox cloud realtime STT (ru -> zh), streaming-only."""

    def __init__(
        self,
        *,
        api_key: str | None,
        context_text: str = "",
        segmentation: str = "accuracy",
        sink: SonioxSink,
    ) -> None:
        if not api_key:
            raise SonioxMissingKeyError(
                "Soniox API key is missing. Set the SONIOX_API_KEY environment "
                "variable or fill in the API key field in Settings → VAD/ASR."
            )
        # Probe the dependency eagerly (not just on first connect): a missing
        # SDK must fail the engine load with an actionable message, not leave
        # the manager retrying in the background.
        from soniox_client import _import_soniox  # noqa: F401 — probe only

        _import_soniox()

        self._manager = SonioxServiceManager(
            SonioxRuntimeConfig(
                api_key=api_key,
                context_text=context_text,
                segmentation=segmentation,
            ),
            sink,
        )
        self._stopped = False
        log.info(
            "Soniox engine created (key %s, segmentation=%s)",
            mask_key(api_key), segmentation,
        )

    # --- ASRClient-compatible shim -----------------------------------------

    @property
    def status(self) -> str:
        # Never report a torn-down engine as usable: _switch_asr_engine
        # reuses the active backend on `status == "ready"`.
        return "stopped" if self._stopped else "ready"

    @property
    def pid(self):
        # No worker process; a None pid makes the memory monitor and
        # worker-recycle logic skip this engine.
        return None

    @property
    def manager(self) -> SonioxServiceManager:
        return self._manager

    def set_language(self, language):
        # Fixed ru -> zh; language is cloud-side config, not runtime state.
        pass

    def set_input_padding(self, pad_seconds):
        # Local-side padding; audio streams unmodified in cloud mode.
        pass

    def to_device(self, device):
        pass

    def transcribe(self, audio, **kwargs):
        raise RuntimeError(
            "The Soniox engine is streaming-only; audio is uploaded "
            "continuously via SonioxServiceManager.feed(), not transcribe()"
        )

    def shutdown(self, timeout: float = 5.0):
        self.unload(timeout)

    def terminate(self, timeout: float = 5.0):
        self.unload(timeout)

    def unload(self, timeout: float = 5.0):
        if self._stopped:
            return
        self._stopped = True
        self._manager.shutdown(timeout=timeout)
