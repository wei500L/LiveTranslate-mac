"""GigaAM Russian speech recognizer.

GigaAM is a transformers/torch model and therefore can use MPS.  Imports and
model construction are lazy so selecting another engine does not require the
optional model stack to be present.
"""

from __future__ import annotations

import gc
import logging

import numpy as np

from torch_backend import normalize_device

log = logging.getLogger("LiveTranslate.GigaAM")

MODEL_ID = "salute-ai/GigaAM-v3-RNNT"


class GigaAMEngine:
    """Russian-only GigaAM v2/v3 pipeline with MPS -> CPU fallback."""

    def __init__(self, device="mps", hub="hf", model_id: str | None = None):
        import torch
        from transformers import pipeline

        from model_manager import get_local_model_path

        requested = normalize_device(device)
        if requested == "mps":
            try:
                if not torch.backends.mps.is_available():
                    requested = "cpu"
            except (AttributeError, RuntimeError):
                requested = "cpu"
        if requested not in ("cpu", "mps") and not requested.startswith("cuda"):
            requested = "cpu"

        self.language = "ru"
        self._device = requested
        self._model_id = model_id or MODEL_ID
        local = get_local_model_path("gigaam", hub="hf")
        model = local or self._model_id
        self._torch = torch
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=requested,
            torch_dtype=torch.float32,
            chunk_length_s=30.0,
            batch_size=1,
        )
        log.info("GigaAM loaded from %s on %s (language=ru)", model, requested)

    def set_language(self, language: str):
        if language not in (None, "auto", "ru"):
            log.info("GigaAM is Russian-only; ignoring language=%s", language)
        self.language = "ru"

    def to_device(self, device: str):
        target = normalize_device(device)
        if target == "mps":
            try:
                if not self._torch.backends.mps.is_available():
                    target = "cpu"
            except (AttributeError, RuntimeError):
                target = "cpu"
        try:
            model = getattr(self._pipe, "model", None)
            if model is not None and hasattr(model, "to"):
                model.to(target, dtype=self._torch.float32)
            if hasattr(self._pipe, "device"):
                self._pipe.device = target
            self._device = target
            return True
        except Exception as exc:
            log.warning("GigaAM device switch failed: %s", exc)
            return False

    def unload(self):
        pipe = getattr(self, "_pipe", None)
        self._pipe = None
        if pipe is not None:
            try:
                model = getattr(pipe, "model", None)
                if model is not None and hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                log.debug("GigaAM model CPU release failed", exc_info=True)
        gc.collect()
        try:
            from torch_backend import empty_cache

            empty_cache(self._device)
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict | None:
        if self._pipe is None:
            raise RuntimeError("GigaAM engine is unloaded")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        with self._torch.inference_mode():
            result = self._pipe(audio)
        text = (result or {}).get("text", "").strip()
        if not text:
            return None
        return {"text": text, "language": "ru", "language_name": "ru"}

