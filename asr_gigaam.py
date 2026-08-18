"""GigaAM v3 Russian speech recognizer."""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import wave

import numpy as np

from torch_backend import normalize_device

log = logging.getLogger("LiveTranslate.GigaAM")

MODEL_ID = "ai-sage/GigaAM-v3"
MODEL_REVISION = "e2e_rnnt"
SAMPLE_RATE = 16_000


class GigaAMEngine:
    """Russian-only GigaAM v3 model with MPS -> CPU load fallback."""

    def __init__(self, device="mps", hub="hf", model_id: str | None = None):
        import torch

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
        self._torch = torch
        self._model_id = model_id or MODEL_ID
        local = get_local_model_path("gigaam", hub="hf")
        model_source = local or self._model_id
        self._pipe = None

        # Keep a small compatibility path for test doubles and very old
        # transformers builds; supported installations use AutoModel below.
        try:
            from transformers import AutoModel
        except ImportError:
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model_source,
                device=requested,
                torch_dtype=torch.float32,
                chunk_length_s=25.0,
                batch_size=1,
            )
            self._model = None
            self._device = requested
            log.warning("Using legacy GigaAM pipeline compatibility path")
            return

        try:
            self._model = self._load_model(AutoModel, model_source, requested)
            self._device = requested
        except Exception:
            if requested != "mps":
                raise
            log.warning(
                "GigaAM failed to load on MPS; retrying on CPU", exc_info=True
            )
            self._release_model()
            self._model = self._load_model(AutoModel, model_source, "cpu")
            self._device = "cpu"

        log.info(
            "GigaAM loaded from %s revision=%s on %s (language=ru)",
            model_source,
            MODEL_REVISION,
            self._device,
        )

    def _load_model(self, auto_model, model_source: str, device: str):
        model = auto_model.from_pretrained(
            model_source,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        try:
            moved = model.to(device)
            return model if moved is None else moved
        except Exception:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            gc.collect()
            raise

    def _release_model(self):
        model = getattr(self, "_model", None)
        self._model = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
        self._pipe = None
        gc.collect()

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
            model = self._model
            if model is not None:
                model.to(target)
            elif self._pipe is not None:
                pipe_model = getattr(self._pipe, "model", None)
                if pipe_model is not None and hasattr(pipe_model, "to"):
                    pipe_model.to(target)
                if hasattr(self._pipe, "device"):
                    self._pipe.device = target
            self._device = target
            return True
        except Exception as exc:
            log.warning("GigaAM device switch failed: %s", exc)
            return False

    def unload(self):
        pipe = self._pipe
        self._pipe = None
        if pipe is not None:
            try:
                model = getattr(pipe, "model", None)
                if model is not None and hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                log.debug("GigaAM pipeline CPU release failed", exc_info=True)
        self._release_model()
        try:
            from torch_backend import empty_cache

            empty_cache(self._device)
        except Exception:
            pass

    @staticmethod
    def _write_wav(audio: np.ndarray) -> str:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        fd, path = tempfile.mkstemp(prefix="livetranslate-gigaam-", suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm.tobytes())
        except Exception:
            os.unlink(path)
            raise
        return path

    def transcribe(self, audio: np.ndarray, **kwargs) -> dict | None:
        if self._model is None and self._pipe is None:
            raise RuntimeError("GigaAM engine is unloaded")
        if self._pipe is not None:
            with self._torch.inference_mode():
                result = self._pipe(audio)
            text = (result or {}).get("text", "").strip()
            return (
                {"text": text, "language": "ru", "language_name": "ru"}
                if text
                else None
            )
        wav_path = self._write_wav(audio)
        try:
            with self._torch.inference_mode():
                text = self._model.transcribe(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except FileNotFoundError:
                pass
        text = str(text or "").strip()
        if not text:
            return None
        return {"text": text, "language": "ru", "language_name": "ru"}
