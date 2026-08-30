"""GigaAM-v3 e2e_rnnt Russian speech recognizer.

The integration follows the official Hugging Face Transformers loading path.
Long-form ``transcribe_longform`` is intentionally not part of this backend yet;
the ASR worker supplies VAD-sized short segments. That also keeps
``pyannote-audio`` (and an ``HF_TOKEN``) out of the dependency set — the model's
remote code only imports it inside that unused path.
"""

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
    """Russian GigaAM-v3 e2e_rnnt model with MPS -> CPU fallback."""

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
        self._audio_loader_installed = self._install_audio_loader()

    def _load_model(self, auto_model, model_source: str, device: str):
        model = auto_model.from_pretrained(
            model_source,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        try:
            # Disable training-only paths and dropout for lower latency and
            # deterministic inference. Remote-code models may not expose
            # ``eval`` in test doubles, so keep this capability-checked.
            eval_model = getattr(model, "eval", None)
            if eval_model is not None:
                eval_model()
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

    @staticmethod
    def _load_wav_without_ffmpeg(audio_path: str, sample_rate: int = SAMPLE_RATE):
        """Decode the WAV files produced by ``_write_wav`` without ffmpeg.

        GigaAM's Hugging Face remote code shells out to ffmpeg even for a
        standard PCM WAV.  LiveTranslate already normalizes every segment to
        mono 16 kHz PCM, so using the stdlib decoder here removes that brittle
        external-process dependency while preserving the model's expected
        float tensor input.
        """
        import torch

        with wave.open(audio_path, "rb") as wav:
            channels = wav.getnchannels()
            source_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        if sample_width == 2:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 1:
            samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 4:
            samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if source_rate != sample_rate and samples.size:
            duration = samples.size / float(source_rate)
            target_size = max(1, round(duration * sample_rate))
            source_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
            target_x = np.linspace(0.0, 1.0, target_size, endpoint=False)
            samples = np.interp(target_x, source_x, samples).astype(np.float32)

        return torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))

    @staticmethod
    def _load_audio_input(audio, sample_rate: int = SAMPLE_RATE):
        """Accept LiveTranslate's normalized in-memory PCM or a WAV path."""
        import torch

        if isinstance(audio, (str, os.PathLike)):
            return GigaAMEngine._load_wav_without_ffmpeg(str(audio), sample_rate)
        if isinstance(audio, torch.Tensor):
            return audio.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))

    def _install_audio_loader(self):
        """Patch the loaded remote-code model to use the local WAV decoder."""
        root = getattr(self, "_model", None)
        candidates = [root, getattr(root, "model", None)]
        patched = False
        for model in candidates:
            if model is None:
                continue
            for method_name in ("transcribe", "prepare_wav"):
                method = getattr(model, method_name, None)
                function = getattr(method, "__func__", method)
                globals_dict = getattr(function, "__globals__", None)
                if not globals_dict or "load_audio" not in globals_dict:
                    continue
                if globals_dict.get("load_audio") is not self._load_audio_input:
                    globals_dict["load_audio"] = self._load_audio_input
                    patched = True
        if patched:
            log.info("GigaAM audio loader: using in-memory PCM input (WAV fallback enabled)")
        return patched

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
        if getattr(self, "_audio_loader_installed", False):
            try:
                with self._torch.inference_mode():
                    text = self._model.transcribe(
                        np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
                    )
            except Exception:
                log.debug("GigaAM in-memory input rejected; using WAV fallback", exc_info=True)
                text = None
            else:
                text = str(text or "").strip()
                if not text:
                    return None
                return {"text": text, "language": "ru", "language_name": "ru"}

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
