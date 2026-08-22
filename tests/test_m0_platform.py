import sys
from types import SimpleNamespace

import numpy as np
import pytest

from audio_capture_base import AudioCaptureBase, FakeAudioCapture
from audio_capture_pyaudio import PyAudioCapture
from asr_worker import _parse_device
from platform_fonts import (
    default_cjk_font_family,
    default_mono_font_family,
    default_ui_font_family,
)
from platform_config import normalize_config
from torch_backend import normalize_device
import platform_clickthrough
import torch_backend
import platform_permissions
import asr_worker
from platform_permissions import MicrophonePermissionDeniedError, PlatformUnavailableError


def test_audio_base_downmixes_resamples_and_emits_fixed_blocks():
    stereo = np.ones((2000, 2), dtype=np.float32)
    mono = AudioCaptureBase.resample_to_mono(stereo, 2, 48000, 16000)
    capture = AudioCaptureBase()
    capture.push_audio(mono)
    item = capture.get_audio(timeout=0)
    assert item is not None
    audio, mic_rms = item
    assert audio.shape == (512,)
    assert audio.dtype == np.float32
    assert mic_rms is None

    native_capture = AudioCaptureBase()
    native_capture.push_audio(stereo, native_channels=2, native_rate=48000)
    native_item = native_capture.get_audio(timeout=0)
    assert native_item[0].shape == (512,)

    channel_first = np.vstack(
        [np.ones(512, dtype=np.float32), np.zeros(512, dtype=np.float32)]
    )
    assert np.allclose(
        AudioCaptureBase.resample_to_mono(channel_first, 2, 16000, 16000), 0.5
    )
    assert np.allclose(
        AudioCaptureBase.resample_to_mono(
            channel_first.T.tobytes(), 2, 16000, 16000
        ),
        0.5,
    )


def test_audio_base_mixes_mic_and_drops_incomplete_tail():
    capture = AudioCaptureBase(queue_size=1)
    capture.push_audio(np.zeros(512, dtype=np.float32), mic_audio=np.ones(512, dtype=np.float32))
    audio, mic_rms = capture.get_audio(timeout=0)
    assert np.allclose(audio, 1.0)
    assert mic_rms == 1.0
    capture.push_audio(np.ones(10, dtype=np.float32))
    capture.flush()
    assert capture.metrics()["output_blocks"] == 1

    backpressured = AudioCaptureBase(queue_size=1)
    backpressured.push_audio(np.zeros(512, dtype=np.float32))
    backpressured.push_audio(np.ones(512, dtype=np.float32))
    latest, _ = backpressured.get_audio(timeout=0)
    assert np.allclose(latest, 1.0)
    assert backpressured.metrics()["dropped_blocks"] == 1


def test_mps_is_cpu_for_ct2_and_other_devices_are_stable():
    assert _parse_device("mps") == ("cpu", 0)
    assert _parse_device("cpu") == ("cpu", 0)
    assert normalize_device("mps", for_ct2=True) == "cpu"


def test_font_defaults_are_nonempty():
    assert default_ui_font_family()
    assert default_mono_font_family()
    assert default_cjk_font_family()


def test_platform_font_and_device_fallbacks_without_torch(monkeypatch):
    monkeypatch.setattr(torch_backend, "_torch", lambda: None)
    assert torch_backend.cuda_available() is False
    assert normalize_device("auto") == "cpu"
    assert normalize_device("mps") == "cpu"
    assert torch_backend.available_devices() == ["cpu"]
    assert default_ui_font_family("darwin") == ".AppleSystemUIFont"
    assert default_mono_font_family("darwin") == "Menlo"
    assert default_cjk_font_family("darwin") == "PingFang SC"


def test_fake_audio_source_is_reusable_for_offline_smoke_pipeline():
    source = FakeAudioCapture([np.ones(512, dtype=np.float32)])
    source.start()
    item = source.get_audio(timeout=1)
    source.stop()
    assert item[0].shape == (512,)


def test_disabled_microphone_keeps_the_32ms_silence_clock_without_pyaudio():
    capture = PyAudioCapture(
        device="__disabled__",
        mic_device=None,
        system_audio="disabled",
        require_permission=False,
    )
    capture.start()
    audio, mic_rms = capture.get_audio(timeout=1)
    capture.stop()
    capture.stop()
    assert audio.shape == (512,)
    assert np.count_nonzero(audio) == 0
    assert mic_rms is None


def test_m0_rejects_system_audio_until_screencapturekit_backend_exists():
    capture = PyAudioCapture(system_audio="enabled", require_permission=False)
    with pytest.raises(PlatformUnavailableError, match="ScreenCaptureKit"):
        capture.start()


def test_microphone_permission_failure_has_a_distinct_error_type(monkeypatch):
    monkeypatch.setattr(platform_permissions, "microphone_permission_status", lambda: "denied")
    with pytest.raises(MicrophonePermissionDeniedError):
        platform_permissions.ensure_microphone_permission()


def test_legacy_windows_config_migrates_to_mac_defaults_without_losing_fields():
    config = {
        "audio": {"device": "__disabled__", "unknown_audio_key": 7},
        "asr": {"device": "cuda", "unknown_asr_key": "kept"},
        "subtitle": {"font_family": "auto"},
        "future_section": {"enabled": True},
    }
    result = normalize_config(
        config, platform_name="darwin", mps_is_available=True
    )
    assert result["audio"]["system_audio"] == "disabled"
    assert result["audio"]["mic_device"] == "__default__"
    assert result["asr"]["device"] == "mps"
    assert result["subtitle"]["font_family"] == "PingFang SC"
    assert result["audio"]["unknown_audio_key"] == 7
    assert result["asr"]["unknown_asr_key"] == "kept"
    assert result["future_section"] == {"enabled": True}
    assert normalize_config(
        result, platform_name="darwin", mps_is_available=True
    ) == result


def test_platform_config_keeps_windows_loopback_and_cuda_defaults():
    config = {"audio": {}, "asr": {"device": "auto"}}
    normalize_config(config, platform_name="win32", cuda_is_available=True)
    assert config["audio"]["system_audio"] == "enabled"
    assert config["audio"]["mic_device"] is None
    assert config["asr"]["device"] == "cuda"


def test_macos_clickthrough_uses_native_window_without_win32(monkeypatch):
    class NativeWindow:
        def __init__(self):
            self.enabled = False

        def setIgnoresMouseEvents_(self, enabled):
            self.enabled = enabled

        def ignoresMouseEvents(self):
            return self.enabled

    native = NativeWindow()
    window = SimpleNamespace(windowHandle=lambda: native)
    monkeypatch.setattr(platform_clickthrough.sys, "platform", "darwin")
    assert platform_clickthrough.set_click_through(window, True)
    assert platform_clickthrough.get_click_through(window) is True


def test_fake_torch_reports_mps_memory_and_never_enables_fp16(monkeypatch):
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True)
        ),
        cuda=SimpleNamespace(is_available=lambda: False),
        mps=SimpleNamespace(current_allocated_memory=lambda: 4 * 1024**2),
    )
    monkeypatch.setattr(torch_backend, "_torch", lambda: fake_torch)
    assert torch_backend.normalize_device("mps") == "mps"
    assert torch_backend.normalize_device("mps", for_ct2=True) == "cpu"
    assert torch_backend.device_supports_fp16("mps") is False
    assert torch_backend.accelerator_memory("mps") == (4.0, 4.0, "MPS")


def test_worker_routes_mps_to_torch_but_cpu_int8_to_whisper(monkeypatch, tmp_path):
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(torch_backend, "_torch", lambda: fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "model_manager",
        SimpleNamespace(MODELS_DIR=tmp_path, apply_cache_env=lambda: None),
    )

    calls = {}

    class Whisper:
        def __init__(self, **kwargs):
            calls["whisper"] = kwargs

        def set_language(self, language):
            pass

    class FunASR:
        def __init__(self, **kwargs):
            calls["funasr"] = kwargs

        def set_language(self, language):
            pass

    monkeypatch.setitem(sys.modules, "asr_engine", SimpleNamespace(ASREngine=Whisper))
    monkeypatch.setitem(sys.modules, "asr_funasr", SimpleNamespace(FunASREngine=FunASR))

    asr_worker._load_engine(
        {
            "engine_type": "whisper",
            "device": "mps",
            "compute_type": "float16",
            "model_size": "tiny",
        }
    )
    asr_worker._load_engine(
        {"engine_type": "funasr", "device": "mps", "funasr_model": "sensevoice-small"}
    )

    assert calls["whisper"]["device"] == "cpu"
    assert calls["whisper"]["compute_type"] == "int8"
    assert calls["funasr"]["device"] == "mps"
