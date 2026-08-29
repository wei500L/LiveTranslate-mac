import sys
from types import SimpleNamespace

import numpy as np
import pytest

import asr_gigaam
import asr_worker
import model_manager
import platform_app
import torch_backend


def test_subtitle_window_offscreen_flags_and_geometry(monkeypatch):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from subtitle_window import SubtitleWindow

    app = QApplication.instance() or QApplication([])
    window = SubtitleWindow({"window_width": 640})
    assert window.width() == 640
    assert window.windowFlags()
    window.set_click_through(True)
    window.set_click_through(False)
    window.close()


def test_dock_policy_is_noop_off_macos(monkeypatch):
    monkeypatch.setattr(platform_app.sys, "platform", "linux")
    assert platform_app.set_dock_visible(False) is False


def test_dock_policy_uses_appkit_activation_policy(monkeypatch):
    policies = []

    class App:
        def setActivationPolicy_(self, policy):
            policies.append(policy)
            return True

    monkeypatch.setattr(platform_app.sys, "platform", "darwin")
    monkeypatch.setitem(
        sys.modules,
        "AppKit",
        SimpleNamespace(
            NSApplication=SimpleNamespace(sharedApplication=lambda: App()),
            NSApplicationActivationPolicyAccessory=0,
            NSApplicationActivationPolicyRegular=1,
        ),
    )
    assert platform_app.set_dock_visible(True)
    assert policies == [1]


def test_gigaam_fake_loader_is_russian_only(monkeypatch, tmp_path):
    calls = {}

    class FakeTorch:
        float32 = "float32"
        backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))

        @staticmethod
        def inference_mode():
            class Ctx:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

            return Ctx()

    class FakeModel:
        def to(self, *args, **kwargs):
            calls["to"] = (args, kwargs)

    class FakePipe:
        model = FakeModel()
        device = "mps"

        def __call__(self, audio):
            calls["audio"] = audio
            return {"text": "Привет"}

    def fake_pipeline(*args, **kwargs):
        calls["pipeline"] = (args, kwargs)
        return FakePipe()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline)
    )
    monkeypatch.setattr(model_manager, "get_local_model_path", lambda *a, **k: None)
    monkeypatch.setattr(torch_backend, "_torch", lambda: FakeTorch)

    engine = asr_gigaam.GigaAMEngine()
    engine.set_language("en")
    result = engine.transcribe(np.ones(512, dtype=np.float32))
    assert result["language"] == "ru"
    assert engine.language == "ru"
    assert calls["pipeline"][1]["device"] == "mps"


def test_worker_selects_gigaam_without_ct2(monkeypatch):
    calls = {}

    class Engine:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def set_language(self, language):
            calls["language"] = language

    monkeypatch.setitem(sys.modules, "asr_gigaam", SimpleNamespace(GigaAMEngine=Engine))
    monkeypatch.setattr(asr_worker, "normalize_torch_device", lambda value: "mps")
    monkeypatch.setattr(asr_worker, "_parse_device", lambda value: ("cpu", 0))
    engine = asr_worker._load_engine(
        {"engine_type": "gigaam", "device": "mps", "language": "en"}
    )
    assert isinstance(engine, Engine)
    assert calls == {"device": "mps", "hub": "hf", "language": "ru"}


def test_gigaam_cache_is_hf_snapshot_only(monkeypatch, tmp_path):
    monkeypatch.setattr(model_manager, "MODELS_DIR", tmp_path)
    # This test is about GigaAM cache detection; whether the silero-vad wheel
    # happens to be installed in the running environment is unrelated noise.
    # Without this the assertion below passes locally and fails in CI, where
    # silero-vad is not installed and get_missing_models() reports Silero VAD.
    monkeypatch.setattr(model_manager, "_has_silero_pkg", lambda: True)
    snap = (
        tmp_path
        / "huggingface"
        / "hub"
        / "models--ai-sage--GigaAM-v3"
        / "snapshots"
        / "abc"
    )
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"weights")
    assert model_manager.is_asr_cached("gigaam", "", "hf")
    assert model_manager.get_missing_models("gigaam", "", "hf") == []
