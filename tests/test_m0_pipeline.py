from pathlib import Path

import numpy as np

import asr_worker
from audio_capture_base import FakeAudioCapture
from transcript_writer import TranscriptWriter


class _FakeVAD:
    def process(self, audio):
        return audio if np.max(np.abs(audio)) > 0 else None


class _FakeASR:
    def transcribe(self, audio):
        assert audio.shape == (512,)
        return {"text": "hello", "language": "en", "language_name": "English"}


class _FakeTranslator:
    def translate_iter(self, text, source_language):
        assert (text, source_language) == ("hello", "en")
        yield "你"
        yield "你好"


class _FakeUI:
    def __init__(self):
        self.original = None
        self.partials = []
        self.final = None

    def add_message(self, msg_id, timestamp, original, source_lang, asr_ms):
        self.original = (msg_id, original, source_lang)

    def update_streaming(self, msg_id, partial):
        self.partials.append((msg_id, partial))

    def update_translation(self, msg_id, translated, translate_ms):
        self.final = (msg_id, translated)


def test_fake_m0_pipeline_reaches_streaming_ui_and_transcript(tmp_path: Path):
    source = FakeAudioCapture([np.ones(512, dtype=np.float32)])
    vad, asr, translator, ui = _FakeVAD(), _FakeASR(), _FakeTranslator(), _FakeUI()
    writer = TranscriptWriter(tmp_path)

    source.start()
    audio, mic_rms = source.get_audio(timeout=1)
    speech = vad.process(audio)
    result = asr.transcribe(speech)
    ui.add_message(1, "12:00:00", result["text"], result["language"], 1.0)
    writer.write_original(1, "12:00:00", result["text"])
    translated = None
    for translated in translator.translate_iter(result["text"], result["language"]):
        ui.update_streaming(1, translated)
    ui.update_translation(1, translated, 1.0)
    writer.write_translation(1, translated)
    writer.close()
    source.stop()

    assert mic_rms is None
    assert ui.original == (1, "hello", "en")
    assert ui.partials == [(1, "你"), (1, "你好")]
    assert ui.final == (1, "你好")
    all_transcript = next(tmp_path.glob("*_all.txt")).read_text(encoding="utf-8")
    assert "hello" in all_transcript
    assert "你好" in all_transcript


class _FakeConn:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    def recv(self):
        if not self.messages:
            raise EOFError
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


def test_worker_command_lifecycle_can_stop_and_start_again(monkeypatch):
    created = []

    class Engine(_FakeASR):
        def unload(self):
            self.unloaded = True

    def load_engine(config):
        engine = Engine()
        created.append(engine)
        return engine

    monkeypatch.setattr(asr_worker, "_load_engine", load_engine)
    for _ in range(2):
        conn = _FakeConn(
            [
                {"id": "1", "type": "transcribe", "payload": {"audio": np.ones(512, dtype=np.float32)}},
                {"id": "2", "type": "shutdown", "payload": {}},
            ]
        )
        asr_worker.worker_main(conn, {"engine_type": "fake", "device": "cpu"})
        assert [message["type"] for message in conn.sent] == ["ready", "result", "shutdown"]
        assert conn.closed

    assert len(created) == 2
    assert all(engine.unloaded for engine in created)
