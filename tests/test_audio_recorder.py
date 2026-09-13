import wave

import numpy as np

from audio_recorder import AudioRecorder


def test_audio_recorder_writes_wav_and_retains_wav_when_ffmpeg_missing(tmp_path):
    recorder = AudioRecorder(tmp_path, ffmpeg_path=str(tmp_path / "missing-ffmpeg"))
    recorder.start("20260101_120000", quality="speech")
    recorder.push(np.zeros(1600, dtype=np.float32), 16000, 1)
    result = recorder.finish(timeout=5)

    assert result.wav is not None
    assert result.mp3 is None
    assert result.status == "failed"
    with wave.open(result.wav, "rb") as source:
        assert source.getframerate() == 16000
        assert source.getnchannels() == 1
        assert source.getnframes() == 1600


def test_audio_recorder_can_start_a_second_session(tmp_path):
    recorder = AudioRecorder(tmp_path, ffmpeg_path=str(tmp_path / "missing"))
    recorder.start("20260101_120000")
    recorder.push(np.ones(320, dtype=np.float32), 16000, 1)
    first = recorder.finish(timeout=5)
    recorder.start("20260101_120001")
    recorder.push(np.ones(320, dtype=np.float32), 16000, 1)
    second = recorder.finish(timeout=5)
    assert first.wav != second.wav
