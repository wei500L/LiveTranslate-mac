"""Download-dialog plumbing: global-state restoration and cancellation (A9).

Only the non-Qt pieces are exercised here — _StderrCapture's file interface and
_OutputCapture's install/restore pairing. Those are the parts that corrupted
process-global state, and they need no QApplication.
"""

import io
import logging
import sys

import pytest

dialogs = pytest.importorskip("dialogs", reason="dialogs.py needs PyQt6")


def test_stderr_capture_implements_the_full_minimal_file_interface():
    """Download code probes fileno/encoding/errors; an AttributeError on any of
    them lands mid-download."""
    original = sys.__stderr__
    capture = dialogs._StderrCapture(lambda _line: None, original)

    assert capture.isatty() is False
    assert isinstance(capture.encoding, str)
    assert isinstance(capture.errors, str)
    capture.flush()
    # fileno may legitimately raise when the original has none, but never
    # AttributeError.
    try:
        capture.fileno()
    except OSError:
        pass


def test_stderr_capture_without_an_original_still_answers_probes():
    capture = dialogs._StderrCapture(lambda _line: None, None)
    assert capture.encoding == "utf-8"
    assert capture.errors == "replace"
    with pytest.raises(OSError):
        capture.fileno()


def test_stderr_capture_forwards_cleaned_lines():
    lines = []
    capture = dialogs._StderrCapture(lines.append, io.StringIO())
    capture.write("\x1b[32mDownloading\x1b[0m 12%\n  \nnext\n")
    assert lines == ["Downloading 12%", "next"]


def test_detached_capture_stops_forwarding():
    lines = []
    capture = dialogs._StderrCapture(lines.append, io.StringIO())
    capture.detach()
    capture.write("ignored\n")
    assert lines == []


def test_a_destroyed_receiver_disconnects_itself():
    """Emitting into a deleted QDialog raises RuntimeError in PyQt; that must
    not propagate into whatever happened to be writing to stderr."""

    def boom(_line):
        raise RuntimeError("wrapped C/C++ object has been deleted")

    capture = dialogs._StderrCapture(boom, io.StringIO())
    capture.write("first\n")  # must not raise
    capture.write("second\n")


def test_output_capture_restores_stderr_and_the_log_handler():
    before_stderr = sys.stderr
    before_handlers = list(logging.getLogger().handlers)

    capture = dialogs._OutputCapture(lambda _line: None)
    capture.install()
    assert sys.stderr is not before_stderr
    assert len(logging.getLogger().handlers) == len(before_handlers) + 1

    capture.restore()
    assert sys.stderr is before_stderr
    assert logging.getLogger().handlers == before_handlers


def test_output_capture_restore_is_idempotent():
    before = sys.stderr
    capture = dialogs._OutputCapture(lambda _line: None)
    capture.install()
    capture.restore()
    capture.restore()
    assert sys.stderr is before


def test_output_capture_restores_on_an_exception():
    before = sys.stderr
    with pytest.raises(ValueError):
        with dialogs._OutputCapture(lambda _line: None):
            raise ValueError("download blew up")
    assert sys.stderr is before


def test_output_capture_does_not_clobber_a_later_replacement():
    before = sys.stderr
    capture = dialogs._OutputCapture(lambda _line: None)
    capture.install()
    other = io.StringIO()
    sys.stderr = other
    try:
        capture.restore()
        assert sys.stderr is other
    finally:
        sys.stderr = before


def test_wizard_defaults_come_from_config_yaml():
    """The wizard used to hardcode vad_threshold=0.3 while config.yaml said
    0.5, so a first launch disagreed with every later one."""
    import yaml

    defaults = dialogs._config_asr_defaults()
    config = yaml.safe_load(dialogs.CONFIG_FILE.read_text(encoding="utf-8"))
    assert defaults["vad_threshold"] == config["asr"]["vad_threshold"]
    assert defaults["min_speech_duration"] == config["asr"]["min_speech_duration"]
