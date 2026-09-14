"""A persisted vad_threshold below the slider's new floor must be migrated.

The slider's range was floored at 5% because threshold 0.0 makes
`confidence >= threshold` always true: every chunk counts as speech, silence
never accumulates, and no pause ever splits a segment -- the VAD is off while
the UI still says it is on.

Clamping alone is not enough, and the gap is invisible from the UI. The panel
calls setValue() *before* connecting the handler, so a persisted 0.0 leaves
the slider at 5, the label reading "0%", _current_settings still holding 0.0
and the engine still running wide open: four answers to one question. Anyone
who has 0.0 on disk got there hunting for a setting that works on a quiet
speaker, which is the person the floor exists for.
"""

import os

import pytest

pytest.importorskip("PyQt6", reason="control_panel needs PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _vad_tab_with(qt_app, saved_threshold):
    """Run the real _create_vad_tab over a bare panel instance.

    Constructing a whole ControlPanel pulls in model detection and the whole
    tab set; this is the one method under test, driven on the state it reads.
    """
    from PyQt6.QtWidgets import QWidget

    import control_panel

    panel = control_panel.ControlPanel.__new__(control_panel.ControlPanel)
    QWidget.__init__(panel)
    panel._current_settings = {
        "vad_threshold": saved_threshold,
        "vad_mode": "silero",
        "energy_threshold": 0.02,
        "min_speech_duration": 2.0,
        "max_speech_duration": 15.0,
        "silence_mode": "auto",
        "silence_duration": 0.8,
        "incremental_asr": True,
        "interim_interval": 1.0,
        "asr_engine": "funasr",
        "models": [],
    }
    panel._save_timer = None
    panel._config = {
        "asr": {"language": "auto"},
        "audio": {"mic_device": None, "system_audio": "disabled"},
    }
    # Hold the tab: it owns the slider, and nothing else references it.
    panel._vad_tab_keepalive = panel._create_vad_tab()
    return panel


def test_a_persisted_zero_threshold_is_migrated_not_just_clamped(qt_app):
    panel = _vad_tab_with(qt_app, 0.0)

    assert panel._vad_threshold_slider.value() == 5
    assert panel._vad_threshold_label.text() == "5%"
    assert panel._current_settings["vad_threshold"] == 0.05


def test_a_threshold_above_the_floor_is_left_alone(qt_app):
    panel = _vad_tab_with(qt_app, 0.5)

    assert panel._vad_threshold_slider.value() == 50
    assert panel._vad_threshold_label.text() == "50%"
    assert panel._current_settings["vad_threshold"] == 0.5
