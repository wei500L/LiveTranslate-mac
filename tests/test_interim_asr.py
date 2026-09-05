"""Incremental ASR state advance and interim-buffer discipline (B6).

The bug this covers: when every split sentence was short enough to buffer,
_do_interim_asr returned before trimming the audio and before updating the echo
tail. The next pass re-recognized the same audio, produced the same fragment,
and appended it to _interim_pending again — so "はい。" showed up N times.
"""

import pytest

main = pytest.importorskip(
    "main", reason="main.py needs torch + PyQt6, which the offline job skips"
)


class Buffered:
    """Just enough of LiveTranslateApp to drive the fragment buffer."""

    _INTERIM_PENDING_MAX = main.LiveTranslateApp._INTERIM_PENDING_MAX
    _buffer_interim_fragment = main.LiveTranslateApp._buffer_interim_fragment
    _reset_interim_state = main.LiveTranslateApp._reset_interim_state

    def __init__(self):
        self._interim_pending = ""
        self._interim_active = False
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""


def test_the_same_fragment_is_not_buffered_twice():
    app = Buffered()
    app._buffer_interim_fragment("はい。")
    app._buffer_interim_fragment("はい。")
    assert app._interim_pending == "はい。"


def test_distinct_fragments_still_accumulate():
    app = Buffered()
    app._buffer_interim_fragment("はい。")
    app._buffer_interim_fragment("そうです。")
    assert app._interim_pending == "はい。そうです。"


def test_the_pending_buffer_is_bounded():
    app = Buffered()
    for i in range(200):
        app._buffer_interim_fragment(f"fragment-{i} ")
    assert len(app._interim_pending) <= Buffered._INTERIM_PENDING_MAX
    # The newest content is what survives.
    assert "fragment-199" in app._interim_pending


def test_reset_clears_every_interim_field():
    app = Buffered()
    app._interim_pending = "leftover"
    app._interim_active = True
    app._interim_committed_tail = "tail"
    app._last_interim_samples = 999
    app._last_interim_check_time = 5.0
    app._reset_interim_state()
    assert app._interim_pending == ""
    assert app._interim_active is False
    assert app._interim_committed_tail == ""
    assert app._last_interim_samples == 0
    assert app._last_interim_check_time == 0.0


def test_vad_counts_density_discards_so_the_pipeline_can_reset():
    """A discarded segment emits no vad_flush, so the counter is the only
    signal the pipeline gets that the utterance ended."""
    import numpy as np

    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.threshold = 0.5
    vad.mode = "silero"
    vad.discarded_segments = 0
    vad._speech_buffer = [np.zeros(512, dtype=np.float32) for _ in range(8)]
    vad._confidence_history = [0.0] * 8  # all below threshold -> density 0
    vad._speech_samples = 512 * 8
    vad._is_speaking = True
    vad._silence_counter = 0
    vad._was_trimmed = False

    assert vad._flush_segment() is None
    assert vad.discarded_segments == 1
    assert vad._speech_buffer == []


def test_a_voiced_segment_is_not_counted_as_discarded():
    import numpy as np

    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.threshold = 0.5
    vad.mode = "silero"
    vad.discarded_segments = 0
    vad._speech_buffer = [np.zeros(512, dtype=np.float32) for _ in range(8)]
    vad._confidence_history = [0.9] * 8
    vad._speech_samples = 512 * 8
    vad._is_speaking = True
    vad._silence_counter = 0
    vad._was_trimmed = False

    assert vad._flush_segment() is not None
    assert vad.discarded_segments == 0


# --- The noise filter must not eat already-recognized fragments -------------


class InterimFinal:
    """_process_interim_final over stubs, to drive its filtering branches."""

    _process_interim_final = main.LiveTranslateApp._process_interim_final
    _buffer_interim_fragment = main.LiveTranslateApp._buffer_interim_fragment
    _INTERIM_PENDING_MAX = main.LiveTranslateApp._INTERIM_PENDING_MAX

    def __init__(self, asr_text, pending="", language="ru"):
        self._asr_text = asr_text
        self._interim_pending = pending
        self._interim_committed_tail = ""
        self._language = language
        self.emitted = []

    # --- the pieces _process_interim_final leans on ---
    def _run_asr(self, segment, kind):
        if self._asr_text is None:
            return None, 5.0
        return {"text": self._asr_text, "language": self._language}, 5.0

    def _strip_committed_overlap(self, text):
        return text

    def _get_asr_language_setting(self):
        return self._language

    def _process_segment_text(self, text, lang, asr_ms=0, **session_identity):
        # The production signature also carries the queue item's session
        # identity (work_id / generation / expected_session); the interim
        # stand-in only asserts on the emitted text.
        self.emitted.append(text)


def _segment(seconds):
    import numpy as np

    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_a_noisy_tail_does_not_discard_a_buffered_reply():
    """The buffered fragment was recognized from earlier audio that already
    passed the noise filter. Folding it in before the check meant a short reply
    plus a quiet tail vanished entirely."""
    app = InterimFinal(asr_text="аа", pending="Да.")
    app._process_interim_final(_segment(3.0))   # >=2s with <=3 alnum chars
    assert app.emitted == ["Да."]


def test_the_noisy_segment_text_itself_is_still_dropped():
    app = InterimFinal(asr_text="аа", pending="")
    app._process_interim_final(_segment(3.0))
    assert app.emitted == []


def test_a_good_segment_still_absorbs_the_buffered_fragment():
    app = InterimFinal(asr_text="продолжим урок", pending="Да. ")
    app._process_interim_final(_segment(3.0))
    assert app.emitted == ["Да. продолжим урок"]


def test_a_short_segment_is_not_subject_to_the_noise_filter():
    """The filter only applies from 2s up; a brief clear utterance stays."""
    app = InterimFinal(asr_text="да", pending="")
    app._process_interim_final(_segment(1.0))
    assert app.emitted == ["да"]


def test_an_empty_result_still_flushes_the_buffer():
    app = InterimFinal(asr_text=None, pending="Спасибо.")
    app._process_interim_final(_segment(3.0))
    assert app.emitted == ["Спасибо."]


def test_the_pending_buffer_is_always_consumed():
    for asr_text, pending in (("аа", "Да."), (None, "Да."), ("текст", "Да.")):
        app = InterimFinal(asr_text=asr_text, pending=pending)
        app._process_interim_final(_segment(3.0))
        assert app._interim_pending == "", (asr_text, pending)


# --- Echo dedup: catch a replay, never delete a repeated word ---------------


class Echo:
    _strip_committed_overlap = main.LiveTranslateApp._strip_committed_overlap
    _is_substantial_echo = main.LiveTranslateApp._is_substantial_echo
    _is_unspaced_script = main.LiveTranslateApp._is_unspaced_script
    _ECHO_BOUNDARY = main.LiveTranslateApp._ECHO_BOUNDARY
    _ECHO_MIN_UNSPACED = main.LiveTranslateApp._ECHO_MIN_UNSPACED

    def __init__(self, tail):
        self._interim_committed_tail = tail


def test_a_multi_word_replay_is_stripped():
    """Committed text always ends in sentence punctuation, so matching it
    verbatim meant this never fired for anything."""
    echo = Echo("...говорит нам о поведении графика.")
    assert echo._strip_committed_overlap(
        "о поведении графика на этом промежутке"
    ) == "на этом промежутке"


def test_a_repeated_leading_word_is_kept():
    """"...производную функции. Функции бывают..." is ordinary speech, and
    deleting that word costs the sentence its subject."""
    echo = Echo("Рассмотрим производную функции.")
    text = "Функции бывают разные."
    assert echo._strip_committed_overlap(text) == text


def test_a_single_word_replay_is_also_kept():
    """Textually identical to the case above, so it is kept on purpose: a
    duplicated word is readable, a deleted one is not recoverable."""
    echo = Echo("...говорит нам о поведении графика.")
    text = "графика на этом промежутке"
    assert echo._strip_committed_overlap(text) == text


def test_an_unspaced_script_uses_a_length_threshold():
    echo = Echo("今天我们学习函数的导数和积分。")
    assert echo._strip_committed_overlap(
        "函数的导数和积分都很重要。"
    ) == "都很重要。"


def test_a_short_unspaced_repeat_is_kept():
    echo = Echo("这就是函数的导数。")
    text = "导数的符号说明什么？"
    assert echo._strip_committed_overlap(text) == text


def test_japanese_is_treated_as_unspaced():
    echo = Echo("今日は関数の微分を学びます。")
    assert echo._strip_committed_overlap("関数の微分を学びますから") == "から"


def test_cyrillic_is_not_treated_as_unspaced():
    """`not text.isascii()` was the old proxy, and it classified a single
    Russian word as a multi-word phrase."""
    assert main.LiveTranslateApp._is_unspaced_script("функции") is False
    assert main.LiveTranslateApp._is_unspaced_script("函数的导数") is True
    assert main.LiveTranslateApp._is_unspaced_script("関数の微分") is True
    assert main.LiveTranslateApp._is_unspaced_script("hello") is False


def test_no_committed_tail_leaves_text_alone():
    assert Echo("")._strip_committed_overlap("anything") == "anything"


def test_a_tail_of_only_punctuation_leaves_text_alone():
    assert Echo("...")._strip_committed_overlap("текст") == "текст"


# --- VAD silence mode ------------------------------------------------------


def _vad_for_settings():
    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad._chunk_duration = 0.032
    vad._silence_mode = "auto"
    vad._fixed_silence_dur = 0.8
    vad._silence_limit = vad._seconds_to_chunks(0.35)  # adaptive drove it down
    vad.mode = "silero"
    vad.threshold = 0.5
    vad.energy_threshold = 0.02
    vad.min_speech_samples = 16000
    vad.max_speech_samples = 128000
    return vad


def test_switching_to_fixed_applies_the_fixed_duration():
    """The limit used to be recomputed only inside the silence_duration branch,
    so a mode change alone left the adaptive value in place."""
    vad = _vad_for_settings()
    vad.update_settings({"silence_mode": "fixed"})
    assert vad._silence_limit == vad._seconds_to_chunks(0.8)


def test_switching_to_fixed_with_a_duration_uses_it():
    vad = _vad_for_settings()
    vad.update_settings({"silence_mode": "fixed", "silence_duration": 1.5})
    assert vad._silence_limit == vad._seconds_to_chunks(1.5)


def test_auto_mode_leaves_the_adaptive_limit_alone():
    """In auto the limit belongs to _update_adaptive_limit."""
    vad = _vad_for_settings()
    before = vad._silence_limit
    vad.update_settings({"silence_mode": "auto", "silence_duration": 1.5})
    assert vad._silence_limit == before


def test_changing_the_duration_while_fixed_takes_effect():
    vad = _vad_for_settings()
    vad.update_settings({"silence_mode": "fixed", "silence_duration": 0.8})
    vad.update_settings({"silence_duration": 2.0})
    assert vad._silence_limit == vad._seconds_to_chunks(2.0)
