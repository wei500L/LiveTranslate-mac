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


def _vad_for_flush(confidences):
    """A VADProcessor positioned exactly as _flush_segment finds it."""
    import numpy as np

    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.threshold = 0.5
    vad.mode = "silero"
    vad._chunk_duration = 0.032
    vad._segment_threshold = 0.5
    vad.discarded_segments = 0
    vad._buffer_epoch = 0
    vad._speech_buffer = [
        np.zeros(512, dtype=np.float32) for _ in range(len(confidences))
    ]
    vad._confidence_history = list(confidences)
    vad._speech_samples = 512 * len(confidences)
    vad._is_speaking = True
    vad._silence_counter = 0
    vad._was_trimmed = False
    return vad


def test_vad_counts_noise_discards_so_the_pipeline_can_reset():
    """A discarded segment emits no vad_flush, so the counter is the only
    signal the pipeline gets that the utterance ended."""
    vad = _vad_for_flush([0.0] * 8)  # nothing voiced at all

    assert vad._flush_segment() is None
    assert vad.discarded_segments == 1
    assert vad._speech_buffer == []


def test_a_voiced_segment_is_not_counted_as_discarded():
    vad = _vad_for_flush([0.9] * 8)

    assert vad._flush_segment() is not None
    assert vad.discarded_segments == 0


def test_a_short_utterance_buried_in_silence_is_not_discarded():
    """The field failure this replaces: a lecturer says one sentence and then
    pauses to write on the board. The segment is 20% voiced, which the old
    ratio test (`density < 0.25`) threw away wholesale -- 1.12s of real speech
    that never reached ASR. Absolute voiced duration is what matters."""
    vad = _vad_for_flush([0.9] * 35 + [0.0] * 143)  # 35/178, exactly as logged

    segment = vad._flush_segment()

    assert segment is not None
    assert vad.discarded_segments == 0


def test_a_flicker_of_voiced_audio_is_still_discarded():
    """The filter must still earn its keep: a couple of noise chunks below the
    voiced floor is not speech."""
    vad = _vad_for_flush([0.9] * 4 + [0.0] * 60)  # 0.128s voiced

    assert vad._flush_segment() is None
    assert vad.discarded_segments == 1


def test_noise_verdict_uses_the_threshold_the_segment_accumulated_under():
    """The slider can move mid-utterance. Judging a confidence history against
    a threshold it was never accumulated under discarded a 5.7s segment in the
    field (chunks accepted at 0.0, judged at 0.11)."""
    vad = _vad_for_flush([0.2] * 30 + [0.0] * 60)
    vad._segment_threshold = 0.11  # what the utterance was accepted under
    vad.threshold = 0.9  # what the user dragged the slider to since

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


# --- Adaptive silence: censored sampling ------------------------------------


def _vad_for_adaptive(limit_seconds=0.8):
    """A VADProcessor set up to run the silence-split branch of process_chunk
    with the Silero model stubbed out (confidence is fed in directly)."""
    import collections

    from vad_processor import VADProcessor

    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.threshold = 0.5
    vad.mode = "silero"
    vad._chunk_duration = 0.032
    vad._segment_threshold = 0.5
    vad.discarded_segments = 0
    vad._buffer_epoch = 0
    vad.last_confidence = 0.0
    vad.min_speech_samples = 16000
    vad.max_speech_samples = 128000
    vad._speech_buffer = []
    vad._confidence_history = []
    vad._speech_samples = 0
    vad._is_speaking = False
    vad._silence_counter = 0
    vad._was_trimmed = False
    vad._pre_speech_chunks = 3
    vad._pre_buffer = collections.deque(maxlen=3)
    vad._silence_mode = "auto"
    vad._fixed_silence_dur = 0.8
    vad._silence_limit = vad._seconds_to_chunks(limit_seconds)
    vad._progressive_tiers = [(3.0, 1.0), (6.0, 0.5), (10.0, 0.25)]
    vad._pause_history = collections.deque(maxlen=100)
    vad._adaptive_min = 0.3
    vad._adaptive_max = 2.0
    # Bypass the Silero model: _get_confidence is driven by this queue.
    vad._scripted = []
    vad._get_confidence = lambda chunk: vad._scripted.pop(0)
    return vad


def _speak_then_pause(vad, speech_chunks, pause_chunks):
    """Feed one utterance followed by a pause, returning any emitted segment."""
    import numpy as np

    chunk = np.zeros(512, dtype=np.float32)
    emitted = []
    vad._scripted = [0.9] * speech_chunks + [0.0] * pause_chunks
    for _ in range(speech_chunks + pause_chunks):
        seg = vad.process_chunk(chunk)
        if seg is not None:
            emitted.append(seg)
    return emitted


def test_a_pause_that_ended_the_segment_is_recorded():
    """The root-cause fix. A pause that reaches the limit ends the segment, so
    its true length is never observed -- but leaving it out of the history
    truncated the sample at the very value it was used to compute, and the
    limit ratcheted down 0.50 -> 0.30s in the field."""
    vad = _vad_for_adaptive(limit_seconds=0.8)
    limit_chunks = vad._silence_limit

    _speak_then_pause(vad, speech_chunks=60, pause_chunks=limit_chunks)

    assert len(vad._pause_history) == 1
    assert vad._pause_history[0] == limit_chunks * 0.032


def test_adaptive_silence_does_not_collapse_on_a_pause_heavy_speaker():
    """The censored-sampling spiral, reproduced.

    A speaker whose pauses straddle the limit: some resume before it (sampled
    by the old code) and some reach it and end the segment (invisible to the
    old code). Sampling only the first kind truncated the history at the very
    value it was used to compute, so P75 sat below the limit by construction
    and each update pushed it lower -- 0.50 -> 0.30s in the field, pinned to
    the floor, a 0.3s breath cutting every sentence. Recording the censored
    observation makes it a fixed point instead. Without the fix this walks to
    _adaptive_min; with it the limit stays where the speaker's rhythm is.
    """
    import numpy as np

    chunk = np.zeros(512, dtype=np.float32)
    vad = _vad_for_adaptive(limit_seconds=0.8)

    # Short mid-utterance pauses (0.32s, well under the limit) mixed with long
    # ones that end the segment. The short ones are what dragged the estimate
    # down; the long ones are the observations that were being thrown away.
    for _ in range(60):
        for pause_chunks in (10, 10, 40):
            vad._scripted = [0.9] * 40 + [0.0] * pause_chunks
            for _ in range(40 + pause_chunks):
                vad.process_chunk(chunk)

    # The broken code converges to 0.38s -- exactly the short pause (0.32) x
    # 1.2, because the short pauses are the only ones it ever saw. The long
    # ones must count too, which holds the limit up near where it started.
    limit_s = vad._silence_limit * 0.032
    assert limit_s >= 0.6, (
        f"adaptive limit walked down to {limit_s:.2f}s; the segment-ending "
        f"pauses are not reaching the estimate"
    )


def test_intra_word_dips_are_not_sampled_as_pauses():
    """Continuous speech is full of 0.1-0.2s dips between words, and they
    outnumber real sentence pauses roughly 2:1. Sampling them made P75 describe
    the gaps between words rather than between sentences -- measured against a
    known pause distribution, that alone keeps the estimate 1.4s too low even
    with censoring fixed."""
    import numpy as np

    vad = _vad_for_adaptive(limit_seconds=1.0)

    # 60 utterances, each with word-gaps well under the sample floor.
    chunk = np.zeros(512, dtype=np.float32)
    for _ in range(60):
        for dip_chunks in (4, 5, 6):  # 0.128-0.192s
            vad._scripted = [0.9] * 20 + [0.0] * dip_chunks
            for _ in range(20 + dip_chunks):
                vad.process_chunk(chunk)

    assert all(
        p >= vad._MIN_PAUSE_SAMPLE_SECONDS for p in vad._pause_history
    ), f"intra-word dips reached the estimate: {sorted(vad._pause_history)[:5]}"
    # And the limit must not have been dragged down by them. Compared in
    # chunks: _seconds_to_chunks rounds, so 31 chunks reads back as 0.992s.
    assert vad._silence_limit == vad._seconds_to_chunks(1.0)


def test_a_progressive_tier_cut_is_not_recorded_as_a_pause_sample():
    """A progressive tier shortens the limit when the buffer is long. That cut
    is our own impatience, not the speaker's rhythm, so feeding it back would
    drag the baseline down."""
    vad = _vad_for_adaptive(limit_seconds=1.0)
    base = vad._silence_limit
    # Past 6s, which is where the tiers actually start biting: the loop in
    # _get_effective_silence_limit applies the multiplier of the last tier
    # whose bound the buffer *reaches*, so <6s is still the full limit. The
    # previous version of this test used a 4.2s buffer and a pause of
    # base // 2, which is both under the tier and under the shortened limit
    # -- the silence branch never ran at all and the assertion held for a
    # reason unrelated to the guard.
    speech_chunks = int(6.5 * 16000 / 512)
    eff = max(1, round(base * 0.5))
    assert eff < base, "tier did not shorten the limit; test is vacuous again"

    _speak_then_pause(vad, speech_chunks=speech_chunks, pause_chunks=eff + 1)

    assert len(vad._pause_history) == 0


def test_fixed_mode_records_pauses_but_never_moves_the_limit():
    """In fixed mode the limit is the user's, so the estimator must not run --
    but the *sample* is still collected, exactly as the exact-pause branch
    has always done. Recording one kind and not the other left the history
    holding nothing but sub-limit pauses, which is the right-truncated sample
    that caused the original collapse; see the mode-flip test below."""
    vad = _vad_for_adaptive(limit_seconds=0.8)
    vad._silence_mode = "fixed"
    before = vad._silence_limit

    _speak_then_pause(vad, speech_chunks=60, pause_chunks=vad._silence_limit)

    assert list(vad._pause_history) == [before * 0.032]
    assert vad._silence_limit == before


def test_flipping_fixed_to_auto_does_not_collapse_the_limit():
    """The regression this guards: the censored branch used to be gated on
    "auto" while the exact branch was not, so a spell in fixed mode filled
    _pause_history with sub-limit samples only. The first update after the
    user flipped the combo back to auto then computed P75 over a sample that
    was right-truncated by construction and dropped the limit in one step --
    measured 1.00s -> 0.32s, straight back into the original bug."""
    vad = _vad_for_adaptive(limit_seconds=1.0)
    vad._silence_mode = "fixed"
    before = vad._silence_limit

    # A spell in fixed mode: utterances split by pauses that reach the limit,
    # plus short mid-utterance ones that resume before it.
    for _ in range(30):
        _speak_then_pause(vad, speech_chunks=40, pause_chunks=8)
        _speak_then_pause(vad, speech_chunks=40, pause_chunks=before + 2)

    vad._silence_mode = "auto"
    _speak_then_pause(vad, speech_chunks=40, pause_chunks=8)

    limit_s = vad._silence_limit * 0.032
    assert limit_s >= 0.9, (
        f"limit collapsed to {limit_s:.2f}s on the fixed->auto flip; the "
        f"history is missing its censored observations"
    )


def test_changing_the_duration_while_fixed_takes_effect():
    vad = _vad_for_settings()
    vad.update_settings({"silence_mode": "fixed", "silence_duration": 0.8})
    vad.update_settings({"silence_duration": 2.0})
    assert vad._silence_limit == vad._seconds_to_chunks(2.0)


# --- Buffer epoch: the interim trim must not outlive its utterance ---------


def test_trim_front_is_refused_after_the_buffer_turned_over():
    """[B1] _do_interim_asr peeks under the lock, drops it for the whole of
    recognition, then trims. In that window the capture thread can end the
    utterance -- and the trim, measured against audio that is already gone,
    used to be applied to whatever was in the buffer by then. Measured: the
    next sentence (1.28s) trimmed to nothing, with _was_trimmed left set on a
    segment that was never trimmed."""
    import numpy as np

    chunk = np.zeros(512, dtype=np.float32)
    vad = _vad_for_adaptive(limit_seconds=0.8)

    # An utterance the interim pass peeks.
    vad._scripted = [0.9] * 94
    for _ in range(94):
        vad.process_chunk(chunk)
    peeked = vad.peek_buffer()
    assert peeked is not None
    _audio, duration, epoch = peeked

    # The capture thread ends it while "recognition" is running.
    vad._scripted = [0.0] * (vad._silence_limit + 1)
    for _ in range(vad._silence_limit + 1):
        vad.process_chunk(chunk)

    # ...and the next sentence starts.
    vad._scripted = [0.9] * 40
    for _ in range(40):
        vad.process_chunk(chunk)
    before = vad._speech_samples
    assert before > 0

    applied = vad.trim_front(int(0.7 * duration * 16000), epoch)

    assert applied is False
    assert vad._speech_samples == before, "the next sentence was trimmed"
    assert vad._was_trimmed is False


def test_trim_front_still_applies_within_the_same_utterance():
    """The guard must not break the normal interim path: same utterance, same
    epoch, the trim goes through."""
    import numpy as np

    chunk = np.zeros(512, dtype=np.float32)
    vad = _vad_for_adaptive(limit_seconds=0.8)
    vad._scripted = [0.9] * 94
    for _ in range(94):
        vad.process_chunk(chunk)

    _audio, _duration, epoch = vad.peek_buffer()
    before = vad._speech_samples

    assert vad.trim_front(512 * 20, epoch) is True
    assert vad._speech_samples < before
    assert vad._was_trimmed is True


def test_a_split_also_invalidates_an_in_flight_trim():
    """Max-duration split replaces the buffer with the remainder, so a trim
    measured against the pre-split audio would over-trim what is left."""
    vad = _vad_for_adaptive(limit_seconds=0.8)
    vad.max_speech_samples = 512 * 40
    import numpy as np

    chunk = np.zeros(512, dtype=np.float32)
    vad._scripted = [0.9] * 30
    for _ in range(30):
        vad.process_chunk(chunk)
    _audio, _duration, epoch = vad.peek_buffer()

    # Drive it past max duration so _split_at_best_pause runs.
    vad._scripted = [0.9] * 6 + [0.05] * 6 + [0.9] * 12
    for _ in range(24):
        vad.process_chunk(chunk)

    assert vad.trim_front(512 * 5, epoch) is False


# --- flush_final: a closing sentence is not a mid-stream short segment -----


def test_flush_final_keeps_a_closing_sentence_shorter_than_min_speech():
    """[B2] pause(), session end and stop all used flush(), which drops
    anything under min_speech_duration so it can merge with the next speech
    onset. At those three moments there is no next onset. With the shipped
    min_speech_duration of 2.0s every closing sentence under two seconds was
    eaten -- uncounted and unlogged."""
    vad = _vad_for_flush([0.9] * 45)  # 1.44s, all voiced
    vad.min_speech_samples = int(2.0 * 16000)

    assert vad.flush() is None, "precondition: flush() drops it mid-stream"

    vad = _vad_for_flush([0.9] * 45)
    vad.min_speech_samples = int(2.0 * 16000)

    segment = vad.flush_final()

    assert segment is not None
    assert len(segment) == 512 * 45
    assert vad.discarded_segments == 0


def test_flush_final_still_discards_room_tone():
    """Unlike force_flush, flush_final keeps the noise check: a buffer holding
    nothing but room tone is discarded, and counted so the pipeline resets."""
    vad = _vad_for_flush([0.0] * 40)
    vad.min_speech_samples = int(2.0 * 16000)

    assert vad.flush_final() is None
    assert vad.discarded_segments == 1


# --- Noise check: two judgements, and the pre-roll sentinel is not evidence -


def test_a_short_dense_segment_survives_the_absolute_floor():
    """[D1] The absolute floor alone is *stricter* than the ratio rule it
    replaced for any segment under a second, and min_speech_duration goes down
    to 0.1s in the panel. 0.096s of measured speech in a 0.48s segment is
    under the floor but 40% dense, so the ratio rule -- kept as a second
    opinion -- rescues it. A discard needs both to agree."""
    vad = _vad_for_flush([0.5] * 3 + [0.9] * 3 + [0.0] * 9)

    segment = vad._flush_segment()

    assert segment is not None
    assert vad.discarded_segments == 0


def test_the_pre_speech_sentinel_is_not_counted_as_voiced():
    """[D2] The pre-speech ring buffer is stored *at* the threshold as a
    sentinel for "not measured" -- its real confidence was below it. Counting
    it handed every fresh segment 0.096s of free credit toward the floor, and
    handed a trimmed remainder (whose pre-roll has been popped off the front)
    none at all: the same audio judged by two different rules depending on
    where in the utterance it sat."""
    vad = _vad_for_flush([0.5] * 3 + [0.9] * 2 + [0.0] * 95)

    # Measured: 2 chunks = 0.064s, under the floor. Density with the sentinel
    # is 5/100 = 5%, under the ratio too -- so this is noise on both counts.
    assert vad._flush_segment() is None
    assert vad.discarded_segments == 1

    # The same shape with the sentinel *counted* would be 5 chunks = 0.16s,
    # over the 0.15s floor, and would have survived. That asymmetry is the bug.
    assert vad._MIN_VOICED_SECONDS < 5 * 0.032


def test_the_field_segment_that_started_all_this_still_survives():
    """35 voiced chunks in a 178-chunk segment: 1.12s of real speech at 20%
    density, thrown away wholesale by the old ratio rule. The absolute floor
    is what rescues it, and it must keep doing so now that the ratio rule is
    back as a second opinion."""
    vad = _vad_for_flush([0.9] * 35 + [0.0] * 143)

    assert vad._flush_segment() is not None
    assert vad.discarded_segments == 0
