import logging
import collections

import numpy as np
import torch

torch.set_num_threads(1)

log = logging.getLogger("LiveTranslate.VAD")


class VADProcessor:
    """Voice activity detection over 32 ms / 512-sample chunks.

    NOT THREAD-SAFE. Every attribute and method — public and private, reads
    included — must be accessed while holding ``LiveTranslateApp._vad_lock``.
    The capture thread calls ``process_chunk`` while the ASR thread calls
    ``peek_buffer``/``trim_front`` and the Qt thread calls ``update_settings``/
    ``flush``/``_reset``; ``_speech_buffer`` and ``_confidence_history`` are
    parallel sequences whose lengths must stay equal, and an unlocked writer
    can break that invariant.

    Narrow, deliberate exceptions — all of them single-attribute reads whose
    result never feeds a segmentation decision, so a torn read costs at most
    one skipped or extra poll:

    * ``last_confidence`` — monitor UI display value.
    * ``discarded_segments`` — monitor UI display value (the capture loop's own
      edge detection on it *is* taken under the lock).
    * ``_speech_buffer`` length — memory diagnostics only.
    * ``_is_speaking`` / ``_speech_samples`` in the capture loop — gate whether
      to *consider* an interim ASR pass; the pass itself re-reads under lock.

    Anything that decides where to cut a segment, or that mutates state, takes
    the lock. When in doubt, take the lock.
    """

    """Voice Activity Detection with multiple modes."""

    # Least *measured* voiced audio a segment must hold to be worth
    # recognizing: roughly one short word ("Da"). See the noise check in
    # _flush_segment for why this is an absolute duration and not a share of
    # the segment.
    #
    # 0.15, not the 0.25 this started as, because the count it is compared
    # against no longer includes the pre-speech ring buffer. Those 3 chunks
    # are stored *at* the threshold as a sentinel (see the onset branch of
    # process_chunk) precisely because their real confidence was below it, so
    # counting them handed every fresh segment 0.096s of free credit -- and
    # handed a trimmed or split remainder, whose pre-roll has been popped off
    # the front, none at all. 0.25 - 0.096 = 0.154 is what the old constant
    # actually demanded of a fresh segment, so 0.15 keeps that calibration
    # while making a remainder judged by the same rule.
    _MIN_VOICED_SECONDS = 0.15

    # Shortest silence that counts as a pause sample for the adaptive limit.
    # Continuous speech is full of intra-word dips of 0.1-0.2s, and they
    # outnumber real sentence pauses by roughly 2:1 -- sampling them made P75
    # describe the gaps *between words*, not between sentences, which is the
    # second half of why the limit sat too low. Measured against a known pause
    # distribution: a floor here converges the estimate onto the injected P75,
    # and without it the estimate stays 1.4s low even with censoring fixed.
    _MIN_PAUSE_SAMPLE_SECONDS = 0.25

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.50,
        min_speech_duration=1.0,
        max_speech_duration=15.0,
        chunk_duration=0.032,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.energy_threshold = 0.02
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self._chunk_duration = chunk_duration
        self.mode = "silero"  # "silero", "energy", "disabled"
        # Monotonic counter of segments dropped by the noise filter. The
        # pipeline watches it to reset interim state on a path that emits
        # nothing (see LiveTranslateApp._capture_loop), and the monitor bar
        # surfaces it so a discard is visible rather than silent.
        self.discarded_segments = 0

        # Silero v5 ships its model inside the `silero-vad` PyPI package, so load
        # it from there (zero network). Only fall back to the torch.hub cache
        # (pinned branch -> offline when already cached) if the package is
        # missing, since torch.hub otherwise probes/downloads from GitHub.
        try:
            from silero_vad import load_silero_vad
        except ImportError:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad:master",
                model="silero_vad",
                trust_repo=True,
            )
        else:
            self._model = load_silero_vad()
        self._model.eval()

        self._speech_buffer = []
        self._confidence_history = []  # per-chunk confidence, synced with _speech_buffer
        self._speech_samples = 0
        # Bumped whenever the buffer stops being the same utterance (_reset,
        # _split_at_best_pause). peek_buffer hands it out and trim_front
        # demands it back: the interim ASR pass runs for seconds *without the
        # lock* between those two calls, and without this a flush that landed
        # in between made trim_front eat the head of the next sentence.
        self._buffer_epoch = 0
        self._is_speaking = False
        self._silence_counter = 0
        self._was_trimmed = False  # True after trim_front (interim ASR active)
        # The threshold this utterance was accumulated under. The density and
        # split verdicts must use it rather than the live self.threshold: the
        # settings slider can move mid-utterance, and judging a confidence
        # history against a threshold it was never accumulated under threw away
        # 5.7s of real speech in the field (chunks accepted at 0.0, judged at
        # 0.11). Snapshotted at speech onset, restored by _reset.
        self._segment_threshold = threshold

        # Pre-speech ring buffer: capture onset consonants before VAD triggers
        self._pre_speech_chunks = 3  # ~96ms at 32ms/chunk
        self._pre_buffer = collections.deque(maxlen=self._pre_speech_chunks)

        # Silence timing
        self._silence_mode = "auto"  # "auto" or "fixed"
        self._fixed_silence_dur = 0.8
        self._silence_limit = self._seconds_to_chunks(0.8)

        # Progressive silence: shorter threshold when buffer is long
        # (min_buffer_seconds, multiplier applied at or above it) -- the last
        # tier whose bound the buffer reaches wins, so the ranges are
        # <6s = full, 6-10s = half, >=10s = quarter. The comments used to
        # claim <3s/3-6s/6-10s, which is one tier off from what the loop in
        # _get_effective_silence_limit actually does; the behaviour is the
        # measured one and the censored-sample guard below is tuned to it, so
        # it is the wording that was wrong.
        self._progressive_tiers = [
            (3.0, 1.0),
            (6.0, 0.5),
            (10.0, 0.25),
        ]

        # Adaptive silence tracking: recent pause durations (seconds). Holds
        # both kinds of observation — pauses that ended when speech resumed
        # (exact) and pauses that reached the limit and ended the accumulation
        # (right-censored, recorded at the limit). Sampling only the first kind
        # is what used to make the estimate collapse; see
        # _update_adaptive_limit and the recording site in process_chunk.
        self._pause_history = collections.deque(maxlen=100)
        self._adaptive_min = 0.3
        self._adaptive_max = 2.0

        # Exposed for monitor
        self.last_confidence = 0.0

    def _seconds_to_chunks(self, seconds: float) -> int:
        return max(1, round(seconds / self._chunk_duration))

    def _update_adaptive_limit(self):
        """Set the silence limit to P75 of recent pauses x 1.2.

        Correctness rests entirely on _pause_history containing *every* pause,
        including the ones that reached the limit and ended the accumulation.
        Those are right-censored — recorded at the limit, though the speaker's
        real pause was longer. Sampling only pauses that ended before the limit
        (the original code) truncated the history at the very value it was used
        to compute, so the sample P75 sat below the limit by construction and
        every update ratcheted the limit down. Measured in the field it fell
        0.80 -> 0.50 -> 0.46 -> 0.42 -> 0.38 -> 0.35 -> 0.31 -> 0.30s and never
        recovered, until a 0.3s breath cut every sentence and a single lecture
        definition arrived as six untranslatable fragments.

        Censoring is self-correcting under the 1.2 multiplier rather than a
        downward bias: once >=25% of the sample sits at the limit, the sample
        P75 *is* the limit and the target grows by 1.2 each update, so the
        fixed point is 1.2 x P75(pauses >= _MIN_PAUSE_SAMPLE_SECONDS) and it is
        approached from below. The pressure that actually matters comes from the
        other end -- _MIN_PAUSE_SAMPLE_SECONDS drops every intra-word dip, which
        raises the surviving P75 and pushes the limit *up*; a full closed-loop
        simulation with no other guard runs it to _adaptive_max. What holds it
        at the measured 1.2-1.5s is the progressive-tier exemption at the
        recording site in process_chunk. Treat that exemption as load-bearing.
        """
        if len(self._pause_history) < 3:
            return
        pauses = sorted(self._pause_history)
        idx = int(len(pauses) * 0.75)
        p75 = pauses[min(idx, len(pauses) - 1)]
        target = max(self._adaptive_min, min(self._adaptive_max, p75 * 1.2))
        new_limit = self._seconds_to_chunks(target)
        if new_limit != self._silence_limit:
            log.debug(
                f"Adaptive silence: {target:.2f}s ({new_limit} chunks), P75={p75:.2f}s"
            )
            self._silence_limit = new_limit

    def update_settings(self, settings: dict):
        if "vad_mode" in settings:
            self.mode = settings["vad_mode"]
        if "vad_threshold" in settings:
            self.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self.energy_threshold = settings["energy_threshold"]
        if "min_speech_duration" in settings:
            self.min_speech_samples = int(
                settings["min_speech_duration"] * self.sample_rate
            )
        if "max_speech_duration" in settings:
            self.max_speech_samples = int(
                settings["max_speech_duration"] * self.sample_rate
            )
        if "silence_mode" in settings:
            self._silence_mode = settings["silence_mode"]
        if "silence_duration" in settings:
            self._fixed_silence_dur = settings["silence_duration"]
        # Driven by the mode now in effect, not by silence_duration happening to
        # be in the dict: switching auto -> fixed left the adaptive limit in
        # place unless that unrelated key came along in the same update. In
        # "auto" the limit belongs to _update_adaptive_limit and is left alone.
        if self._silence_mode == "fixed":
            self._silence_limit = self._seconds_to_chunks(self._fixed_silence_dur)
        log.info(
            f"VAD settings updated: mode={self.mode}, threshold={self.threshold}, "
            f"silence={self._silence_mode} "
            f"({self._silence_limit} chunks = {self._silence_limit * self._chunk_duration:.2f}s)"
        )

    def _silero_confidence(self, audio_chunk: np.ndarray) -> float:
        window_size = 512 if self.sample_rate == 16000 else 256
        chunk = np.asarray(audio_chunk[:window_size], dtype=np.float32)
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        with torch.inference_mode():
            tensor = torch.from_numpy(chunk).float()
            return float(self._model(tensor, self.sample_rate).item())

    def _energy_confidence(self, audio_chunk: np.ndarray) -> float:
        samples = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.dot(samples, samples) / max(samples.size, 1)))
        return min(1.0, rms / (self.energy_threshold * 2))

    def _get_confidence(self, audio_chunk: np.ndarray) -> float:
        if self.mode == "silero":
            return self._silero_confidence(audio_chunk)
        elif self.mode == "energy":
            return self._energy_confidence(audio_chunk)
        else:  # disabled
            return 1.0

    def _get_effective_silence_limit(self) -> int:
        """Progressive silence: accept shorter pauses as split points when buffer is long."""
        buf_seconds = self._speech_samples / self.sample_rate
        multiplier = 1.0
        for tier_sec, tier_mult in self._progressive_tiers:
            if buf_seconds < tier_sec:
                break
            multiplier = tier_mult
        effective = max(1, round(self._silence_limit * multiplier))
        return effective

    def process_chunk(self, audio_chunk: np.ndarray):
        confidence = self._get_confidence(audio_chunk)
        self.last_confidence = confidence

        effective_threshold = self.threshold if self.mode == "silero" else 0.5
        eff_silence_limit = self._get_effective_silence_limit()

        if confidence >= effective_threshold:
            # Record pause duration for adaptive mode
            if self._is_speaking and self._silence_counter > 0:
                pause_dur = self._silence_counter * self._chunk_duration
                if pause_dur >= self._MIN_PAUSE_SAMPLE_SECONDS:
                    self._pause_history.append(pause_dur)
                    if self._silence_mode == "auto":
                        self._update_adaptive_limit()

            if not self._is_speaking:
                # Speech onset: this utterance is judged against the threshold
                # in effect right now, not whatever the slider reads by the time
                # it is flushed.
                self._segment_threshold = effective_threshold
                # Prepend pre-speech buffer to capture leading consonants.
                # Use threshold as confidence so these chunks don't create false valleys
                for pre_chunk in self._pre_buffer:
                    self._speech_buffer.append(pre_chunk)
                    self._confidence_history.append(effective_threshold)
                    self._speech_samples += len(pre_chunk)
                self._pre_buffer.clear()

            self._is_speaking = True
            self._silence_counter = 0
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)
        elif self._is_speaking:
            self._silence_counter += 1
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)
        else:
            # Not speaking: feed pre-speech ring buffer
            self._pre_buffer.append(audio_chunk)

        # Force segment if max duration reached — backtrack to find best split point
        if self._speech_samples >= self.max_speech_samples:
            return self._split_at_best_pause()

        # End segment after enough silence (progressive threshold)
        if self._is_speaking and self._silence_counter >= eff_silence_limit:
            # This pause reached the limit, so its true length was never seen:
            # a right-censored observation, recorded at the limit. It belongs in
            # the history no matter which of the three exits below runs, because
            # what the estimator measures is the speaker's pause rhythm, not
            # whether we happened to emit a segment.
            #
            # Skipped when a progressive tier shortened the limit: that cut is
            # our own impatience with a long buffer, not the speaker's rhythm,
            # and feeding it back would drag the baseline down. Verified not to
            # starve the sample — a full-loop simulation still collected 800
            # censored observations and converged to 0.99s (ideal 1.10s).
            # Recorded in *both* silence modes, exactly like the exact-pause
            # branch above -- only the limit update is gated on "auto". The
            # two used to disagree: fixed mode kept appending exact pauses
            # while refusing censored ones, so a spell in fixed mode left
            # _pause_history holding nothing but sub-limit samples, and the
            # first update after the user flipped the combo back to auto
            # collapsed the limit in a single step (measured: 1.00s -> 0.32s).
            # That is the original bug, reachable from a dropdown.
            censored_pause = self._silence_counter * self._chunk_duration
            if (
                eff_silence_limit == self._silence_limit
                and censored_pause >= self._MIN_PAUSE_SAMPLE_SECONDS
            ):
                self._pause_history.append(censored_pause)
                if self._silence_mode == "auto":
                    self._update_adaptive_limit()
            if self._speech_samples >= self.min_speech_samples:
                return self._flush_segment()
            elif self._was_trimmed:
                # Interim ASR trimmed the buffer; return remainder instead of dropping
                log.debug(
                    f"Short segment after trim ({self._speech_samples / self.sample_rate:.1f}s), "
                    f"force flushing for interim final"
                )
                return self.force_flush()
            else:
                # Too short — keep buffer, merge with next speech onset
                log.debug(
                    f"Short segment {self._speech_samples / self.sample_rate:.1f}s "
                    f"< min {self.min_speech_samples / self.sample_rate:.1f}s, "
                    f"keeping for merge"
                )
                self._is_speaking = False
                self._silence_counter = 0
                return None

        return None

    def _find_best_split_index(self) -> int:
        """Find the best chunk index to split at using smoothed confidence.
        A sliding window average reduces single-chunk noise, then we find
        the center of the lowest valley. Works even when the speaker never
        fully pauses (e.g. fast commentary).
        Returns -1 if no usable split point found."""
        n = len(self._confidence_history)
        if n < 4:
            return -1

        # Smooth confidence with a sliding window (~160ms = 5 chunks at 32ms)
        smooth_win = min(5, n // 2)
        smoothed = []
        for i in range(n):
            lo = max(0, i - smooth_win // 2)
            hi = min(n, i + smooth_win // 2 + 1)
            smoothed.append(sum(self._confidence_history[lo:hi]) / (hi - lo))

        # Search in the latter 70% of the buffer (avoid splitting too early)
        search_start = max(1, n * 3 // 10)

        # Find global minimum in smoothed curve
        min_val = float("inf")
        min_idx = -1
        for i in range(search_start, n):
            if smoothed[i] <= min_val:
                min_val = smoothed[i]
                min_idx = i

        if min_idx <= 0:
            return -1

        # Check if this is a meaningful dip
        avg_conf = sum(smoothed[search_start:]) / max(1, n - search_start)
        dip_ratio = min_val / max(avg_conf, 1e-6)

        effective_threshold = (
            self._segment_threshold if self.mode == "silero" else 0.5
        )
        if min_val < effective_threshold or dip_ratio < 0.8:
            log.debug(
                f"Split point at chunk {min_idx}/{n}: "
                f"smoothed={min_val:.3f}, avg={avg_conf:.3f}, dip_ratio={dip_ratio:.2f}"
            )
            return min_idx

        # Fallback: any point below average is better than hard cut
        if min_val < avg_conf:
            log.debug(
                f"Split point (fallback) at chunk {min_idx}/{n}: "
                f"smoothed={min_val:.3f}, avg={avg_conf:.3f}"
            )
            return min_idx

        return -1

    def _split_at_best_pause(self):
        """When hitting max duration, backtrack to find the best pause point.
        Flushes the first part and keeps the remainder for continued accumulation."""
        if not self._speech_buffer:
            return None

        split_idx = self._find_best_split_index()

        if split_idx <= 0:
            # No good split point — hard flush everything
            log.info(
                f"Max duration reached, no good split point, "
                f"hard flush {self._speech_samples / self.sample_rate:.1f}s"
            )
            return self._flush_segment()

        # Split: emit first part, keep remainder
        first_bufs = self._speech_buffer[:split_idx]
        remain_bufs = self._speech_buffer[split_idx:]
        remain_confs = self._confidence_history[split_idx:]

        first_samples = sum(len(b) for b in first_bufs)
        remain_samples = sum(len(b) for b in remain_bufs)

        log.info(
            f"Max duration split at {first_samples / self.sample_rate:.1f}s, "
            f"keeping {remain_samples / self.sample_rate:.1f}s remainder"
        )

        segment = np.concatenate(first_bufs)

        # Keep remainder in buffer for next segment
        self._speech_buffer = remain_bufs
        self._confidence_history = remain_confs
        self._speech_samples = remain_samples
        # The buffer is no longer the audio any in-flight interim pass peeked.
        self._buffer_epoch += 1
        self._is_speaking = True
        self._silence_counter = 0

        return segment

    def _flush_segment(self):
        if not self._speech_buffer:
            return None
        # Noise check: discard only when the segment holds almost no speech at
        # all. Deliberately an *absolute* voiced duration, not a ratio: a ratio
        # punishes exactly the speaker this app exists for. A lecturer who says
        # one sentence and then pauses to write on the board produces a segment
        # that is mostly silence -- 18-24% density, measured -- and the old
        # `density < 0.25` test threw the whole thing away, so nothing reached
        # ASR at all. Lowering the VAD threshold could not rescue it either
        # (0.5 -> 0.05 moved density only 18% -> 24%). In the field this
        # discarded 22 segments, one of them holding 1.12s of real speech.
        if len(self._confidence_history) >= 4:
            effective_threshold = (
                self._segment_threshold if self.mode == "silero" else 0.5
            )
            n_chunks = len(self._confidence_history)
            # Strictly above: the pre-speech ring buffer is stored *at* the
            # threshold as a sentinel for "not measured" (its real confidence
            # was below it), so it is not evidence of speech and must not pay
            # into the floor. Real confidences landing exactly on the
            # threshold are a float coincidence worth no special handling.
            measured_voiced = sum(
                1 for c in self._confidence_history if c > effective_threshold
            )
            voiced_seconds = measured_voiced * self._chunk_duration
            # The old ratio rule, kept verbatim (sentinel included, `>=`) as a
            # second opinion. A segment is discarded only when *both* agree it
            # is noise, which makes "never stricter than what it replaced" a
            # property of the code rather than a hope: the absolute floor
            # rescues the long sparse segment a ratio punished (a lecturer who
            # says one sentence then pauses to write on the board), and the
            # ratio rescues the short dense one an absolute floor would punish
            # (min_speech_duration goes down to 0.1s in the panel).
            legacy_voiced = sum(
                1 for c in self._confidence_history if c >= effective_threshold
            )
            density = legacy_voiced / n_chunks
            if voiced_seconds < self._MIN_VOICED_SECONDS and density < 0.25:
                dur = self._speech_samples / self.sample_rate
                log.info(
                    f"Discarding {dur:.1f}s segment: only {voiced_seconds:.2f}s voiced "
                    f"({measured_voiced}/{n_chunks} chunks, density {density:.0%}), "
                    f"below the {self._MIN_VOICED_SECONDS:.2f}s floor"
                )
                self._reset()
                # A discarded segment produces no vad_flush event, so the
                # pipeline never learned the utterance had ended and carried its
                # interim state (pending fragments, echo tail) into the next one.
                self.discarded_segments += 1
                return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def _reset(self):
        # Rebind the two parallel sequences in one statement so no concurrent
        # reader can observe a buffer/history length mismatch between them.
        # Every caller must hold LiveTranslateApp._vad_lock (see class docstring),
        # but keeping the invariant atomic here makes the class safe to reason
        # about on its own.
        self._speech_buffer, self._confidence_history = [], []
        self._speech_samples = 0
        # Whatever an in-flight interim pass peeked is now gone; its trim
        # must not land on the next utterance.
        self._buffer_epoch += 1
        self._is_speaking = False
        self._silence_counter = 0
        self._was_trimmed = False
        # No utterance is in flight, so the next flush (before any onset can
        # snapshot) should judge against the live threshold.
        self._segment_threshold = self.threshold

    def peek_buffer(self):
        """Read the buffer without flushing. Returns (audio, duration, epoch).

        ``epoch`` identifies the utterance the audio came from. Hand it back
        to ``trim_front`` after the (unlocked, seconds-long) ASR pass: if the
        buffer turned over in the meantime the trim is refused instead of
        being applied to somebody else's audio.
        """
        if not self._speech_buffer or not self._is_speaking:
            return None
        audio = np.concatenate(self._speech_buffer)
        duration = self._speech_samples / self.sample_rate
        return audio, duration, self._buffer_epoch

    def trim_front(self, n_samples: int, epoch=None) -> bool:
        """Remove the first n_samples. Returns whether the trim was applied.

        ``epoch`` is the value peek_buffer returned alongside the audio the
        caller measured. The interim ASR pass drops the lock for the whole of
        recognition, and in that window the capture thread can end the
        utterance (silence limit), split it (max duration) or discard it as
        noise -- after any of those, ``n_samples`` describes audio this buffer
        no longer holds. Applying it anyway deleted the head of the *next*
        sentence (measured: a 1.28s utterance trimmed to nothing) and left
        ``_was_trimmed`` set on a segment that was never trimmed.
        """
        if epoch is not None and epoch != self._buffer_epoch:
            log.debug(
                f"trim_front refused: buffer moved on (epoch {epoch} -> "
                f"{self._buffer_epoch}); {n_samples} samples belong to a "
                f"finished utterance"
            )
            return False
        if n_samples <= 0:
            return False
        removed = 0
        while self._speech_buffer and removed < n_samples:
            chunk = self._speech_buffer[0]
            if removed + len(chunk) <= n_samples:
                self._speech_buffer.pop(0)
                self._confidence_history.pop(0)
                removed += len(chunk)
            else:
                # Partial trim of first chunk
                keep = removed + len(chunk) - n_samples
                self._speech_buffer[0] = chunk[-keep:]
                removed = n_samples
        self._speech_samples = sum(len(b) for b in self._speech_buffer)
        self._was_trimmed = True
        log.debug(f"trim_front: removed {removed} samples, remaining {self._speech_samples / self.sample_rate:.2f}s")
        return True

    def force_flush(self):
        """Flush buffer regardless of min_speech_samples."""
        if not self._speech_buffer:
            return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def flush(self):
        if self._speech_samples >= self.min_speech_samples:
            return self._flush_segment()
        self._reset()
        return None

    def flush_final(self):
        """Flush for a pause, a session end or app exit: no min-speech gate.

        ``flush()`` drops anything shorter than ``min_speech_duration`` on the
        floor, because in mid-stream that buffer is kept to merge with the
        next speech onset. At a pause, a session end or a stop there is no
        next onset -- the merge never happens and the audio is simply gone.
        With the shipped ``min_speech_duration`` of 2.0s that silently ate
        every closing sentence shorter than two seconds, uncounted and
        unlogged.

        The noise check still applies (unlike ``force_flush``), so a buffer
        holding nothing but room tone is still discarded and still counted.
        """
        return self._flush_segment()
