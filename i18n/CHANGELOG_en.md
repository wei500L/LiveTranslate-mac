# Changelog

## 2026-09-01
- New on macOS: `build_mac_app.sh` builds a double-clickable `LiveTranslate.app` (`--install` also copies it into /Applications), so the app starts from Launchpad or Spotlight instead of a terminal. The bundle executable is a tiny compiled Mach-O launcher — LaunchServices rejects shell-script executables with error -10669 — that starts the venv python with the right working directory, and the icon is generated from the same design the in-app icon uses. The launcher is byte-identical across rebuilds, so the microphone permission granted to the app survives a rebuild; the first launch asks for Microphone permission again because TCC now attributes it to the app rather than the terminal it was launched from
- Fixed the overlay's Subtitle button and paused state having no visual feedback at all: their highlighted stylesheets were built by string replacement, but the strings being replaced had stopped existing when the button CSS was reworked, so the replacement silently did nothing — paused looked identical to running, and the Subtitle button (whose text never changes) gave no hint whether the subtitle window was on. Paused now shows an amber wash, subtitle-on a green one
- Fixed a restarted session inheriting the previous one's statistics: the transcript writer is created once per app run — stopping closes the session, starting reuses the same instance to open a new one — but the entry counts, speech time and engine info were not reset with it, so the second meeting's sidecar and closing summary added the first meeting's numbers on top, and the meeting-records page showed inflated entry counts. Each session now starts from zero
- Fixed every export after a Clear claiming it was truncated: the overlay keeps only the last 50 messages and counts what rotated out, and the export uses that count to warn "the full record is in the meeting files". Clearing the message list did not reset the counter, so from one Clear onward every export popped a misleading truncation warning pointing at stale transcript files of a session that no longer mattered. The counter now resets on Clear
- Fixed permanent pipeline silence when macOS system audio keeps failing to decode: the ScreenCaptureKit capture thread treated every decode exception as a transient glitch with a warning, but once the stream delivers a format this decoder cannot read (permanent by nature) every buffer fails — the app keeps running, the UI keeps saying "Running", and no audio ever arrives again. The microphone and Windows capture paths already escalate after a run of consecutive failures; SCK now does the same after 20, reporting a terminal error per the capture contract so the pipeline stops and surfaces it
- Fixed the log viewer swallowing angle brackets: log lines were concatenated into HTML before being appended, so Qt ate text like `<think>…</think>` as a tag — exactly the lines someone opens the window to read when diagnosing a thinking model. Lines are now HTML-escaped and rendered verbatim
- Fixed the benchmark Test button getting permanently stuck: a model config missing its name (or a crash while aggregating results) let the exception escape through the thread pool and kill the benchmark thread before it could send the trailing `__DONE__` line — which is exactly what re-enables the button, so only an app restart recovered it. The run now always finishes, and a single model's error is recorded as that model's failure without affecting the others
- Fixed the add-model dialog silently discarding an incomplete entry: with the display name or model ID left blank the dialog closed fine, but the model-list handler dropped the entry on the floor — the user filled in the form, pressed OK, and nothing happened. Required fields are now checked with a warning and the dialog stays open
- Fixed the target language reverting when any panel setting was changed after picking it in the overlay: the language was written into the *copy* that `get_settings()` returns, so the file was correct but the panel still held the old value and wrote it straight back on its next auto-save. The panel now exposes `update_setting()` as the one entry point for external writes
- Fixed the subtitle line list showing rows no operation could act on when the settings had no `lines` key: the display fell back to the default lines while add/edit/remove/reorder all fell back to an empty list. The missing upper bound on "move up" was fixed with it
- Fixed the incremental echo dedup never firing at all: committed text is always a complete sentence and therefore always ends in punctuation, which was included in the comparison, and a new recognition never begins with a full stop. Ignoring trailing punctuation lets it catch a real replay, while a single repeated word is now deliberately kept — a lecturer opening the next sentence with the previous one's keyword is common, and deleting that word costs the sentence its subject, whereas a duplicated word is merely wordy
- Fixed the VAD silence-mode switch depending on an unrelated setting: going from adaptive to fixed only recomputed the silence threshold when `silence_duration` happened to arrive in the same settings update, otherwise the adaptive value stayed in effect
- Fixed pausing and resuming splicing two utterances into one line: the VAD buffer was left untouched across a pause, so audio from after the resume was appended to the half-sentence from before it — across a gap of any length — producing one incoherent subtitle. Pausing now hands the in-flight utterance to the recognizer as a normal line and clears the buffer, so the resume starts clean
- Fixed every settings change re-rendering all subtitles: the control panel emits the whole settings dict on each auto-save, so the style was re-applied unconditionally — about 59ms with a full 50-message buffer, on the UI thread, triggered by moving a VAD slider or ticking a checkbox. An unchanged style is now a no-op
- Fixed short replies being silently dropped in incremental mode: a buffered fragment ("да", "okay") was folded into the current segment's text *before* the noise filter ran, so if that segment happened to be noise and its tail exceeded 2s, the whole thing was discarded — the user spoke and nothing ever appeared. The filter now judges only the segment's own recognition, and buffered content that already passed it is emitted regardless
- Fixed recognition stopping permanently after the ASR worker exits unexpectedly: a killed or crashed worker made the client raise the base exception, while the pipeline's recovery branch only catches the worker-death type, so every later segment hit the same error and no restart ever fired. Measured: killing the worker yielded 3 of 12 segments and then silence; it now restarts automatically and yields 11 (losing only the segment in flight at the moment of the kill)
- Fixed the meeting record freezing when the translation model is switched mid-session: the in-flight translation is discarded for history purposes, but neither the transcript entry nor the overlay entry was closed out. Because the record releases entries in order, that one entry blocked every later one (measured: 0 of 3 written), and its overlay message stayed on "translating" forever

## 2026-08-31
- Fixed the benchmark and live translation still diverging on "disable thinking": `Translator`'s own `no_think` default was False while every caller passed True, so anything constructing a Translator directly (the benchmark included) silently got "send no parameter" for a config without an explicit `thinking_style`, where the runtime got "auto" — the very parameter the benchmark exists to diagnose. Both paths now go through one constructor, `translator_from_model_config()`, instead of each keeping its own argument list
- Documented the settings that are easy to get wrong: `config.yaml` now explains the empty api_key, the context-turn semantics, max_tokens and the sampling temperature; the model dialog's "context turns" hint covers the two traps (a `{context}` placeholder embeds history instead of sending turns, and translation-specialized models react differently to the two forms); and "Model Connect Timeout" now explains that the first request of a session is much slower
- Added a Troubleshooting section to the README covering `debug_pipeline.py`
- The local HY-MT model now uses multi-turn context and greedy decoding: `context_turns` was **silently ignored** for `no_system_role` providers (setting it to 3 produced no context and no warning), and history is now sent as real conversation turns — measured, it resolves pronouns the isolated sentence cannot ("now find it" -> "now find the derivative of the function"), stays 8/8 stable across a lecture, and costs only ~60 extra prompt tokens. This holds only in the multi-turn form: pasting the same history into the prompt text still makes the model continue it (0/3)
- The local model now decodes greedily (temperature 0.7 -> 0.0): translating the same sentence repeatedly used to agree only 2 times out of 4, and now agrees 4/4, at equal quality and lower latency (0.39s vs 0.49s average) — a subtitle should not keep changing for the same sentence. `top_k` was dropped with it (meaningless under greedy) and `repetition_penalty` stays as the runaway guard
- Fixed the local HY-MT model emitting the prompt instead of a translation: it is a translation-specialized model, not an instruction-following one, so it continues a multi-line instruction block rather than obeying it — the long prompt failed 3/3 when measured (regurgitated the prompt, burned the whole 128-token budget, ~2.5s per sentence) while a one-line directive succeeded 4/4 (~0.3s). `context_turns` is now 0 for the same reason: with `no_system_role` the only channel for context is the `{context}` placeholder, and a context block re-triggers the runaway (0/3). Configs still carrying a prompt this project shipped are migrated; an edited prompt is left alone
- Fixed a ~5 second wait for the local model service on every exit: the app and the settings panel each held their own service manager, so the one doing the stopping had never spawned the process and could not reap it — it polled a zombie until the grace period expired and then wrongly reported "could not be killed". They now share one manager, and quitting went from 7s to 1s
- Fixed the translator being rebuilt twice when the local service starts (which also cleared the context history twice)
- Fixed an empty API key making the app impossible to start at all: `config.yaml` ships an empty key, and recent openai SDKs reject one when the client is constructed, so any fresh install without a configured key crashed before a window appeared; an empty key is now treated as "a local server that does not check it" (LM Studio, Ollama and the managed MLX service all work this way)
- Fixed meeting-record entries being out of order: translations complete concurrently across 8 workers, and each one was written the moment it landed, so the all/translation records were in completion order rather than the order things were said; entries are now released in utterance order, and anything still awaiting a translation at exit is written out instead of being lost
- Added a Meeting Records settings page: lists past sessions (time, entries, duration, ASR engine, translation model) and lets you preview, export and delete individual meetings
- Richer meeting records: each entry carries its language and duration, a Markdown record is written alongside the text files, the session ends with a summary (start/end, total duration, speech time, entry counts, engine and model), and a JSON sidecar backs the list view
- Added a diagnostic harness: `debug_pipeline.py` replays an audio file through the real pipeline (VAD → ASR worker → translation → transcript), captures the full DEBUG log and reports what each stage produced; a stage that was asked to do work and produced nothing now fails the run instead of passing quietly
- Fixed an ASR worker that finished loading before the pipeline started being discarded as "superseded by a newer switch"
- Fixed the control panel's background threads (cache scan, MLX probe) possibly being destroyed while still running at exit
- The shutdown log no longer prints "Pipeline stopped" twice

## 2026-08-30
- Fixed a possible hang, or leftover processes, on exit: the stop sequence delivered its stop signal with a blocking queue write, which waited forever when the queue was full and the recognition thread had already gone; stopping is now idempotent and bounded, and a failure in one cleanup step no longer skips reclaiming the threads, transcript files, recognition worker and local service after it
- Fixed exit and engine switching taking up to 120 seconds while the recognition worker was busy transcribing: the client's request lock and lifecycle lock were the same lock, so quitting queued behind the in-flight transcription; shutdown now aborts the worker immediately and the UI responds within a second or two, and switching engines mid-model-load can also be aborted at once
- Fixed the recognition worker reporting "ready" after it was gone: a process that exited on its own, or whose handles were already closed, was treated as reusable, so later requests neither recovered nor reported an error
- Fixed "delete all models and exit" leaving files behind: it used to quit Qt directly, bypassing pipeline cleanup while the recognition worker still held the model directories open; it now stops everything and confirms the worker is gone before deleting, deletes in the background with progress shown, and lists any paths it could not remove
- Fixed Ctrl-C running several seconds of cleanup inside the signal handler; pressing it twice no longer duplicates that cleanup
- Fixed pausing or quitting within 500ms of launch being overridden by the deferred start callback; the tray and overlay no longer claim to be running before the pipeline actually is
- Fixed a roughly 2-second freeze when clicking the model list with the HY-MT model selected: deployment, environment and service status each spawned a subprocess or issued an HTTP request synchronously on the UI thread; all three now run in the background and the UI reads cached values
- Fixed the translation context being cleared every 5 seconds while the HY-MT service was down: every probe reset the translator; it now happens once, on the available-to-unavailable transition, and automatic restarts have backoff and an attempt limit, after which you are told to start it manually
- Fixed cancellation having no effect during quiet stretches of a model download: cancellation was only checked when the child process emitted a new line, so a download that printed nothing could not be interrupted; it now takes effect at any point and terminates the whole process tree, leaving no stray process or pid file
- Fixed the console going silent after quitting mid-download: the replaced sys.stderr was only restored on the success path, so an early exit left it pointing at a destroyed window forever; every exit path now restores it
- Fixed HY-MT model preparation installing dependencies into the app's own runtime environment, and possibly deleting a model directory the running service still had open
- Fixed the first-run wizard writing a VAD threshold that disagreed with config.yaml (0.3 vs 0.5)
- Fixed the subtitle window dropping sentences during a burst: a new final subtitle cancelled the one still waiting to be shown; they are now queued and displayed in arrival order
- Fixed repetition detection depending on the streaming toggle: the non-streaming path skipped it entirely; added detection for output that starts fine and then falls into a loop
- Fixed each sentence costing two full timeouts when the translation service is unreachable: the streaming fallback caught every exception and retried the whole request
- Fixed incremental recognition emitting the same short reply repeatedly: buffering a short sentence did not advance the audio trim or the echo-dedup state, so the next pass recognized the same words again
- Fixed leftover incremental state from a discarded noise segment carrying into the next utterance
- Fixed settings, the overlay and the running translator possibly pointing at different models after a deletion: removing an entry before the active one shifted every later index up
- Fixed the model benchmark issuing a different request than real translation: it now reuses the same request assembly (thinking mode, overrides, extra_body, JSON mode, role split), and benchmarking a thinking model or an unreachable endpoint no longer crashes or takes twice as long
- Fixed quadratic line-wrapping cost on the UI thread for long subtitles, and a leading separator when one target language had an empty translation
- Fixed repeatedly switching to the cache page spawning several concurrent model-directory scans
- Fixed exported conversation logs being silently incomplete past 50 messages: the export now says it holds only the most recent N and points at the full transcript file
- Fixed Windows audio capture going silent on queue congestion or a failed device restart: it now drops the oldest block instead of killing the capture thread, and retries a failed restart before stopping explicitly
- Fixed a terminal capture failure being reported as "no data right now" in the macOS microphone-only mode; all three capture backends now express "no data" and "terminal failure" the same way
- The remote ASR server now listens on 127.0.0.1 by default (0.0.0.0 exposed an unauthenticated GPU inference endpoint to the network), with an optional shared secret and a request body size limit; `uvicorn asr_server:app` no longer fails to start for lack of configuration
- Fun-ASR-Nano audio loading failures now keep the original error instead of being masked by a follow-on UnboundLocalError
- All HY-MT service progress and error messages, and the language-change notice, now follow the UI language
- Removed the unused pyannote-audio / torchcodec macOS dependencies (only the long-form transcription path needs them, which this project does not take); added a two-way dependency symmetry check so a package can no longer go missing on one platform

## 2026-08-29
- Fixed subtitles being stuck on "translating" forever when the local HY-MT model is selected but its service is not running: that failure shared one exception type with "the app is exiting" and was silently skipped, so the message never settled, never reached the combined transcript, and the internal pending table grew for the rest of the session; it now shows an explicit error and closes out properly
- Fixed recognition stopping permanently when an ASR backend returned a malformed result: a missing or wrongly typed field killed the ASR thread while the capture thread kept filling a queue nobody drained, so no new subtitles appeared and no restart logic fired; results are now validated against one shared contract, and a single bad result is dropped and logged while later audio keeps being recognized
- Fixed an unreachable remote ASR server being treated as silence: server outages, connection timeouts, HTTP 5xx and non-JSON responses now go through the same recovery path as a dead local worker and surface as ASR unavailable, instead of discarding audio quietly; a legitimately empty recognition result is still skipped silently
- Fixed audio dropping out for about 8 seconds on macOS whenever any setting changed: the control panel re-emits the full settings dict on every auto-save and the system audio backend had no device equality check, so it tore down and rebuilt the stream for a device that never changed; it now switches only on a real change, restores the previous device when a switch fails, and stops explicitly when recovery is impossible
- Fixed possible corruption of the speech buffer when settings were changed or the ASR engine was switched while speaking: the UI thread reset VAD state without the lock, which could leave the buffer and confidence sequences at different lengths and skew split-point selection and silence detection
- Fixed the recognition thread exiting outright when the incremental-ASR segmentation library is missing: it now degrades to no splitting and logs an explicit error instead of taking the pipeline down
- `requirements.txt` / `requirements-mac.txt` are now sufficient on their own: added yasbd-lib (previously installed only by the install scripts), socksio on Windows (needed for socks5 proxies), and a transformers <5 upper bound matching what the local MLX service asserts
- Fixed the test and release pipeline: a launcher message regression broke the test suite, incomplete CI dependencies stopped some tests from being collected, and the Windows release job did not depend on the test job (so a red test run still published); both platforms are now gated on tests, with a new assertion preventing that gate from going missing again

## 2026-08-22
- Added macOS 13+ Apple Silicon support documentation and a native arm64 CI job with offline platform/audio tests.
- Added a macOS arm64 source bundle artifact (`LiveTranslate-macos-arm64-*.tar.gz`); signing and notarization remain release follow-ups.
- The audio diagnostic now runs a device-free normalization smoke test by default; Windows WASAPI probing is opt-in with `--live-windows`.
- Documented ScreenCaptureKit and microphone permissions, CPU-only CTranslate2 behavior, MPS expectations, and known unsigned-app TCC limitations.
- Clarified the GigaAM-v3 integration: official Hugging Face `e2e_rnnt` revision, Russian ASR scope, 25-second short-audio limit, and upstream long-form support not yet wired into the app.

## 2026-08-17
- Incremental ASR sentence segmentation switched from PySBD to yasbd-lib (#37): API-compatible with no behavior change, adds native rules for Korean and 16 more languages (Korean previously fell back to English rules), and is much faster on long text
- Fixed empty translations showing up as untranslated same-language text with DeepSeek (#38): DeepSeek defaults to thinking mode ON and only accepts thinking.type=disabled to turn it off, while the previously sent enable_thinking=false is a Qwen-style flag it ignores; the model edit dialog now has a "Disable thinking" style selector (auto-detect / DeepSeek·Volcano Ark·GLM / Qwen·DashScope·SiliconFlow / self-hosted vLLM·SGLang / OpenAI·Grok reasoning_effort / do not send), auto-detect no longer sends unknown parameters to official OpenAI-style endpoints that reject them, and a diagnostic warning is logged when reasoning burns the whole token budget and returns an empty completion
- Fixed settings windows overflowing the screen at 150%+ DPI scaling, which left the OK/Cancel buttons unreachable (#39): the model edit dialog and control panel tabs now scroll when needed, and window heights are clamped to the screen's usable area
- Install and update no longer write download caches to the system drive: the portable build keeps the uv cache and managed Python inside the app folder, and the install/update scripts keep pip's cache and temp files inside the project folder, cleaned up after a successful install

## 2026-07-11
- Fixed Fun-ASR-Nano first load: the Qwen3-0.6B weight download could be killed by the 180s worker startup timeout (#32); weights are now fetched up-front in the model download phase, so worker startup no longer waits on large downloads
- Fixed model detection misses caused by the ModelScope 1.38+ cache location change (#32, #33): all cache layouts across SDK versions are recognized, so upgrading the SDK no longer re-downloads existing models
- Qwen3-0.6B weights can now be downloaded from ModelScope; ModelScope-hub users are no longer forced through HuggingFace (#33)

## 2026-06-20
- New "Remote Whisper" ASR engine: offload speech recognition to a separate GPU machine (ships `asr_server.py` server), so a box without a GPU can still transcribe in real time
- New "WebID / ID Verify" translation prompt preset, tuned for video identity-verification calls
- ASR now runs in an isolated subprocess: the worker auto-restarts on crash/timeout and recycles when memory grows past a threshold, so a recognition failure no longer drags down the UI
- Subtitle window mouse click-through (#28): a toggle in the subtitle settings plus a "Subtitle Click-through" tray shortcut; when on, clicks pass to the window behind (middle-click drag is disabled while on — turn it off to reposition)

## 2026-05-10
- New "Export to file" menu: original / translation / combined formats, accessible from overlay right-click menu and tray menu
- New "Transcript persistence" (enabled by default): each session creates 3 files under `transcripts/` (original / translation / combined), appended in real time per segment — no longer bounded by the 50-message overlay cap
- Settings panel "Cache" tab: added "Transcript persistence" group with toggle and open-folder button
- Memory ceiling protection: tray notification shown once when RSS exceeds 4096MB, advising restart (ASR backends keep native-side workspaces/caches that Python GC and `torch.cuda.empty_cache` may not reclaim)
- New `MEM[asr#/tick]` log lines: per-ASR-call RSS / GPU (alloc/reserved) / audio duration / overlay message count / VAD buffer length for memory diagnostics

## 2026-04-20
- Removed Qwen3-ASR engine (ONNX + GGUF hybrid had compatibility issues; model files and llama.cpp runtime dependencies cleaned up)
- Model config: new "Advanced Parameters" group — `temperature`, `top_p`, `max_tokens`, `frequency_penalty`, `presence_penalty`, `seed`, each gated by an independent "Override" checkbox (unchecked = use server default)
- Model config: new `extra_body` (JSON) field for provider-specific parameters (e.g. `thinking_budget`, `reasoning_effort`), validated on save
- Fix: Anime-Whisper download dialog was a silent no-op when the model was not cached
- Fix: settings panel "Changelog" tab showed blank (regex expected H3 but files used H2 headings — broken since the tab was added in March)

## 2026-04-18
- New ASR engine: Anime-Whisper (litagin/anime-whisper), Japanese-only, specialized for anime / galgame speech (breaths, sighs, non-verbal sounds)
- Fix HF cache detection: aborted downloads leaving empty dirs no longer trigger false "cached" state

## 2026-03-31
- Pipeline thread split: capture+VAD+ASR was a single thread; now capture and ASR run on separate threads, so long-segment ASR no longer blocks live RMS/VAD bar updates
- ASR scheduling uses a bounded queue (16 segments); oldest interim segments are dropped when full to prevent backlog-induced latency

## 2026-03-26
- Default translation prompt improvements: added ASR error-correction rules (fix typos/homophones from context) and fluency rules (avoid word-for-word literal translation)

## 2026-03-25
- Style tab: new "Reset window positions" button — subtitle window returns to (100,100), overlay returns to bottom-right of screen
- Subtitle window default position changed from bottom-center to (100,100); minimum height adjusted to 200px
- Window position restore now validates against the visible screen area (`availableGeometry` excludes the taskbar); height changes clamp to screen bounds to prevent windows from being pushed off-screen

## 2026-03-24
- Subtitle window: auto word-wrap for long text (no more split segments), smooth height animation, pixmap render cache
- Overlay & subtitle window: position/size persistence across restarts
- Overlay: compact mode toggle animation
- Settings: removed valid-key whitelist restriction

## 2026-03-23
- Rebranded LiveTrans → LiveTranslate
- Model config: streaming toggle, structured output, context count, disable thinking (default on)
- Streaming translation display in overlay
- Prompt improvements: no alternatives, instant apply
- Repetition loop detection and user warning
- ASR engine labels: Accurate / Fast
- Changelog tab in settings panel
