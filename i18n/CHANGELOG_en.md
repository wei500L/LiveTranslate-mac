# Changelog

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
