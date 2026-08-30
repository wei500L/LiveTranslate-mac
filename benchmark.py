import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from translator import stream_option_errors, translator_from_model_config

log = logging.getLogger("LiveTranslate.Benchmark")

BENCH_SENTENCES = {
    "ja": [
        "こんにちは、今日はいい天気ですね。",
        "この映画はとても面白かったです。",
        "明日の会議は何時からですか？",
        "日本の桜は本当に美しいですね。",
        "新しいレストランに行ってみましょう。",
    ],
    "en": [
        "Hello, the weather is nice today.",
        "That movie was really interesting.",
        "What time does tomorrow's meeting start?",
        "The cherry blossoms in Japan are truly beautiful.",
        "Let's try going to the new restaurant.",
    ],
    "zh": [
        "你好，今天天气真不错。",
        "那部电影真的很有意思。",
        "明天的会议几点开始？",
        "日本的樱花真的很美丽。",
        "我们去试试那家新餐厅吧。",
    ],
    "ko": [
        "안녕하세요, 오늘 날씨가 좋네요.",
        "그 영화 정말 재미있었어요.",
        "내일 회의는 몇 시부터인가요?",
        "일본의 벚꽃은 정말 아름답네요.",
        "새로운 레스토랑에 가볼까요?",
    ],
    "fr": [
        "Bonjour, il fait beau aujourd'hui.",
        "Ce film était vraiment intéressant.",
        "À quelle heure commence la réunion demain?",
        "Les cerisiers en fleurs au Japon sont magnifiques.",
        "Allons essayer le nouveau restaurant.",
    ],
    "de": [
        "Hallo, heute ist schönes Wetter.",
        "Der Film war wirklich interessant.",
        "Um wie viel Uhr beginnt das Meeting morgen?",
        "Die Kirschblüten in Japan sind wunderschön.",
        "Lass uns das neue Restaurant ausprobieren.",
    ],
}


def build_bench_translator(model: dict, prompt: str, target_lang: str, timeout_s):
    """A Translator configured exactly like the runtime one, minus the state.

    context_turns stays 0 and nothing here touches the app's active Translator:
    a benchmark must measure the real request shape without writing history or
    changing what the pipeline is using.
    """
    return translator_from_model_config(
        model,
        target_language=target_lang,
        system_prompt=model.get("system_prompt") or prompt,
        timeout=timeout_s,
    )


def run_benchmark(models, source_lang, target_lang, timeout_s, prompt, result_callback):
    """Run benchmark in a background thread. Calls result_callback(str) for each output line."""
    sentences = BENCH_SENTENCES.get(source_lang, BENCH_SENTENCES["en"])
    rounds = len(sentences)

    result_callback(
        f"Testing {len(models)} model(s) x {rounds} rounds  |  "
        f"timeout={timeout_s}s  |  {source_lang} -> {target_lang}\n"
        f"{'=' * 60}\n"
    )

    def _test_model(m):
        # Inside no try of its own: a config entry without "name" raised here,
        # the future re-raised it in _run_all, "__DONE__" was never sent and the
        # Test button stayed disabled until the app was restarted.
        name = m.get("name") or m.get("model") or "(unnamed)"
        lines = [f"Model: {name}", f"  {'─' * 50}"]
        try:
            bench_translator = build_bench_translator(
                m, prompt, target_lang, timeout_s
            )
            client = bench_translator._client
            system_prompt = bench_translator._build_system_prompt(source_lang)
            ttfts = []
            totals = []

            for i, text in enumerate(sentences):
                # Exactly the runtime request: thinking style, overrides,
                # extra_body, response_format and the system/user role split all
                # come from the same assembly point the pipeline uses, so a
                # benchmark result describes the request the app will actually
                # send.
                kwargs = bench_translator._build_request_kwargs(
                    system_prompt, text, stream=True
                )
                try:
                    t0 = time.perf_counter()
                    stream = client.chat.completions.create(**kwargs)
                    ttft = None
                    chunks = []
                    for chunk in stream:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        # A usage-only frame has no choices; indexing [0] here
                        # raised IndexError and silently fell through to the
                        # non-streaming path, polluting the latency numbers.
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta.content:
                            chunks.append(delta.content)
                    total_ms = (time.perf_counter() - t0) * 1000
                    result_text = "".join(chunks).strip()
                    ttft = ttft or total_ms
                except Exception as exc:
                    # Only fall back for a rejected stream parameter. A
                    # connection error retried here costs a second full timeout
                    # per sentence, multiplied by the round count.
                    if not isinstance(exc, stream_option_errors()):
                        raise
                    kwargs.pop("stream", None)
                    kwargs.pop("stream_options", None)
                    t0 = time.perf_counter()
                    resp = client.chat.completions.create(**kwargs)
                    total_ms = (time.perf_counter() - t0) * 1000
                    ttft = total_ms
                    # `or ""`: a thinking model that burned its whole budget
                    # returns content=None, and this tool is exactly what a user
                    # reaches for to diagnose that (issue #38).
                    result_text = (resp.choices[0].message.content or "").strip()

                ttfts.append(ttft)
                totals.append(total_ms)
                lines.append(
                    f"  Round {i + 1}: {total_ms:7.0f}ms "
                    f"(TTFT {ttft:6.0f}ms) | {result_text[:60]}"
                )

            avg_total = statistics.mean(totals)
            std_total = statistics.stdev(totals) if len(totals) > 1 else 0
            avg_ttft = statistics.mean(ttfts)
            std_ttft = statistics.stdev(ttfts) if len(ttfts) > 1 else 0
            lines.append(
                f"  Avg: {avg_total:.0f}ms \u00b1 {std_total:.0f}ms  "
                f"(TTFT: {avg_ttft:.0f}ms \u00b1 {std_ttft:.0f}ms)"
            )

            result_callback("\n".join(lines))
            return {
                "name": name,
                "avg_ttft": avg_ttft,
                "std_ttft": std_ttft,
                "avg_total": avg_total,
                "std_total": std_total,
                "error": None,
            }

        except Exception as e:
            err_msg = str(e).split("\n")[0][:120]
            lines.append(f"  FAILED: {err_msg}")
            result_callback("\n".join(lines))
            return {
                "name": name,
                "avg_ttft": 0,
                "std_ttft": 0,
                "avg_total": 0,
                "std_total": 0,
                "error": err_msg,
            }

    def _run_all():
        try:
            _benchmark_all()
        except Exception:
            # The caller re-enables its button on "__DONE__", so anything that
            # escapes here would disable it for the rest of the session.
            log.error("Benchmark run failed", exc_info=True)
            result_callback(f"\nBenchmark aborted: see the log for details")
        finally:
            result_callback("__DONE__")

    def _benchmark_all():
        results = []
        with ThreadPoolExecutor(max_workers=max(1, len(models))) as pool:
            futures = {pool.submit(_test_model, m): m for m in models}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    log.error("Benchmark worker failed", exc_info=True)
                    results.append({
                        "name": (futures[fut] or {}).get("name", "?"),
                        "avg_ttft": 0, "std_ttft": 0,
                        "avg_total": 0, "std_total": 0,
                        "error": str(exc).split("\n")[0][:120],
                    })

        ok = [r for r in results if not r["error"]]
        ok.sort(key=lambda r: r["avg_ttft"])
        result_callback(f"\n{'=' * 60}")
        result_callback("Ranking by Avg TTFT:")
        for i, r in enumerate(ok):
            result_callback(
                f"  #{i + 1}  TTFT {r['avg_ttft']:6.0f}ms \u00b1 {r['std_ttft']:4.0f}ms  "
                f"Total {r['avg_total']:6.0f}ms \u00b1 {r['std_total']:4.0f}ms  "
                f"{r['name']}"
            )
        failed = [r for r in results if r["error"]]
        for r in failed:
            result_callback(f"  FAIL  {r['name']}: {r['error']}")

    threading.Thread(target=_run_all, daemon=True).start()
