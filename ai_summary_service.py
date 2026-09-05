"""AI meeting-summary service: prompts, chunking, workers and persistence.

The service is deliberately split into a pure ``SummaryRequest``/``run``
pipeline (testable without Qt, network calls injected as a callable) and a
``SummaryWorker(QThread)`` shell the UI drives with signals.

Provider selection reuses the app's persisted model list — no second client
implementation. ``ensure_model_ids()`` stamps every model entry with a stable
``id`` and migrates the saved ``ai_summary_provider`` from legacy list indexes
to IDs, so deleting a model elsewhere never silently re-points summaries at
a different provider.

Nothing here logs API keys, request bodies or record content: progress logs
carry the stage, part number and lengths only.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal

import meeting_records as records
from translator import (
    LOCAL_API_KEY_PLACEHOLDER,
    make_openai_client,
    resolve_thinking_style,
    thinking_disable_body,
)

log = logging.getLogger("LiveTranslate.Summary")

PROMPT_VERSION = 1
SUMMARY_TEMPLATES = ("meeting", "classroom")
OUTPUT_LANG_FOLLOW = "follow"

# Chinese/English bilingual prompts: the model's output language is a
# parameter, so the prompt itself stays fixed no matter what it renders.
_TEMPLATES = {
    "meeting": (
        "你是专业的会议纪要撰写人。根据提供的会议记录，用{lang}输出一份结构化 Markdown 纪要，"
        "必须包含以下小节（没有内容的小节写「无」）：\n"
        "## 会议概述\n## 核心内容\n## 重要结论\n## 决策事项\n## 待办事项\n## 待确认问题\n"
        "规则：\n"
        "- 保留重要的数字、金额、日期、人名、机构名、链接和承诺时间点，原样引用，不要改写。\n"
        "- 合并重复内容；按主题组织，不按发言顺序罗列。\n"
        "- 待办事项如有明确负责人或期限，单独一行写出。\n"
        "- 不要编造记录中没有的信息。记录含错字时按最可能的意图理解，不标注错误。\n"
        "- 只输出纪要本身，不要前言、后记或解释。"
    ),
    "classroom": (
        "你是认真的课堂笔记整理者。根据提供的课堂记录，用{lang}输出一份结构化 Markdown 笔记，"
        "必须包含以下小节（没有内容的小节写「无」）：\n"
        "## 课程主题\n## 知识点\n## 重要定义或公式\n## 教师强调内容\n## 作业与截止时间\n## 需要复习的问题\n"
        "规则：\n"
        "- 公式用 Markdown 代码块或行内代码原样保留，符号不改写。\n"
        "- 保留术语的原文形式（如首次出现时附原语言写法）。\n"
        "- 作业、考试、截止时间单独列出，保留原始日期表述。\n"
        "- 按知识结构组织，不按时间顺序罗列发言。\n"
        "- 不要编造课程中没有的内容。\n"
        "- 只输出笔记本身，不要前言、后记或解释。"
    ),
}

_CHUNK_PROMPT = (
    "你是会议记录分析员。下面是一场{kind_desc}记录的第 {index}/{total} 部分"
    "（时间区间 {span}）。用{lang}提取这一部分的关键信息，输出条目式 Markdown：\n"
    "- 保留所有数字、日期、金额、人名、决定和待办事项，原样引用。\n"
    "- 保留带时间点的关键发言（标注 [HH:MM:SS]）。\n"
    "- 忽略寒暄、重复和无实质内容的片段。\n"
    "- 只输出提取结果，不要解释你在做什么。"
)

_MERGE_PROMPT = (
    "你是会议纪要撰写人。以下是对同一场{kind_desc}（{span}，共 {total} 部分）"
    "分段提取的关键信息。合并成一份最终{lang}纪要：\n"
    "- 去除各部分之间重复的内容，合并同一话题。\n"
    "- 任何在任一部分出现的数字、日期、人名、决定、待办事项都必须保留。\n"
    "- 输出结构遵循给定的纪要模板。\n"
    "- 只输出纪要本身。"
)


class SummaryError(Exception):
    """User-visible summary failure. ``kind`` maps to an i18n key."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(detail or kind)
        self.kind = kind
        self.detail = detail


class Cancelled(Exception):
    pass


# --- provider selection ---------------------------------------------------------

def ensure_model_ids(settings: dict) -> bool:
    """Give every model entry a stable id; migrate the summary provider ref.

    Returns True when settings changed (caller persists). Legacy
    ``ai_summary_provider`` values — an int index into ``models``, or an id
    that no longer resolves — become None so the UI asks for a choice instead
    of silently summarizing with a different model.
    """
    changed = False
    models = settings.get("models")
    if not isinstance(models, list):
        return False
    existing_ids = set()
    for entry in models:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("id")
        if not isinstance(mid, str) or not mid or mid in existing_ids:
            mid = "m_" + uuid.uuid4().hex[:10]
            entry["id"] = mid
            changed = True
        existing_ids.add(mid)

    ref = settings.get("ai_summary_provider")
    if ref is None:
        return changed
    if isinstance(ref, int):  # legacy: index into models
        settings["ai_summary_provider"] = None
        if 0 <= ref < len(models):
            settings["ai_summary_provider"] = models[ref].get("id")
        return True
    if not any(isinstance(m, dict) and m.get("id") == ref for m in models):
        settings["ai_summary_provider"] = None
        return True
    return changed


def resolve_provider(settings: dict) -> dict | None:
    """The model entry selected for summaries, or None."""
    models = settings.get("models")
    if not isinstance(models, list):
        return None
    ref = settings.get("ai_summary_provider")
    if ref is None:
        return None
    for entry in models:
        if isinstance(entry, dict) and entry.get("id") == ref:
            return entry
    return None


def provider_missing_key(entry: dict) -> bool:
    """True when the entry needs a key it does not have.

    Local servers ignore keys entirely (LM Studio, Ollama /v1, the managed
    MLX server), so an entry with no key only counts as missing when its
    api_base points at a remote host rather than localhost/127.0.0.1.
    """
    api_base = str(entry.get("api_base") or "").lower()
    api_key = str(entry.get("api_key") or "").strip()
    if api_key and api_key != LOCAL_API_KEY_PLACEHOLDER:
        return False
    if not api_base or any(
        host in api_base
        for host in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "host.docker.internal")
    ):
        return False
    return True


def provider_label(entry: dict) -> str:
    return str(entry.get("name") or entry.get("model") or "?")


def provider_unsuitable(entry: dict) -> bool:
    """True when a model entry is known to be translation-specialized.

    Uses the entry's own metadata (``managed_service.type`` — the HY-MT MLX
    preset), never the model name: name matching breaks the first time a
    provider renames a model. Such models can still be picked — the records
    page labels them "not recommended" rather than hiding them — because
    whether a local translation model can follow a summarization prompt is
    not something this code can prove.
    """
    service = (entry.get("managed_service") or {})
    return service.get("type") == "mlx_lm"


# --- request assembly ------------------------------------------------------------

def _normalize_output_lang(lang: str, default: str) -> str:
    if not lang or lang == OUTPUT_LANG_FOLLOW:
        return default
    return lang


def build_request_messages(
    entries: list[dict],
    *,
    template: str = "meeting",
    output_lang: str = "中文",
    chunk_index: int | None = None,
    chunk_total: int | None = None,
    extracted: str | None = None,
) -> list[dict]:
    """Messages for one summary request (single-shot, chunk or final merge).

    Pure function so tests can assert the shape without any client.
    """
    if template not in SUMMARY_TEMPLATES:
        template = "meeting"
    body = records.entries_to_text(entries)
    if chunk_index is None:
        system = _TEMPLATES[template].format(lang=output_lang)
        user = f"以下是完整会议记录：\n\n{body}"
    else:
        system = _CHUNK_PROMPT.format(
            kind_desc=_kind_desc(template),
            index=chunk_index,
            total=chunk_total,
            span=_span(entries),
            lang=output_lang,
        )
        user = body if extracted is None else extracted
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_merge_messages(
    extracted_parts: list[str],
    *,
    template: str,
    output_lang: str,
    span_all: str,
) -> list[dict]:
    system = _MERGE_PROMPT.format(
        kind_desc=_kind_desc(template),
        span=span_all,
        total=len(extracted_parts),
        lang=output_lang,
    )
    joined = "\n\n---\n\n".join(
        f"### 第 {i + 1} 部分\n\n{text}" for i, text in enumerate(extracted_parts)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _TEMPLATES[template].format(lang=output_lang)
         + "\n\n分段提取结果：\n\n" + joined},
    ]


_KIND_DESCS = {"meeting": "会议", "classroom": "课堂"}


def _kind_desc(template: str) -> str:
    return _KIND_DESCS.get(template, "会议")


def _span(entries: list[dict]) -> str:
    if not entries:
        return "-"
    first = entries[0].get("timestamp") or "-"
    last = entries[-1].get("timestamp") or "-"
    return first if first == last else f"{first} – {last}"


def _final_system(template: str, output_lang: str) -> str:
    return _TEMPLATES[template].format(lang=output_lang)


# --- the chat callable ------------------------------------------------------------

# Summaries need far more output room than subtitles. A model tuned for live
# translation (e.g. the HY-MT preset's max_tokens=128) would silently truncate
# every summary to a couple of lines, so token limits are never inherited from
# the translation profile — the per-model override only applies when the user
# raised it above the summary floor.
SUMMARY_MIN_MAX_TOKENS = 2048
_SUMMARY_TEMPERATURE = 0.3


def make_chat_fn(model_entry: dict, timeout: float):
    """Build a ``chat(messages) -> str`` callable from a persisted model entry.

    Reuses the app's single OpenAI client factory (proxy, timeout) and its
    thinking-style rules, but deliberately does NOT inherit the translation
    profile: no translation system prompt, no conversation history, no
    streaming/JSON mode, and token limits never below the summary floor.
    Only sampling overrides that make sense for a summary pass through
    (frequency/presence penalty, seed); ``max_tokens`` applies only when the
    model entry raises it above ``SUMMARY_MIN_MAX_TOKENS``. The factory is
    resolved through the module attribute so tests (and any future
    local-provider shim) can substitute the client.
    """
    from connection_config import translation_api_base, translation_api_key, translation_model

    api_base = translation_api_base(model_entry.get("api_base"))
    api_key = translation_api_key(model_entry.get("api_key"))
    model = translation_model(model_entry.get("model"))
    proxy = model_entry.get("proxy", "none")
    client = make_openai_client(api_base, api_key, proxy, timeout=timeout)

    style = resolve_thinking_style(
        model_entry.get("thinking_style"), api_base, model
    )
    extra_body = thinking_disable_body(style)
    # extra_body is copied but capped: unknown provider-specific keys from the
    # translation profile can be huge or irrelevant to summaries, and
    # thinking/budget keys are already resolved above. Only small, scalar
    # pass-throughs survive.
    entry_extra = model_entry.get("extra_body")
    if isinstance(entry_extra, dict):
        for key, value in entry_extra.items():
            if isinstance(value, (int, float, str, bool)):
                extra_body[key] = value
    no_system_role = bool(model_entry.get("no_system_role", False))
    overrides = {
        k: v for k, v in (model_entry.get("overrides") or {}).items()
        if v is not None
    }
    max_tokens = overrides.get("max_tokens")
    if not isinstance(max_tokens, (int, float)) or max_tokens < SUMMARY_MIN_MAX_TOKENS:
        max_tokens = SUMMARY_MIN_MAX_TOKENS
    overrides = {
        k: v for k, v in overrides.items()
        if k in ("frequency_penalty", "presence_penalty", "seed")
    }
    return _chat_callable(
        client, model, extra_body, no_system_role,
        dict(overrides, max_tokens=int(max_tokens)),
    )


def _chat_callable(client, model, extra_body, no_system_role, overrides):
    def chat(messages: list[dict]) -> str:
        if no_system_role and messages and messages[0]["role"] == "system":
            messages = [
                {
                    "role": "user",
                    "content": messages[0]["content"] + "\n\n"
                    + messages[1]["content"],
                }
            ] + messages[2:]
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=overrides.get("max_tokens", SUMMARY_MIN_MAX_TOKENS),
            temperature=overrides.get("temperature", _SUMMARY_TEMPERATURE),
        )
        for key in ("top_p", "frequency_penalty", "presence_penalty", "seed"):
            if key in overrides:
                kwargs[key] = overrides[key]
        if extra_body:
            kwargs["extra_body"] = dict(extra_body)
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    # The callable carries its client so a cooperative cancel can close the
    # transport and interrupt a request already on the wire.
    chat._client = client
    return chat


# --- the pipeline --------------------------------------------------------------------

def summarize(
    entries: list[dict],
    *,
    template: str = "meeting",
    output_lang: str,
    chat,
    on_progress=None,
    cancel_event=None,
    max_chars: int = 6000,
) -> str:
    """Run the chunked summary pipeline and return the final Markdown.

    ``chat(messages) -> str`` and a ``cancel_event`` (threading.Event or
    callable returning bool) are injected, so tests drive this with fakes.
    Progress callback receives ``(stage, index, total)`` where stage is
    'part' or 'merge'.
    """
    if not entries:
        raise SummaryError("summary_empty_record")
    if cancel_event is not None and _cancelled(cancel_event):
        raise Cancelled()

    chunks = records.chunk_entries(entries, max_chars=max_chars)
    if len(chunks) <= 1:
        messages = build_request_messages(
            entries, template=template, output_lang=output_lang
        )
        if on_progress:
            on_progress("part", 1, 1)
        result = _call_chat(chat, messages)
        return _validated_output(result)

    extracted = []
    for i, chunk in enumerate(chunks):
        if cancel_event is not None and _cancelled(cancel_event):
            raise Cancelled()
        if on_progress:
            on_progress("part", i + 1, len(chunks))
        messages = build_request_messages(
            chunk,
            template=template,
            output_lang=output_lang,
            chunk_index=i + 1,
            chunk_total=len(chunks),
        )
        extracted.append(_call_chat(chat, messages))

    # Multi-level merge: the extracted parts joined together can exceed the
    # model's context just like the raw record did. When the joined extract
    # passes the chunk budget, merge it in groups first (each merge is itself
    # an extract-style reduction), so the final request stays bounded for
    # arbitrarily long meetings instead of growing linearly.
    while _joined_extract_chars(extracted) > max_chars and len(extracted) > 1:
        if cancel_event is not None and _cancelled(cancel_event):
            raise Cancelled()
        groups = _group_for_merge(extracted, max_chars)
        if len(groups) >= len(extracted):
            # Packing did not shrink the list (every merge group held a
            # single part): another round would not either — the parts are
            # individually too large to reduce further, so take the final
            # (oversized) merge rather than loop forever.
            break
        if on_progress:
            on_progress("merge", 1, len(groups))
        merged = []
        for group in groups:
            messages = build_merge_messages(
                group,
                template=template,
                output_lang=output_lang,
                span_all=_span(entries),
            )
            merged.append(_call_chat(chat, messages))
        if len(merged) >= len(extracted):
            # The model refused to compress; a further round would loop.
            break
        extracted = merged

    if cancel_event is not None and _cancelled(cancel_event):
        raise Cancelled()
    if on_progress:
        on_progress("merge", 1, 1)
    messages = build_merge_messages(
        extracted,
        template=template,
        output_lang=output_lang,
        span_all=_span(entries),
    )
    return _validated_output(_call_chat(chat, messages))


def _joined_extract_chars(parts: list[str]) -> int:
    # Mirrors build_merge_messages' join format closely enough for budgeting.
    return sum(len(p) + 24 for p in parts)


def _group_for_merge(parts: list[str], max_chars: int) -> list[list[str]]:
    """Pack parts into groups near ``max_chars`` without splitting a part."""
    groups = []
    current: list[str] = []
    current_chars = 0
    for part in parts:
        size = len(part) + 24
        if current and current_chars + size > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(part)
        current_chars += size
    if current:
        groups.append(current)
    return groups


def _call_chat(chat, messages):
    try:
        return chat(messages)
    except Cancelled:
        raise
    except SummaryError:
        raise
    except Exception as exc:
        raise _classify(exc) from exc


def _classify(exc: Exception) -> SummaryError:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timeout" in text or "timed out" in text:
        return SummaryError("summary_error_timeout", str(exc))
    if "auth" in name or "401" in text or "api key" in text:
        return SummaryError("summary_error_auth", str(exc))
    if "connection" in name or "connect" in text or "refused" in text or (
        "unreachable" in text
    ):
        return SummaryError("summary_error_unreachable", str(exc))
    if "rate" in text and "limit" in text:
        return SummaryError("summary_error_ratelimit", str(exc))
    return SummaryError("summary_error_api", str(exc))


def _validated_output(result: str) -> str:
    if not result or not result.strip():
        raise SummaryError("summary_error_empty")
    return result.strip()


def _cancelled(cancel_event) -> bool:
    if cancel_event is None:
        return False
    if callable(cancel_event):
        return bool(cancel_event())
    return bool(cancel_event.is_set())


# --- Qt worker ----------------------------------------------------------------------

class SummaryWorker(QThread):
    """Run one summary generation off the UI thread.

    Signals are the only channel back to the UI: progress (translated by the
    page, not here), success (content + metadata for persistence) or failure
    (an i18n kind + detail). Cancellation is cooperative — the worker checks
    the event between requests and never emits success after cancel.

    Ownership: the parent widget must keep the worker alive until ``run()``
    ends; ``cleanup()`` on the page cancels and, if the thread cannot finish
    quickly, detaches it to the QApplication so a slow request never takes
    the page down with it (the thread quits on its own; results emitted
    after the page is gone are dropped by Qt's dead-receiver cleanup).
    """

    progress = pyqtSignal(str, int, int)   # stage, index, total
    succeeded = pyqtSignal(str, dict)      # content, meta
    failed = pyqtSignal(str, str)          # i18n kind, detail

    # Per-request timeout. 300s meant quitting the app could block on a hung
    # request for five minutes; 120s still covers a cold local model's first
    # load while keeping a cooperative-cancel wait bounded.
    REQUEST_TIMEOUT_S = 120.0

    def __init__(
        self,
        base_dir,
        session: str,
        entries: list[dict],
        provider_entry: dict,
        *,
        template: str = "meeting",
        output_lang: str,
        default_output_lang: str,
        generation: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._base_dir = base_dir
        self._session = session
        self._entries = entries
        self._provider = provider_entry
        self._template = template if template in SUMMARY_TEMPLATES else "meeting"
        self._output_lang = _normalize_output_lang(output_lang, default_output_lang)
        # Generation token: the page stamps every create with a new one and
        # verifies it on each callback, so a worker from an earlier run (or an
        # earlier session) can never update the current view.
        self.generation = generation
        # Public read-only mirror of the session for UI-side validation.
        self.session = session
        self._cancel = threading.Event()
        self._save_lock = threading.Lock()
        self._saved = False
        # The chat callable's underlying OpenAI client (held once built in
        # run()) so cancel can close it: closing the httpx transport is the
        # only way to interrupt a synchronous request already on the wire —
        # the cooperative cancel flag alone waits for it to return.
        self._client = None

    def cancel(self):
        # The flag is set under _save_lock so the flag-set linearizes with
        # the save-and-emit section of run(): a cancel either lands before
        # that section (which then observes the flag and keeps the old
        # summary) or after it has fully completed (the new summary is
        # legitimately saved and announced). There is no window where a
        # cancel is acknowledged yet the new summary still overwrites the
        # old one afterwards.
        with self._save_lock:
            self._cancel.set()
        # Best-effort interrupt of an in-flight request: httpx closes the
        # connection pool, and the openai SDK's next read off that transport
        # raises instead of waiting out the timeout. Not instantaneous — a
        # request mid-read can still take a moment to surface the error —
        # and a genuinely hung socket is only bounded by the per-request
        # timeout. Honest limits, stated here rather than in the registry.
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                log.debug("Summary client close failed", exc_info=True)

    def attach_client(self, client):
        """Run() reports the client it built, so a later cancel can close it."""
        self._client = client

    def run(self):
        started = time.time()
        try:
            chat = make_chat_fn(self._provider, timeout=self.REQUEST_TIMEOUT_S)
            # Give the cooperative cancel a hard handle on the transport: a
            # cancel between requests only sets a flag, a cancel during a
            # request needs the client closed to interrupt the read.
            self.attach_client(getattr(chat, "_client", None))
            content = summarize(
                self._entries,
                template=self._template,
                output_lang=self._output_lang,
                chat=chat,
                on_progress=self._emit_progress,
                cancel_event=self._cancel,
            )
        except Cancelled:
            # A cancel that landed before the first request: the client was
            # built after (or during) the cancel, so cancel() had no handle
            # to close — close it here so the fresh connection pool is not
            # left dangling for the process lifetime.
            self._close_client_quietly()
            return  # no signal: the UI already switched to its cancel state
        except SummaryError as exc:
            log.warning(
                "Summary failed (%s) after %.1fs, %d chars of record",
                exc.kind, time.time() - started, len(records.entries_to_text(self._entries)),
            )
            self._close_client_quietly()
            self.failed.emit(exc.kind, exc.detail)
            return
        except Exception as exc:  # worker must never die silently
            log.error("Summary worker crashed", exc_info=True)
            self._close_client_quietly()
            self.failed.emit("summary_error_api", str(exc))
            return

        meta = {
            "provider_id": self._provider.get("id"),
            "provider_name": provider_label(self._provider),
            "model": self._provider.get("model"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_hash": records.source_hash(self._entries),
            "template": self._template,
            "output_language": self._output_lang,
            "prompt_version": PROMPT_VERSION,
            "entries_count": len(self._entries),
            "generation": self.generation,
        }
        # Save + announce under the same lock cancel() sets its flag under:
        # once this section starts, a concurrent cancel waits for it, so a
        # completed save is announced exactly once and an acknowledged
        # cancel never sees a newer summary overwrite the old one after the
        # fact. The emits stay inside the lock: releasing between save and
        # emit would re-open the window.
        with self._save_lock:
            if self._cancel.is_set():
                self._close_client_quietly()
                return  # raced a cancel: keep the old summary untouched
            ok = records.save_summary(
                self._base_dir, self._session, content, meta
            )
            self._saved = ok
            self._close_client_quietly()
            if ok:
                self.succeeded.emit(content, meta)
            else:
                self.failed.emit("summary_error_save", "")

    def _close_client_quietly(self):
        """Close the run's client once it is no longer needed (every terminal
        path of run(): cancel, failure, crash, and after the save). Closing
        twice is harmless (cancel may already have closed it to interrupt a
        request); leaving it open on the cancelled-before-request path is
        what this fixes — a connection pool dangling for the process
        lifetime each time a cancel lands before the first request."""
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                log.debug("Summary client close failed", exc_info=True)

    def _emit_progress(self, stage: str, index: int, total: int):
        # Never log content — the stage and part numbers are all the log needs.
        log.debug("Summary %s %d/%d", stage, index, total)
        self.progress.emit(stage, index, total)
