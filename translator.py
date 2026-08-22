import json
import logging
import time

import httpx
from openai import OpenAI
from connection_config import normalize_api_base

log = logging.getLogger("LiveTranslate.TL")

LANGUAGE_DISPLAY = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "hu": "Hungarian",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "he": "Hebrew",
}

DEFAULT_PROMPT = (
    "你是俄语课堂的实时翻译助手。请将课堂中的{source_lang}内容翻译成{target_lang}。\n"
    "场景：学校或大学课堂，内容可能包括教师讲解、学生提问、课堂讨论、例句、术语、板书和作业要求。\n"
    "规则：\n"
    "- 只输出一条准确、自然的{target_lang}译文，不要解释、分析、前缀、引号或多个候选。\n"
    "- 保持教师讲解或学生发言的逻辑、语气、否定、条件、因果、时间和指代关系，不擅自补充未说内容。\n"
    "- 课程术语、人名、地名、书名、课程名、缩写、数字、公式和符号使用目标语言通行表达；没有把握时保留原文，不要臆造。\n"
    "- 结合课堂语境和近期上下文纠正俄语 ASR 的错词、同音词和断句；无法确定时忠实翻译，不要编造。\n"
    "- 可适度压缩口语重复和填充词，但不要省略定义、例子、数字、公式、作业要求或关键限定。\n"
    "- 保持适合实时字幕的简洁长度；原句未完时翻译当前可确定的内容，不添加说明。\n"
    "近期课堂上下文：\n"
    "{context}"
)

PROMPT_PRESETS = {
    "classroom": DEFAULT_PROMPT,
    "daily": (
        "You are a real-time subtitle translator for casual conversation. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep proper nouns, names, and brand names untranslated.\n"
        "- Use natural, casual, everyday language. Keep it conversational and concise.\n"
        "- Auto-correct likely ASR errors based on context and common sense."
    ),
    "esports": (
        "You are a real-time subtitle translator for esports/gaming live streams. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep player names (IGN), team names, game terms, and brand names untranslated.\n"
        "- Use energetic, concise language appropriate for competitive gaming commentary.\n"
        "- Auto-correct likely ASR errors based on context and common sense."
    ),
    "anime": (
        "You are a real-time subtitle translator for anime, movies, and TV shows. "
        "Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Keep character names, place names, and cultural terms untranslated.\n"
        "- Use natural, expressive language that matches the tone and emotion of the dialogue.\n"
        "- Auto-correct likely ASR errors based on context and common sense."
    ),
    "webid": (
        "You are a real-time subtitle translator for an online identity-verification "
        "(WebID / video KYC) call. Translate {source_lang} into {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY one single best translation, nothing else.\n"
        "- Never include alternatives, parenthetical options, annotations, or explanations.\n"
        "- Context: a verification agent and a customer on a video call inspect ID documents "
        "(passport, ID card). Words about reading or seeing refer to the document or the camera "
        "image, NOT literacy — e.g. 'I can't read it' means the text/photo is unclear, not that "
        "the person is illiterate.\n"
        "- Render camera/document instructions naturally (hold it up, tilt it, move closer, "
        "lighting, focus, read the number aloud, turn it over).\n"
        "- Keep names, document numbers, and verification codes exactly as spoken.\n"
        "- Auto-correct likely ASR errors based on this verification context."
    ),
}


def make_openai_client(
    api_base: str, api_key: str, proxy: str = "none", timeout=None
) -> OpenAI:
    kwargs = {"base_url": normalize_api_base(api_base), "api_key": api_key}
    if timeout is not None:
        kwargs["timeout"] = httpx.Timeout(timeout, connect=5.0)
    if proxy == "system":
        pass
    elif proxy in ("none", "", None):
        kwargs["http_client"] = httpx.Client(trust_env=False)
    else:
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    return OpenAI(**kwargs)


class RepetitionError(Exception):
    """Raised when model output contains repetition loops."""
    pass


_OVERRIDE_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


# Per-provider request shapes that turn thinking/reasoning off. Thinking
# left ON silently burns the whole max_tokens budget on reasoning and the
# completion comes back empty (issue #38), which the UI then renders as
# untranslated same-language text.
#   deepseek: DeepSeek API, Volcano Ark, Zhipu GLM (nested thinking object)
#   qwen:     DashScope/Model Studio, SiliconFlow (flat enable_thinking)
#   vllm:     self-hosted vLLM/SGLang (chat template kwarg)
#   openai:   OpenAI GPT-5.1+/Grok 4.3+ (reasoning_effort=none)
#   off:      send nothing (non-thinking models, LM Studio, Ollama /v1)
THINKING_STYLES = ("auto", "deepseek", "qwen", "vllm", "openai", "off")

_NESTED_THINKING_MODELS = ("deepseek", "glm")
_NESTED_THINKING_ENDPOINTS = ("deepseek", "volces", "api.z.ai", "bigmodel")
_PARAMLESS_ENDPOINTS = ("api.openai.com", "api.x.ai", "api.anthropic.com")


def resolve_thinking_style(style, api_base, model) -> str:
    """Resolve a thinking_style setting to a concrete provider style.

    "auto" guesses from the endpoint/model id; official OpenAI-like
    endpoints get "off" because they reject unknown request parameters.
    """
    if style in THINKING_STYLES and style != "auto":
        return style
    endpoint = str(api_base or "").lower()
    model_id = str(model or "").lower()
    if any(m in model_id for m in _NESTED_THINKING_MODELS) or any(
        m in endpoint for m in _NESTED_THINKING_ENDPOINTS
    ):
        return "deepseek"
    if any(m in endpoint for m in _PARAMLESS_ENDPOINTS):
        return "off"
    return "qwen"


def thinking_disable_body(style: str) -> dict:
    """Request-body fragment that disables thinking for a concrete style."""
    if style == "deepseek":
        return {"thinking": {"type": "disabled"}}
    if style == "qwen":
        return {"enable_thinking": False}
    if style == "vllm":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if style == "openai":
        return {"reasoning_effort": "none"}
    return {}


class Translator:
    """LLM-based translation using OpenAI-compatible API."""

    def __init__(
        self,
        api_base,
        api_key,
        model,
        target_language="zh",
        max_tokens=256,
        temperature=0.3,
        streaming=True,
        system_prompt=None,
        proxy="none",
        no_system_role=False,
        no_think=False,
        json_response=False,
        timeout=10,
        overrides=None,
        extra_body=None,
        thinking_style=None,
    ):
        self._client = make_openai_client(api_base, api_key, proxy, timeout=timeout)
        self._no_system_role = no_system_role
        if thinking_style is None:
            # Legacy configs only carry the no_think bool
            thinking_style = "auto" if no_think else "off"
        self._thinking_style = resolve_thinking_style(
            thinking_style, api_base, model
        )
        self._json_response = json_response
        if self._thinking_style != "off":
            log.info(
                f"Translator: thinking disabled for {model} via "
                f"{self._thinking_style} style "
                f"({thinking_disable_body(self._thinking_style)})"
            )
        if json_response:
            log.info(f"Translator: json_response enabled for {model}")
        self._model = model
        self._target_language = target_language
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._streaming = streaming
        self._timeout = timeout
        self._overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self._extra_body = dict(extra_body) if extra_body else {}
        if self._overrides:
            log.info(f"Translator overrides: {self._overrides}")
        if self._extra_body:
            log.info(f"Translator extra_body: {self._extra_body}")
        self._system_prompt_template = system_prompt or DEFAULT_PROMPT
        self._context_turns = 0
        self._history = []  # list of (source_text, translated_text)
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0

    @property
    def last_usage(self):
        """(prompt_tokens, completion_tokens) from last translate call."""
        return self._last_prompt_tokens, self._last_completion_tokens

    def set_target_language(self, target_language: str):
        self._target_language = target_language

    def set_timeout(self, timeout: int):
        self._timeout = timeout
        self._client = self._client.copy(timeout=timeout)

    def set_context_turns(self, n: int):
        self._context_turns = n
        if n == 0:
            self._history.clear()

    def clear_history(self):
        self._history.clear()

    def _format_context(self) -> str:
        if self._context_turns <= 0 or not self._history:
            return ""
        lines = []
        for src, tgt in self._history[-self._context_turns:]:
            lines.append(f"Source: {src}")
            lines.append(f"Translation: {tgt}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def with_target_language(self, target_language: str) -> "Translator":
        """Create a new Translator with a different target language, sharing the same client."""
        t = Translator.__new__(Translator)
        t._client = self._client
        t._no_system_role = self._no_system_role
        t._thinking_style = self._thinking_style
        t._json_response = self._json_response
        t._model = self._model
        t._target_language = target_language
        t._max_tokens = self._max_tokens
        t._temperature = self._temperature
        t._streaming = self._streaming
        t._timeout = self._timeout
        t._overrides = dict(self._overrides)
        t._extra_body = dict(self._extra_body)
        t._system_prompt_template = self._system_prompt_template
        t._context_turns = 0
        t._history = []
        t._last_prompt_tokens = 0
        t._last_completion_tokens = 0
        return t

    def fork_for_request(self, target_language=None, history_snapshot=None) -> "Translator":
        """Create isolated mutable state for a concurrent translation request."""
        t = self.with_target_language(target_language or self._target_language)
        t._context_turns = self._context_turns
        t._history = list(
            history_snapshot if history_snapshot is not None else self._history
        )
        return t

    def _build_system_prompt(self, source_lang):
        src = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        tgt = LANGUAGE_DISPLAY.get(self._target_language, self._target_language)
        try:
            prompt = self._system_prompt_template.format(
                source_lang=src,
                target_lang=tgt,
                context=self._format_context(),
            )
        except (KeyError, IndexError, ValueError) as e:
            log.warning(f"Bad prompt template, falling back to default: {e}")
            prompt = DEFAULT_PROMPT.format(source_lang=src, target_lang=tgt)
        if self._json_response:
            prompt += '\nRespond in JSON format: {"t": "translated text"}'
        return prompt

    def _build_messages(self, system_prompt, text):
        if self._no_system_role:
            msgs = [{"role": "user", "content": f"{system_prompt}\n{text}"}]
        else:
            msgs = [{"role": "system", "content": system_prompt}]
            # Append recent history as context
            if (
                self._context_turns > 0
                and self._history
                and "{context}" not in self._system_prompt_template
            ):
                for src, tgt in self._history[-self._context_turns:]:
                    msgs.append({"role": "user", "content": src})
                    msgs.append({"role": "assistant", "content": tgt})
            msgs.append({"role": "user", "content": text})
        return msgs

    def _append_history(self, text, result):
        if self._context_turns > 0 and result:
            self._history.append((text, result))
            max_keep = self._context_turns + 2
            if len(self._history) > max_keep:
                self._history = self._history[-self._context_turns:]

    def _build_request_kwargs(self, system_prompt, text, stream=False):
        kwargs = dict(
            model=self._model,
            messages=self._build_messages(system_prompt, text),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        for k in _OVERRIDE_KEYS:
            if k in self._overrides:
                kwargs[k] = self._overrides[k]
        extra_body = thinking_disable_body(self._thinking_style)
        if self._extra_body:
            extra_body.update(self._extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self._json_response:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"t": {"type": "string"}},
                        "required": ["t"],
                        "additionalProperties": False,
                    },
                },
            }
        if stream:
            kwargs["stream"] = True
        return kwargs

    def translate(self, text: str, source_language: str = "en"):
        system_prompt = self._build_system_prompt(source_language)
        if self._streaming:
            result = self._translate_streaming(system_prompt, text)
        else:
            result = self._translate_sync(system_prompt, text)
        if self._check_repetition(result):
            raise RepetitionError(result)
        self._append_history(text, result)
        return result

    def translate_iter(self, text: str, source_language: str = "en"):
        """Generator that yields accumulated partial text, then final result.

        Non-streaming or json_response mode: yields once with the final result.
        Streaming mode: yields partial accumulated text as chunks arrive.
        The final yielded value is always the complete translation.
        Caller should use the last yielded value as the final result.
        """
        system_prompt = self._build_system_prompt(source_language)
        if not self._streaming:
            result = self._translate_sync(system_prompt, text)
            self._append_history(text, result)
            yield result
            return

        # Streaming path
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        base_kwargs = self._build_request_kwargs(system_prompt, text, stream=True)
        try:
            stream = self._client.chat.completions.create(
                **base_kwargs,
                stream_options={"include_usage": True},
            )
        except Exception:
            stream = self._client.chat.completions.create(**base_kwargs)

        deadline = time.monotonic() + self._timeout
        chunks = []
        for chunk in stream:
            if time.monotonic() > deadline:
                stream.close()
                raise TimeoutError(
                    f"Translation exceeded {self._timeout}s total timeout"
                )
            if hasattr(chunk, "usage") and chunk.usage:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
                    if not self._json_response:
                        yield "".join(chunks)
        result = "".join(chunks).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        self._warn_if_thinking_burned(result)
        if self._check_repetition(result):
            raise RepetitionError(result)
        self._append_history(text, result)
        yield result

    def _extract_json_translation(self, raw: str) -> str:
        """Extract translation from JSON response, fallback to raw text."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "t" in data:
                return data["t"]
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    def _warn_if_thinking_burned(self, result: str):
        """Diagnose empty completions caused by an unclosed thinking mode."""
        if not result and self._last_completion_tokens > 0:
            log.warning(
                f"Empty translation but {self._last_completion_tokens} completion "
                "tokens were used - the model likely spent the whole max_tokens "
                "budget on reasoning; pick the correct thinking style for this "
                f"provider in the model edit dialog (current: {self._thinking_style})"
            )

    @staticmethod
    def _check_repetition(text: str) -> bool:
        """Detect repetition loops in model output."""
        if not text or len(text) < 40:
            return False
        for plen in range(8, len(text) // 2 + 1):
            if text[plen:plen * 2] == text[:plen]:
                return True
        return False

    def _translate_sync(self, system_prompt, text):
        kwargs = self._build_request_kwargs(system_prompt, text, stream=False)
        resp = self._client.chat.completions.create(**kwargs)
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        if resp.usage:
            self._last_prompt_tokens = resp.usage.prompt_tokens or 0
            self._last_completion_tokens = resp.usage.completion_tokens or 0
        result = (resp.choices[0].message.content or "").strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        self._warn_if_thinking_burned(result)
        return result

    def _translate_streaming(self, system_prompt, text):
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        base_kwargs = self._build_request_kwargs(system_prompt, text, stream=True)
        try:
            stream = self._client.chat.completions.create(
                **base_kwargs,
                stream_options={"include_usage": True},
            )
        except Exception:
            stream = self._client.chat.completions.create(**base_kwargs)

        deadline = time.monotonic() + self._timeout
        chunks = []
        for chunk in stream:
            if time.monotonic() > deadline:
                stream.close()
                raise TimeoutError(
                    f"Translation exceeded {self._timeout}s total timeout"
                )
            if hasattr(chunk, "usage") and chunk.usage:
                self._last_prompt_tokens = chunk.usage.prompt_tokens or 0
                self._last_completion_tokens = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
        result = "".join(chunks).strip()
        if self._json_response:
            result = self._extract_json_translation(result)
        self._warn_if_thinking_burned(result)
        return result
