"""Chat LLMs over OpenAI-compatible APIs.

`DeepSeekLLM` (deepseek-chat) is the shipped default. `MiMoLLM` adds Xiaomi MiMo
compatibility: MiMo-RL is a *reasoning* model, so it (a) emits a `<think>...</think>`
block before its answer and (b) does not reliably honour the `response_format`
JSON mode the way deepseek-chat does. The base class handles both: it can strip the
reasoning block and fall back to brace-extraction parsing when JSON mode is off.
Select between them with `make_llm(config)` driven by `config.llm_provider`.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from ..config import AutoMemConfig

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICompatibleLLM:
    """Generic OpenAI-compatible chat client. Subclasses fill in credentials."""

    track_usage = False  # set True to accumulate per-call token/cache stats in .usage

    #: whether the model honours response_format={"type": "json_object"}.
    supports_json_mode = True
    #: whether to strip a reasoning model's <think>...</think> preamble from output.
    strip_think = False

    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=6,  # ride out 429 TPM windows (reset ~once per minute)
            timeout=60.0,
        )
        self._model = model
        self.usage: dict[str, dict[str, int]] = {}

    def _record(self, system: str, resp) -> None:
        if not self.track_usage:
            return
        # bucket by which fixed system prompt drove the call (extraction/reconcile/...)
        tag = system.split("\n", 1)[0][:32]
        u = getattr(resp, "usage", None)
        b = self.usage.setdefault(
            tag, {"calls": 0, "prompt": 0, "cache_hit": 0, "completion": 0}
        )
        b["calls"] += 1
        if u is not None:
            b["prompt"] += getattr(u, "prompt_tokens", 0) or 0
            b["completion"] += getattr(u, "completion_tokens", 0) or 0
            # DeepSeek reports prefix-cache hits here
            b["cache_hit"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0

    def _clean(self, text: str) -> str:
        text = text or ""
        if self.strip_think:
            text = _THINK_RE.sub("", text)
        return text

    def complete(self, system: str, user: str, *, temperature: float = 0.1) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        self._record(system, resp)
        return self._clean(resp.choices[0].message.content).strip()

    def complete_json(self, system: str, user: str, *, temperature: float = 0.1):
        """Complete with JSON output and parse the result."""
        kwargs = {}
        if self.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        self._record(system, resp)
        text = self._clean(resp.choices[0].message.content) or "{}"
        return _parse_json_lenient(text)


class DeepSeekLLM(OpenAICompatibleLLM):
    """Shipped default: deepseek-chat. Native JSON mode, no reasoning preamble."""

    def __init__(self, config: AutoMemConfig):
        if not config.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        super().__init__(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.llm_model,
        )


class MiMoLLM(OpenAICompatibleLLM):
    """Xiaomi MiMo (e.g. MiMo-7B-RL) over an OpenAI-compatible endpoint.

    Reasoning model: emits <think>...</think> (stripped) and is not guaranteed to
    support JSON response_format, so we parse leniently from the cleaned text.
    Credentials default to SiliconFlow (the open-model endpoint AutoMem already
    uses); override MIMO_BASE_URL / MIMO_API_KEY / MIMO_MODEL in .env as needed.
    """

    supports_json_mode = False
    strip_think = True

    def __init__(self, config: AutoMemConfig):
        if not config.mimo_api_key:
            raise ValueError(
                "MIMO_API_KEY is not set (falls back to SILICONFLOW_API_KEY if that "
                "is configured; see AutoMemConfig.from_env)"
            )
        super().__init__(
            api_key=config.mimo_api_key,
            base_url=config.mimo_base_url,
            model=config.mimo_model,
        )


def make_llm(config: AutoMemConfig) -> OpenAICompatibleLLM:
    """Construct the chat LLM selected by config.llm_provider ('deepseek' | 'mimo')."""
    provider = (config.llm_provider or "deepseek").lower()
    if provider == "mimo":
        return MiMoLLM(config)
    if provider == "deepseek":
        return DeepSeekLLM(config)
    raise ValueError(f"Unknown llm_provider: {config.llm_provider!r}")


def _parse_json_lenient(text: str):
    """Parse JSON, tolerating markdown code fences and truncated responses.

    A very long session can drive the extractor past the output-token cap, cutting
    the JSON off mid-string. Rather than lose the whole call (and, in a batch run,
    abort everything), we salvage the array elements that did complete."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        salvaged = _salvage_truncated(text)
        if salvaged is not None:
            return salvaged
        # last resort: a reasoning model may wrap the object in prose; grab the
        # outermost {...} and retry once.
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}  # unrecoverable: caller fails safe (extract -> [], reconcile -> ADD)


def _salvage_truncated(text: str):
    """Recover a truncated {"memories": [ {...}, {...}, <cut> ]} by keeping every
    array element that fully decoded."""
    m = re.search(r'"memories"\s*:\s*\[', text)
    if not m:
        return None
    decoder = json.JSONDecoder()
    idx, n = m.end(), len(text)
    items: list = []
    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n or text[idx] == "]":
            break
        try:
            obj, idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break  # the truncated tail object
        items.append(obj)
    return {"memories": items} if items else None
