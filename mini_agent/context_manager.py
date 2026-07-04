"""Context assembly and basic compression.

Two jobs:

1. :meth:`build` -- assemble the exact message list sent to the LLM each step:
   ``system prompt`` + ``rolling summary`` (if any) + ``recent verbatim turns``.
2. :meth:`maybe_compress` -- when history outgrows a threshold, summarize the
   oldest turns into the rolling summary and drop them, keeping the tail verbatim.

Only *basic* compression is implemented on purpose (the task says complex
compression is out of scope). The one subtlety that matters for correctness:
an ``assistant`` message carrying ``tool_calls`` MUST stay adjacent to its
``tool`` results, so we only ever cut the history at a ``user`` turn boundary --
never in the middle of a tool exchange (which the API would reject).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mini_agent.context")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can call tools to answer questions. "
    "Think about whether a tool is needed; if so, call it, then use its result. "
    "When you have enough information, reply to the user directly without calling a tool. "
    "Prefer the calculator for exact math and cite tool results faithfully."
)


class ContextManager:
    def __init__(self, llm: Any = None, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 max_history_messages: int = 20, keep_recent_messages: int = 8):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_history = max_history_messages
        self.keep_recent = keep_recent_messages

    def build(self, session: Any, *, memory_block: str | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if session.summary:
            messages.append({
                "role": "system",
                "content": "Summary of earlier conversation (older turns were compressed):\n"
                           + session.summary,
            })
        # Long-term memory recall is injected here: after the durable prompt/summary,
        # before the current turn's messages.
        if memory_block:
            messages.append({
                "role": "system",
                "content": "Relevant long-term memory (may help answer):\n" + memory_block,
            })
        messages.extend(session.messages)
        return messages

    def maybe_compress(self, session: Any) -> bool:
        """Compress oldest turns if history is too long. Returns True if it did."""
        if self.llm is None or len(session.messages) <= self.max_history:
            return False

        split = self._safe_split(session.messages)
        if split <= 0:
            return False  # no safe boundary yet; wait for the next turn

        old, recent = session.messages[:split], session.messages[split:]
        try:
            summary = self._summarize(old, session.summary)
        except Exception as err:  # never break a turn because summarization failed
            logger.warning("compression skipped (summarizer failed): %s", err)
            return False

        session.summary = summary
        session.messages = recent
        logger.info("compressed %d old messages; %d kept verbatim", len(old), len(recent))
        return True

    def _safe_split(self, messages: list[dict]) -> int:
        """Index of the earliest ``user`` turn at/after the keep-recent boundary.

        Cutting there keeps every assistant/tool pair intact and makes ``recent``
        start on a clean user turn. Returns 0 if no safe boundary exists.
        """
        floor = max(1, len(messages) - self.keep_recent)
        for i in range(floor, len(messages)):
            if messages[i].get("role") == "user":
                return i
        return 0

    def _summarize(self, old: list[dict], prior_summary: str) -> str:
        transcript = _render(old)
        instruction = (
            "Compress the conversation excerpt below into a concise summary. "
            "PRESERVE concrete specifics: names, numbers, dates, decisions, open "
            "questions, and any pending user requests or todos. Do not invent facts. "
            "Return only the summary."
        )
        parts = []
        if prior_summary:
            parts.append("Existing summary:\n" + prior_summary)
        parts.append("New excerpt to fold in:\n" + transcript)
        resp = self.llm.chat(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()


def _render(messages: list[dict]) -> str:
    """Flatten OpenAI-format messages into a plain transcript for summarization."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                lines.append(f"assistant -> tool {fn.get('name')}({fn.get('arguments')})")
            if m.get("content"):
                lines.append(f"assistant: {m['content']}")
        elif role == "tool":
            lines.append(f"tool result: {m.get('content', '')}")
        else:
            lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)
