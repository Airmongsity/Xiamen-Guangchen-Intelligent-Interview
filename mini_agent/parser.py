"""Parse an LLM response into thought / tool calls / final answer.

With OpenAI-compatible function calling the model already returns structured
``tool_calls``, but we still defend against reality:
  * the model may put its reasoning in ``content`` alongside a tool call;
  * ``tool_calls[i].function.arguments`` is a *string* that is occasionally
    malformed or truncated JSON -- so we salvage it rather than crash.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Sentinel keys used to flag an un-parseable argument blob to the runtime.
PARSE_ERROR_KEY = "_parse_error"
RAW_KEY = "_raw"


@dataclass
class ParsedResponse:
    thought: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {id, name, arguments}
    final_answer: str | None = None


def salvage_json(raw: str) -> dict[str, Any]:
    """Best-effort parse of a tool-argument string.

    Returns a dict on success; on failure returns ``{_parse_error: True, _raw: ...}``
    so the caller can feed a useful error back to the model instead of throwing.
    """
    if raw is None or not str(raw).strip():
        return {}
    text = str(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Grab the first {...} block if the model wrapped it in prose.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Try closing a truncated object.
    for suffix in ('"}', '"}}', "}", "}}"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
    return {PARSE_ERROR_KEY: True, RAW_KEY: text}


def parse_message(message: Any) -> ParsedResponse:
    """Turn an OpenAI-style message object into a :class:`ParsedResponse`.

    Works with both the real SDK object and test fakes: it only touches
    ``message.content`` and ``message.tool_calls``.
    """
    content = (getattr(message, "content", None) or "").strip()
    raw_calls = getattr(message, "tool_calls", None) or []

    tool_calls = [
        {
            "id": tc.id,
            "name": tc.function.name,
            "arguments": salvage_json(tc.function.arguments),
        }
        for tc in raw_calls
    ]

    if tool_calls:
        # Any content alongside tool calls is the model's thought.
        return ParsedResponse(thought=content, tool_calls=tool_calls, final_answer=None)
    return ParsedResponse(thought="", tool_calls=[], final_answer=content)
