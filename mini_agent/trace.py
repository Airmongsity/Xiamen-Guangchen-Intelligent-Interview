"""Execution trace for tool calls.

Every tool invocation is recorded (tool, args, ok/error, latency, result preview)
and logged, so a run can be audited afterwards -- the "工具调用 trace / 执行日志"
the task asks for. Events are kept in memory and also emitted via ``logging``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger("mini_agent.trace")


@dataclass
class TraceEvent:
    session_id: str
    step: int
    tool: str
    args: dict
    ok: bool
    latency_ms: float
    result: str = ""      # truncated preview of the tool output
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Tracer:
    def __init__(self, preview_chars: int = 200) -> None:
        self.events: list[TraceEvent] = []
        self._preview = preview_chars

    def record(self, *, session_id: str, step: int, tool: str, args: dict,
               ok: bool, latency_ms: float, result: str = "", error: str = "") -> TraceEvent:
        event = TraceEvent(
            session_id=session_id, step=step, tool=tool, args=args, ok=ok,
            latency_ms=round(latency_ms, 1),
            result=(result or "")[: self._preview],
            error=error,
        )
        self.events.append(event)
        logger.info("TRACE %s", json.dumps(event.as_dict(), ensure_ascii=False, default=str))
        return event

    def for_session(self, session_id: str) -> list[TraceEvent]:
        return [e for e in self.events if e.session_id == session_id]
