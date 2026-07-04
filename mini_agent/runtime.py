"""The Agent runtime -- the self-implemented core loop.

One turn of :meth:`Agent.chat` runs this loop:

    receive user input
      -> ask the LLM (decide: answer directly, or call tools?)
      -> if final answer: return it to the user
      -> if tool call(s): execute, append results, loop again
      -> stop at max_steps (forced finalize) or on repeated tool failure

No agent framework is used; orchestration lives here. The LLM, tool registry,
sessions, context manager, and tracer are all injected so each is testable and
replaceable.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .context_manager import DEFAULT_SYSTEM_PROMPT, ContextManager
from .memory import NullMemory
from .parser import PARSE_ERROR_KEY, RAW_KEY, parse_message
from .session import SessionManager
from .tools import ToolContext, ToolRegistry, default_registry

logger = logging.getLogger("mini_agent.runtime")

# Stop if this many steps in a row produced only failing tool calls.
_MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class AgentResult:
    answer: str
    session_id: str
    steps: int
    stop_reason: str  # "final" | "max_steps" | "tool_failures"
    thoughts: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, *, llm: Any, registry: ToolRegistry | None = None,
                 sessions: SessionManager | None = None,
                 context: ContextManager | None = None, tracer: Any = None,
                 memory: Any = None, max_steps: int = 8,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.llm = llm
        self.registry = registry or default_registry()
        self.sessions = sessions or SessionManager()
        self.context = context or ContextManager(llm, system_prompt=system_prompt)
        self.tracer = tracer
        self.memory = memory or NullMemory()
        self.max_steps = max_steps

    # -- public API ---------------------------------------------------------
    def chat(self, user_input: str, session_id: str | None = None,
             user_id: str = "default") -> AgentResult:
        session = self.sessions.get_or_create(session_id, user_id=user_id)
        session.add({"role": "user", "content": user_input})
        self.context.maybe_compress(session)

        # Long-term memory recall: once per turn, before the loop (not per step).
        # Namespaced by user_id (like todos), so a user's facts follow them across
        # windows; conversation context stays per-session.
        memory_block = self._recall(user_input, session.user_id)

        ctx = ToolContext(session=session)
        thoughts: list[str] = []
        consecutive_failures = 0

        for step in range(1, self.max_steps + 1):
            messages = self.context.build(session, memory_block=memory_block)
            try:
                resp = self.llm.chat(messages, tools=self.registry.schemas())
            except Exception as err:  # LLM itself failed after its own retries
                logger.error("LLM call failed at step %d: %s", step, err)
                answer = "Sorry, I hit an error contacting the model. Please try again."
                session.add({"role": "assistant", "content": answer})
                return AgentResult(answer, session.session_id, step, "llm_error", thoughts)

            message = resp.choices[0].message
            parsed = parse_message(message)
            session.add(_assistant_dict(message))
            if parsed.thought:
                thoughts.append(parsed.thought)

            # Direct answer -> done.
            if parsed.final_answer is not None:
                self._remember(session.user_id, user_input, parsed.final_answer)
                return AgentResult(parsed.final_answer, session.session_id, step, "final", thoughts)

            # Otherwise execute each requested tool call and feed results back.
            all_failed = True
            for call in parsed.tool_calls:
                result_text, ok = self._run_tool(call, ctx, session.session_id, step)
                all_failed = all_failed and not ok
                session.add({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result_text,
                })

            consecutive_failures = consecutive_failures + 1 if all_failed else 0
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                answer = ("I'm repeatedly failing to use my tools, so I'll stop here. "
                          "Please rephrase or try again later.")
                session.add({"role": "assistant", "content": answer})
                return AgentResult(answer, session.session_id, step, "tool_failures", thoughts)

        # Ran out of steps -> force a tool-free final answer.
        return self._force_finalize(session, thoughts, user_input, memory_block)

    # -- internals ----------------------------------------------------------
    def _run_tool(self, call: dict, ctx: ToolContext, session_id: str, step: int) -> tuple[str, bool]:
        """Execute one tool call. Returns (result_json_text, ok)."""
        name, args = call["name"], call["arguments"]
        started = time.time()

        def finish(payload: dict, ok: bool, error: str = "") -> tuple[str, bool]:
            text = json.dumps(payload, ensure_ascii=False, default=str)
            if self.tracer is not None:
                self.tracer.record(
                    session_id=session_id, step=step, tool=name, args=args, ok=ok,
                    latency_ms=(time.time() - started) * 1000, result=text, error=error,
                )
            return text, ok

        tool = self.registry.get(name)
        if tool is None:
            return finish({"error": f"unknown tool '{name}'"}, False, "unknown tool")
        if args.get(PARSE_ERROR_KEY):
            return finish(
                {"error": "could not parse tool arguments as JSON", "raw": args.get(RAW_KEY, "")},
                False, "bad arguments",
            )
        try:
            result = tool.func(ctx, **args)
            payload = result if isinstance(result, dict) else {"result": result}
            return finish(payload, True)
        except Exception as err:  # tool raised -> report back so the model can recover
            logger.warning("tool '%s' failed: %s", name, err)
            return finish({"error": f"{type(err).__name__}: {err}"}, False, str(err))

    def _force_finalize(self, session: Any, thoughts: list[str], user_input: str,
                        memory_block: str | None) -> AgentResult:
        messages = self.context.build(session, memory_block=memory_block)
        messages.append({
            "role": "system",
            "content": "You have reached the tool-use limit for this turn. "
                       "Give your best final answer now using what you already know. "
                       "Do not call any tools.",
        })
        try:
            resp = self.llm.chat(messages)  # no tools offered
            answer = (resp.choices[0].message.content or "").strip() or \
                "I couldn't fully complete that within the step limit."
        except Exception as err:
            logger.error("force-finalize LLM call failed: %s", err)
            answer = "I couldn't complete that within the step limit."
        session.add({"role": "assistant", "content": answer})
        self._remember(session.user_id, user_input, answer)
        return AgentResult(answer, session.session_id, self.max_steps, "max_steps", thoughts)

    # -- memory helpers (no-op unless a backend is configured) --------------
    def _recall(self, user_input: str, user_id: str) -> str | None:
        try:
            return self.memory.recall(user_input, user_id=user_id)
        except Exception as err:  # memory must never break a turn
            logger.warning("memory recall failed: %s", err)
            return None

    def _remember(self, user_id: str, user_input: str, answer: str) -> None:
        try:
            self.memory.record(user_id=user_id, user_text=user_input,
                               assistant_text=answer)
        except Exception as err:
            logger.warning("memory record failed: %s", err)


def _assistant_dict(message: Any) -> dict[str, Any]:
    """Convert an OpenAI-style assistant message to a re-sendable history dict."""
    out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return out


def build_agent(*, settings: Any = None, tracer: Any = None, memory: Any = None,
                use_memory: bool = False, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Agent:
    """Wire up a real agent against the configured OpenAI-compatible endpoint.

    Pass ``use_memory=True`` (or an explicit ``memory=`` backend) to attach the
    AutoMemory long-term memory layer.
    """
    from .config import Settings
    from .llm_client import LLMClient
    from .trace import Tracer

    settings = settings or Settings.from_env()
    llm = LLMClient(settings)
    context = ContextManager(
        llm,
        system_prompt=system_prompt,
        max_history_messages=settings.max_history_messages,
        keep_recent_messages=settings.keep_recent_messages,
    )
    if memory is None and use_memory:
        from .memory import AutoMemoryBackend
        memory = AutoMemoryBackend()
    return Agent(
        llm=llm,
        context=context,
        tracer=tracer or Tracer(),
        memory=memory,
        max_steps=settings.max_steps,
        system_prompt=system_prompt,
    )
