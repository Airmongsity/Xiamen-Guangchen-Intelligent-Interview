"""Optional long-term memory backend for the agent.

The runtime's ``context_manager`` handles *within-session* working memory
(recent turns + a rolling summary). This module adds *cross-session, long-term*
memory as a pluggable backend, with an adapter around the sibling **AutoMemory**
project (hybrid retrieval + forgetting curve + outcome feedback).

Recall timing & placement (the two things the task's README must explain):
  * WHEN: recall runs once per user turn, before the LLM loop; recording runs
    once after the final answer.
  * WHERE: the recalled block is injected as a ``system`` message placed AFTER
    the base prompt/summary and BEFORE the current turn (see context_manager).

Scope: memories are namespaced by ``user_id`` (like todos), so personal facts a
user shares in one window are recallable from any of their windows — matching the
"the agent gets to know *you* over time" goal. Conversation *context* stays
per-session; only long-term memory and todos are user-scoped.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger("mini_agent.memory")


@runtime_checkable
class MemoryBackend(Protocol):
    def recall(self, query: str, *, user_id: str) -> str | None:
        """Return an injectable memory block, or None if nothing relevant."""

    def record(self, *, user_id: str, user_text: str, assistant_text: str) -> None:
        """Persist the just-finished exchange for future recall."""

    def report(self, *, user_id: str, quality: float) -> None:
        """Give feedback on the most recent recall for this user (0..1)."""


class NullMemory:
    """No-op backend (the default) so the runtime stays self-contained."""

    def recall(self, query: str, *, user_id: str) -> str | None:
        return None

    def record(self, *, user_id: str, user_text: str, assistant_text: str) -> None:
        pass

    def report(self, *, user_id: str, quality: float) -> None:
        pass


class AutoMemoryBackend:
    """Adapter over the AutoMemory library.

    Long-term memory is namespaced by ``user_id`` (like todos), so personal facts
    learned in one window are recallable from any of that user's windows.

    Parameters
    ----------
    db_path:      SQLite file for the memory store.
    record_infer: if True (default), each recorded turn is run through
                  AutoMemory's LLM extraction so facts are immediately
                  recallable; set False to store raw turns verbatim (cheaper).
    """

    def __init__(self, *, db_path: str = "mini_agent_memory.db",
                 record_infer: bool = True, am=None):
        self.record_infer = record_infer
        self._last_retrieval: dict[str, str] = {}  # user_id -> retrieval_id
        self.am = am if am is not None else _build_automemory(db_path)

    def recall(self, query: str, *, user_id: str) -> str | None:
        result = self.am.recall(query, user_id=user_id)
        self._last_retrieval[user_id] = result.retrieval_id
        if not (result.long_term or result.short_term):
            return None
        return result.to_prompt()

    def record(self, *, user_id: str, user_text: str, assistant_text: str) -> None:
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        self.am.add(messages, user_id=user_id, infer=self.record_infer)

    def report(self, *, user_id: str, quality: float) -> None:
        retrieval_id = self._last_retrieval.get(user_id)
        if retrieval_id:
            self.am.report_outcome(retrieval_id=retrieval_id, quality=quality)


def _build_automemory(db_path: str):
    """Import AutoMemory (installed, or from the sibling ./AutoMemory dir) and
    build an instance, bridging the .env key-name mismatch.

    AutoMemory expects ``SILICONFLOW_API_KEY``; this repo's .env names it
    ``EMBEDDING_API_KEY`` (same SiliconFlow key), so we pass it through.
    """
    try:
        from automem import AutoMemConfig, AutoMemory
    except ImportError:
        sibling = Path(__file__).resolve().parents[1] / "AutoMemory"
        if sibling.is_dir():
            import sys
            sys.path.insert(0, str(sibling))
            from automem import AutoMemConfig, AutoMemory
        else:
            raise

    from .config import load_env
    load_env()  # ensure DEEPSEEK_API_KEY / EMBEDDING_API_KEY are in the environment

    siliconflow_key = (os.environ.get("SILICONFLOW_API_KEY")
                       or os.environ.get("EMBEDDING_API_KEY", ""))
    cfg = AutoMemConfig.from_env(
        env_file=None,  # env already loaded above
        db_path=db_path,
        siliconflow_api_key=siliconflow_key,
    )
    if not siliconflow_key:
        logger.warning("no SiliconFlow key found; AutoMemory embedding/rerank will fail")
    return AutoMemory(cfg)
