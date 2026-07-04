"""MCP server exposing AutoMemory to agents (Claude Code, etc).

Run: python -m automem.mcp_server [--transport stdio|streamable-http] [--db PATH]
Requires DEEPSEEK_API_KEY / SILICONFLOW_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import argparse
import json

from fastmcp import FastMCP

from .config import AutoMemConfig
from .memory import AutoMemory

mcp = FastMCP("automem")

_instance: AutoMemory | None = None


def _am() -> AutoMemory:
    global _instance
    if _instance is None:
        _instance = AutoMemory(AutoMemConfig.from_env())
    return _instance


@mcp.tool
def remember(content: str, kind: str = "experience", importance: float | None = None) -> str:
    """Store a self-authored long-term memory. Use this when you learn a durable
    lesson, a stable user preference, or a conclusion worth keeping across
    sessions — not for transient conversation details. kind is one of
    'experience' (a lesson/method that worked or failed), 'fact', or 'summary'.
    importance is 0-1 (defaults to 0.75 for self-authored memories)."""
    mem = _am().remember(content, kind=kind, importance=importance)
    return json.dumps(
        {"id": mem.id, "content": mem.content, "kind": mem.memory_kind,
         "importance": mem.importance},
        ensure_ascii=False,
    )


@mcp.tool
def recall(query: str, top_k: int = 8, include_short_term: bool = True) -> str:
    """Retrieve memories relevant to the current task or question. Returns recent
    short-term context plus the top-k long-term memories ranked by relevance,
    recency, importance, and proven usefulness. The result includes a
    retrieval_id — after you finish the task, call report_outcome with it so the
    memory system learns which memories actually help."""
    result = _am().recall(query, top_k=top_k, include_short_term=include_short_term)
    return result.to_prompt()


@mcp.tool
def report_outcome(retrieval_id: str, quality: float) -> str:
    """Report how useful a recall() result turned out to be. quality is 0-1:
    1.0 = the memories clearly improved your answer, 0.5 = no effect,
    0.0 = the memories were misleading or harmful. This feedback raises or
    lowers each retrieved memory's utility score, which affects future ranking
    and how fast it is forgotten. Call this once per retrieval_id."""
    out = _am().report_outcome(retrieval_id=retrieval_id, quality=quality)
    return json.dumps(out, ensure_ascii=False)


@mcp.tool
def forget(memory_id: str) -> str:
    """Soft-delete a memory that is wrong or obsolete. It stops being retrieved
    but is kept on disk for audit."""
    _am().forget(memory_id)
    return json.dumps({"forgotten": memory_id})


@mcp.tool
def add_conversation(messages_json: str, user_id: str = "default") -> str:
    """Feed a finished conversation into the extraction pipeline. messages_json
    is a JSON array of {"role": "user"|"assistant", "content": str}. The LLM
    extracts durable facts, deduplicates them against existing memories, and
    stores the result. Use this at the end of a session, not per message."""
    messages = json.loads(messages_json)
    events = _am().add(messages, user_id=user_id)
    return json.dumps(
        [
            {"event": e.event,
             "memory": e.memory.content if e.memory else None,
             "id": e.memory.id if e.memory else None}
            for e in events
        ],
        ensure_ascii=False,
    )


@mcp.tool
def link_memories(src_id: str, dst_id: str, kind: str = "related") -> str:
    """Create an explicit logic-chain link between two memories. Linked memories
    boost each other during retrieval (activation spreading). kind is 'related',
    'derived_from', or 'contradicts'."""
    _am().link(src_id, dst_id, kind=kind)
    return json.dumps({"linked": [src_id, dst_id], "kind": kind})


@mcp.tool
def memory_stats() -> str:
    """Inspect the memory store: counts, average importance/utility, link count,
    retrievals awaiting feedback, and which vector backend is active."""
    return json.dumps(_am().stats(), ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoMemory MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    parser.add_argument("--db", default=None, help="override AUTOMEM_DB_PATH")
    parser.add_argument("--port", type=int, default=8848)
    args = parser.parse_args()

    if args.db:
        import os

        os.environ["AUTOMEM_DB_PATH"] = args.db

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport="streamable-http", port=args.port)


if __name__ == "__main__":
    main()
