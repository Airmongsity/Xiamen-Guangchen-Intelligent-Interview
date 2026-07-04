"""search tool (mocked).

Returns deterministic canned results so the whole system is testable offline and
**time-invariant** (no live web, no clock). Swapping in a real search API later
means replacing only ``_lookup`` -- the schema and registration stay identical.
"""

from __future__ import annotations

from .registry import Tool, ToolContext

# Keyword -> canned hits. Deterministic on purpose.
_CANNED: dict[str, list[dict]] = {
    "agent": [
        {"title": "What is an LLM agent?",
         "snippet": "An agent wraps an LLM in a loop that observes, decides, and calls tools."},
        {"title": "ReAct: reasoning + acting",
         "snippet": "Interleaving thought and tool calls improves multi-step task success."},
    ],
    "python": [
        {"title": "Python official docs",
         "snippet": "Python is a high-level, general-purpose programming language."},
    ],
    "weather": [
        {"title": "How weather forecasting works",
         "snippet": "Forecasts combine numerical models with observational data."},
    ],
}


def _lookup(query: str, top_k: int) -> list[dict]:
    q = query.lower()
    for keyword, hits in _CANNED.items():
        if keyword in q:
            return hits[:top_k]
    # Generic deterministic fallback.
    return [{
        "title": f"Result for: {query}",
        "snippet": f"(mock) No curated entry for '{query}'; returning a placeholder result.",
    }][:top_k]


def search(ctx: ToolContext, query: str, top_k: int = 3) -> dict:
    return {"query": query, "results": _lookup(query, max(1, int(top_k)))}


SEARCH = Tool(
    name="search",
    description=(
        "Search the web for information you do not know. Returns a list of "
        "titled snippets. Use when the user asks about external facts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "top_k": {"type": "integer", "description": "How many results to return.",
                      "default": 3},
        },
        "required": ["query"],
    },
    func=search,
)
