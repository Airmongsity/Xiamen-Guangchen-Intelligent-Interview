"""Tool registration mechanism.

Every tool carries a **name**, a **description**, and a **JSON Schema** for its
parameters -- exactly the three things an OpenAI-compatible LLM needs to decide,
on its own, whether and how to call it. The registry turns registered tools into
the ``tools=[...]`` payload the model sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolContext:
    """Per-invocation context handed to every tool.

    Holds the active :class:`~mini_agent.session.Session` so that session-scoped
    tools (e.g. ``todo``) read and write the correct, isolated state. Stateless
    tools (``calculator``) simply ignore it.
    """

    session: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]                 # JSON Schema (object type)
    func: Callable[..., Any]                    # func(ctx: ToolContext, **kwargs)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """The ``tools`` payload passed to the LLM."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
