"""todo tool: the user's to-do list.

Todos are **user-scoped**, not conversation-scoped: a reminder belongs to the
person, so it is shared across all of that user's windows (see ``session.py`` —
``ctx.session.todos`` is bound to the shared per-user list). Add a todo in one
window and it is visible when the user asks in another.
"""

from __future__ import annotations

from .registry import Tool, ToolContext


def todo(ctx: ToolContext, action: str, content: str = "", index: int | None = None) -> dict:
    if ctx.session is None:
        raise RuntimeError("todo tool requires an active session")
    items: list[str] = ctx.session.todos

    if action == "add":
        if not content.strip():
            raise ValueError("'content' is required to add a todo")
        items.append(content.strip())
        return {"ok": True, "action": "add", "todos": list(items)}

    if action == "list":
        return {"action": "list", "todos": list(items)}

    if action == "done":
        if index is None or not (0 <= index < len(items)):
            raise ValueError(f"index out of range: {index} (have {len(items)} todos)")
        removed = items.pop(index)
        return {"ok": True, "action": "done", "removed": removed, "todos": list(items)}

    raise ValueError(f"unknown action '{action}' (expected add | list | done)")


TODO = Tool(
    name="todo",
    description=(
        "Manage the user's to-do list (shared across all their windows). "
        "action='add' with content to add an item; action='list' to show all; "
        "action='done' with index (0-based) to complete/remove one."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "done"]},
            "content": {"type": "string", "description": "Todo text (for action='add')."},
            "index": {"type": "integer", "description": "0-based index (for action='done')."},
        },
        "required": ["action"],
    },
    func=todo,
)
