"""Built-in tools and a factory that registers them."""

from __future__ import annotations

from .calculator import CALCULATOR
from .registry import Tool, ToolContext, ToolRegistry
from .search import SEARCH
from .todo import TODO
from .weather import WEATHER

__all__ = [
    "Tool", "ToolContext", "ToolRegistry",
    "CALCULATOR", "SEARCH", "TODO", "WEATHER",
    "default_registry",
]


def default_registry() -> ToolRegistry:
    """A registry preloaded with the four built-in tools."""
    reg = ToolRegistry()
    for tool in (CALCULATOR, SEARCH, WEATHER, TODO):
        reg.register(tool)
    return reg
