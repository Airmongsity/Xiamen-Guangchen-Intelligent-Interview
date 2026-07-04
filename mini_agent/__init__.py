"""mini_agent: a minimal, from-scratch LLM agent runtime.

Public surface:
    Agent, AgentResult, build_agent  -- the runtime and its factory
    ToolRegistry, Tool, ToolContext  -- the tool registration mechanism
    SessionManager, Session          -- session isolation
"""

from __future__ import annotations

from .memory import AutoMemoryBackend, MemoryBackend, NullMemory
from .runtime import Agent, AgentResult, build_agent
from .session import Session, SessionManager
from .tools import Tool, ToolContext, ToolRegistry, default_registry

__all__ = [
    "Agent", "AgentResult", "build_agent",
    "Tool", "ToolContext", "ToolRegistry", "default_registry",
    "Session", "SessionManager",
    "MemoryBackend", "NullMemory", "AutoMemoryBackend",
]
