"""Session state and isolation.

Two different scopes, on purpose:

* **Conversation** state (chat history + rolling summary) is per **session**
  (one browser window / conversation), keyed by ``session_id``. Two windows are
  independent conversations and never bleed into each other.
* **Todos** are per **user**, NOT per session. A todo is a reminder to the
  *person*, so it must reach them from any of their windows. All sessions of the
  same ``user_id`` share one todo list.

So for the task's scenario (one user A, two windows):

    user A, window 1: check weather + add todo "bring umbrella"
    user A, window 2: write weekly report + add todo "finish report"

the two *conversations* stay isolated, while asking "what are my todos?" in
either window shows both. The :class:`SessionManager` is thread-safe.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

DEFAULT_USER = "default"


@dataclass
class Session:
    session_id: str
    user_id: str = DEFAULT_USER
    # OpenAI-format message dicts, EXCLUDING the system prompt (rebuilt each turn).
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Rolling compressed summary of older, evicted turns (see context_manager).
    summary: str = ""
    # User-scoped tool state. When a session is created via SessionManager this is
    # rebound to the shared per-user list, so it is NOT session-private; the
    # default_factory only matters for standalone Session() construction in tests.
    todos: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._todos_by_user: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None = None,
                      user_id: str = DEFAULT_USER) -> Session:
        with self._lock:
            if session_id is None:
                session_id = uuid.uuid4().hex[:8]
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(session_id=session_id, user_id=user_id)
                # Point the session's todo list at the shared per-user list, so
                # every window of the same user reads/writes the same reminders.
                session.todos = self._todos_by_user.setdefault(user_id, [])
                self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def todos_for(self, user_id: str = DEFAULT_USER) -> list[str]:
        """The shared todo list for a user (across all their sessions)."""
        return self._todos_by_user.setdefault(user_id, [])

    def list_ids(self) -> list[str]:
        return list(self._sessions)
