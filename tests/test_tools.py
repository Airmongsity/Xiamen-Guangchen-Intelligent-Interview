import pytest

from mini_agent.session import Session
from mini_agent.tools.registry import ToolContext
from mini_agent.tools.search import search
from mini_agent.tools.todo import todo
from mini_agent.tools.weather import get_weather


def _ctx(session=None):
    return ToolContext(session=session or Session(session_id="t"))


def test_search_is_deterministic():
    # Same query -> same results (time-invariant, offline).
    a = search(_ctx(), query="what is an agent")
    b = search(_ctx(), query="what is an agent")
    assert a == b
    assert a["results"]  # non-empty


def test_weather_is_deterministic_and_clockfree():
    a = get_weather(_ctx(), city="Xiamen")
    b = get_weather(_ctx(), city="xiamen")
    # Lookup is case-insensitive; only the echoed 'city' preserves the caller's spelling.
    assert a["condition"] == b["condition"] == "sunny"
    assert a["temp_c"] == b["temp_c"]


def test_todo_is_user_scoped_not_session_scoped():
    from mini_agent.session import SessionManager

    mgr = SessionManager()
    # Same user A, two windows (sessions).
    w1 = mgr.get_or_create("w1", user_id="A")
    w2 = mgr.get_or_create("w2", user_id="A")
    # A different user B.
    wb = mgr.get_or_create("wb", user_id="B")

    todo(_ctx(w1), action="add", content="bring umbrella")     # added in window 1
    todo(_ctx(wb), action="add", content="user B's private item")

    # A todo is a reminder to the *user*: window 2 sees window 1's todo.
    assert todo(_ctx(w2), action="list")["todos"] == ["bring umbrella"]
    # But different users are isolated.
    assert todo(_ctx(wb), action="list")["todos"] == ["user B's private item"]


def test_todo_done_and_validation():
    s = Session(session_id="w")
    todo(_ctx(s), action="add", content="task A")
    out = todo(_ctx(s), action="done", index=0)
    assert out["removed"] == "task A"
    assert out["todos"] == []
    with pytest.raises(ValueError):
        todo(_ctx(s), action="done", index=5)
    with pytest.raises(ValueError):
        todo(_ctx(s), action="bogus")
