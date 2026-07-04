"""End-to-end loop tests using a scripted FakeLLM (no network, time-invariant)."""

from mini_agent.runtime import Agent
from mini_agent.trace import Tracer


def _agent(llm, **kw):
    return Agent(llm=llm, tracer=Tracer(), **kw)


def test_direct_answer_no_tool(fake_llm_factory):
    agent = _agent(fake_llm_factory([{"content": "Hi! How can I help?"}]))
    r = agent.chat("hello", session_id="s")
    assert r.answer == "Hi! How can I help?"
    assert r.steps == 1
    assert r.stop_reason == "final"


def test_single_tool_then_final(fake_llm_factory):
    llm = fake_llm_factory([
        {"tool_calls": [("calculator", {"expression": "2 + 2"})]},
        {"content": "The answer is 4."},
    ])
    agent = _agent(llm)
    r = agent.chat("what is 2+2?", session_id="s")
    assert r.answer == "The answer is 4."
    assert r.steps == 2 and r.stop_reason == "final"
    # The tool actually ran and was traced successfully.
    trace = agent.tracer.for_session("s")
    assert trace[0].tool == "calculator" and trace[0].ok is True


def test_multi_step_loop(fake_llm_factory):
    llm = fake_llm_factory([
        {"tool_calls": [("weather", {"city": "Xiamen"})]},
        {"tool_calls": [("todo", {"action": "add", "content": "bring umbrella"})]},
        {"content": "It's sunny in Xiamen and I've added your todo."},
    ])
    agent = _agent(llm)
    r = agent.chat("weather in Xiamen and remember to bring umbrella", session_id="s")
    assert r.steps == 3 and r.stop_reason == "final"
    assert agent.sessions.get("s").todos == ["bring umbrella"]


def test_todos_shared_across_sessions_of_same_user(fake_llm_factory):
    # Add a todo in window w1, then LIST it from window w2 (same default user).
    llm = fake_llm_factory([
        {"tool_calls": [("todo", {"action": "add", "content": "bring umbrella"})]},
        {"content": "added"},
        {"tool_calls": [("todo", {"action": "list"})]},
        {"content": "here is your list"},
    ])
    agent = _agent(llm)
    agent.chat("remember to bring an umbrella", session_id="w1")
    agent.chat("what's on my todo list?", session_id="w2")

    # The list call in w2 saw w1's todo (user-scoped, not session-scoped).
    list_trace = [e for e in agent.tracer.for_session("w2") if e.tool == "todo"][-1]
    assert "bring umbrella" in list_trace.result
    assert agent.sessions.todos_for("default") == ["bring umbrella"]


def test_session_isolation_through_agent(fake_llm_factory):
    llm = fake_llm_factory([{"content": "answer for w1"}, {"content": "answer for w2"}])
    agent = _agent(llm)
    agent.chat("hi", session_id="w1")
    agent.chat("hi", session_id="w2")
    w1, w2 = agent.sessions.get("w1"), agent.sessions.get("w2")
    assert w1.messages != w2.messages
    assert w1.messages[0]["content"] == "hi"


def test_max_steps_forces_finalize(fake_llm_factory):
    # Always asks for a (successful) tool -> never volunteers a final answer.
    llm = fake_llm_factory([{"tool_calls": [("calculator", {"expression": "1 + 1"})]}] * 5)
    agent = _agent(llm, max_steps=3)
    r = agent.chat("loop forever", session_id="s")
    assert r.stop_reason == "max_steps"
    assert r.steps == 3


def test_repeated_tool_failure_stops(fake_llm_factory):
    # 'a + 1' is not pure arithmetic -> calculator raises every time.
    llm = fake_llm_factory([{"tool_calls": [("calculator", {"expression": "a + 1"})]}] * 5)
    agent = _agent(llm)
    r = agent.chat("break the calculator", session_id="s")
    assert r.stop_reason == "tool_failures"
    assert r.steps == 3


def test_unknown_tool_is_reported_not_crashed(fake_llm_factory):
    llm = fake_llm_factory([
        {"tool_calls": [("nonexistent", {"x": 1})]},
        {"content": "Sorry, I could not do that."},
    ])
    agent = _agent(llm)
    r = agent.chat("use a missing tool", session_id="s")
    assert r.stop_reason == "final"
    assert agent.tracer.for_session("s")[0].ok is False
