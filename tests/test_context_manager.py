from mini_agent.context_manager import ContextManager
from mini_agent.session import Session


def _history():
    """10 messages ending on a user turn, with a tool exchange near the tail."""
    msgs = []
    for i in range(7):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"m{i}"})
    # A tool exchange at indices 7 (assistant+tool_calls) and 8 (tool result).
    msgs.append({"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c0", "type": "function",
                                 "function": {"name": "weather", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": "c0", "content": "{'temp': 20}"})
    msgs.append({"role": "user", "content": "and tomorrow?"})
    return msgs


def test_no_compression_when_short(fake_llm_factory):
    cm = ContextManager(fake_llm_factory([]), max_history_messages=20, keep_recent_messages=8)
    s = Session(session_id="x", messages=[{"role": "user", "content": "hi"}])
    assert cm.maybe_compress(s) is False
    assert s.summary == ""


def test_compression_summarizes_and_keeps_tail(fake_llm_factory):
    llm = fake_llm_factory([{"content": "SUMMARY OF OLD TURNS"}])
    cm = ContextManager(llm, max_history_messages=6, keep_recent_messages=3)
    s = Session(session_id="x", messages=_history())

    assert cm.maybe_compress(s) is True
    assert s.summary == "SUMMARY OF OLD TURNS"
    # Tail is kept verbatim and starts on a clean user boundary...
    assert s.messages[0]["role"] == "user"
    # ...and no orphan tool message leaks to the front of the kept tail.
    assert all(not (m["role"] == "tool") for m in s.messages[:1])


def test_build_injects_summary(fake_llm_factory):
    cm = ContextManager(fake_llm_factory([]))
    s = Session(session_id="x", summary="prior summary",
                messages=[{"role": "user", "content": "hello"}])
    built = cm.build(s)
    assert built[0]["role"] == "system"
    assert any("prior summary" in m["content"] for m in built if m["role"] == "system")
    assert built[-1] == {"role": "user", "content": "hello"}
