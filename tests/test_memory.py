"""Memory-backend wiring, tested with a fake backend (no network)."""

from mini_agent.memory import MemoryBackend, NullMemory
from mini_agent.runtime import Agent
from mini_agent.trace import Tracer


class FakeMemory:
    """Records calls and returns a canned recall block, keyed by user_id."""

    def __init__(self, block_for: dict[str, str] | None = None):
        self.block_for = block_for or {}
        self.recall_calls: list[tuple[str, str]] = []
        self.record_calls: list[tuple[str, str, str]] = []

    def recall(self, query, *, user_id):
        self.recall_calls.append((query, user_id))
        return self.block_for.get(user_id)

    def record(self, *, user_id, user_text, assistant_text):
        self.record_calls.append((user_id, user_text, assistant_text))

    def report(self, *, user_id, quality):
        pass


def test_null_memory_is_the_default(fake_llm_factory):
    agent = Agent(llm=fake_llm_factory([{"content": "hi"}]), tracer=Tracer())
    assert isinstance(agent.memory, NullMemory)
    # NullMemory satisfies the protocol.
    assert isinstance(NullMemory(), MemoryBackend)


def test_recall_block_is_injected_into_context(fake_llm_factory):
    mem = FakeMemory(block_for={"default": "<memory>user likes Python</memory>"})
    llm = fake_llm_factory([{"content": "You like Python."}])
    agent = Agent(llm=llm, tracer=Tracer(), memory=mem)

    agent.chat("what do I like?", session_id="s")

    # recall was queried by USER (not session)...
    assert mem.recall_calls == [("what do I like?", "default")]
    # ...and the recalled block reached the LLM as a system message.
    sent = llm.calls[0]["messages"]
    assert any(m["role"] == "system" and "likes Python" in m["content"] for m in sent)


def test_memory_is_user_scoped_across_sessions(fake_llm_factory):
    mem = FakeMemory()
    agent = Agent(llm=fake_llm_factory([{"content": "a"}, {"content": "b"}]),
                  tracer=Tracer(), memory=mem)
    # Two different windows (sessions) of the same user hit the same memory namespace.
    agent.chat("hi", session_id="w1", user_id="A")
    agent.chat("hi", session_id="w2", user_id="A")
    assert [c[1] for c in mem.recall_calls] == ["A", "A"]
    assert [c[0] for c in mem.record_calls] == ["A", "A"]


def test_exchange_is_recorded_after_answer(fake_llm_factory):
    mem = FakeMemory()
    agent = Agent(llm=fake_llm_factory([{"content": "The answer is 4."}]), tracer=Tracer(),
                  memory=mem)
    agent.chat("2+2?", session_id="s")
    assert mem.record_calls == [("default", "2+2?", "The answer is 4.")]


def test_memory_failure_does_not_break_turn(fake_llm_factory):
    class BoomMemory(FakeMemory):
        def recall(self, query, *, user_id):
            raise RuntimeError("backend down")

    agent = Agent(llm=fake_llm_factory([{"content": "still works"}]), tracer=Tracer(),
                  memory=BoomMemory())
    r = agent.chat("hello", session_id="s")
    assert r.answer == "still works" and r.stop_reason == "final"
