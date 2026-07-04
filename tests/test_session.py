from mini_agent.session import SessionManager


def test_sessions_are_isolated():
    mgr = SessionManager()
    w1 = mgr.get_or_create("w1")
    w2 = mgr.get_or_create("w2")

    w1.add({"role": "user", "content": "hi from window 1"})
    w2.add({"role": "user", "content": "hi from window 2"})

    assert w1.messages != w2.messages
    assert len(w1.messages) == 1 and len(w2.messages) == 1
    assert w1 is mgr.get_or_create("w1")  # same object on re-fetch


def test_auto_id_when_none():
    mgr = SessionManager()
    s = mgr.get_or_create()
    assert s.session_id in mgr.list_ids()
