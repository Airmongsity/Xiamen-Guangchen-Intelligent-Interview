"""Tiny live smoke: confirm the slot path + UPDATE-archiving fire end-to-end
against the real DeepSeek/SiliconFlow stack. Uses a throwaway db.

  python evaluation/longmemeval/smoke_slot.py
"""

from __future__ import annotations

import os
import tempfile

from common import make_memory

UID = "smoke"


def main() -> None:
    db = os.path.join(tempfile.gettempdir(), "automem_smoke_slot.db")
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass
    am = make_memory(db)

    print("== ingest session 1 (employer = Acme) ==")
    ev1 = am.add(
        [
            {"role": "user", "content": "I just started a new job at Acme Corp as a data analyst."},
            {"role": "assistant", "content": "Congrats on joining Acme as a data analyst!"},
        ],
        user_id=UID,
        conversation_date="2024-01-10",
        created_at="2024-01-10T09:00:00+00:00",
    )

    for e in ev1:
        print(f"   s1 event: {e.event} slot={getattr(e.memory,'slot',None)} :: {getattr(e.memory,'content',None)}")

    print("== ingest session 2 (employer changes -> Globex) ==")
    ev2 = am.add(
        [
            {"role": "user", "content": "Big news: I left Acme and now work at Globex Industries as a product manager."},
            {"role": "assistant", "content": "Exciting move to Globex as a PM!"},
        ],
        user_id=UID,
        conversation_date="2024-06-01",
        created_at="2024-06-01T09:00:00+00:00",
    )

    for e in ev2:
        print(f"   s2 event: {e.event} slot={getattr(e.memory,'slot',None)} :: {getattr(e.memory,'content',None)}")

    print("\n== all memories for slot=employer (history included) ==")
    for m in am.store.get_by_slot("employer", user_id=UID, current_only=False):
        flag = f" superseded@{m.valid_to[:10]}" if m.valid_to else " CURRENT"
        print(f"  [{m.id[:8]}]{flag} slot={m.slot} :: {m.content}")

    print("\n== full memory list ==")
    for m in am.list_memories(user_id=UID, limit=50):
        flag = f" superseded->{m.superseded_by[:8]}" if m.valid_to else " current"
        print(f"  [{m.id[:8]}]{flag} slot={m.slot} :: {m.content}")

    def show(title, q, mode):
        res = am.recall(q, user_id=UID, include_short_term=False, history_mode=mode)
        print(f"\n== {title} :: q={q!r} mode={mode} ==")
        print("  [current top_k]")
        for sm in res.long_term:
            print(f"    {sm.final_score:.3f} :: {sm.memory.content}")
        print(f"  [history quota: {len(res.history)}]")
        for sm in res.history:
            print(f"    {sm.final_score:.3f} (until {sm.memory.valid_to[:10]}) :: {sm.memory.content}")
        assert all(sm.memory.valid_to is None for sm in res.long_term), "current budget leaked history"
        return res

    # current-only question: auto gate should NOT attach history
    r_now = show("present-tense (auto)", "Where does the user work now?", "auto")
    assert not r_now.history, "auto attached history for a present-tense query"
    # retrospective question: auto gate SHOULD attach history
    r_past = show("retrospective (auto)", "Where did the user work before?", "auto")
    assert r_past.history, "auto failed to attach history for a retrospective query"
    show("forced on", "Where does the user work now?", "on")

    am.close()
    print("\nOK")


if __name__ == "__main__":
    main()
