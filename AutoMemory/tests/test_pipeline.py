"""Write-pipeline + feedback integration tests with fake providers; no API keys."""

import numpy as np
import pytest

from automem.config import AutoMemConfig
from automem.memory import AutoMemory

DIM = 8


class FakeEmbedder:
    """Deterministic bag-of-words-ish embedding over a tiny vocabulary."""

    dim = DIM
    VOCAB = ["python", "cooking", "weather", "pizza", "cheese", "chicken", "milan", "rainy"]

    def embed(self, text: str):
        t = text.lower()
        vec = np.full(DIM, 0.05)
        for i, w in enumerate(self.VOCAB):
            if w in t:
                vec[i] += 1.0
        return (vec / np.linalg.norm(vec)).tolist()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class FakeLLM:
    """Returns queued JSON responses; records prompts for assertions."""

    def __init__(self):
        self.queue = []
        self.calls = []

    def complete_json(self, system, user, **kw):
        self.calls.append((system, user))
        if not self.queue:
            raise AssertionError("FakeLLM queue exhausted")
        return self.queue.pop(0)

    def complete(self, system, user, **kw):
        return ""


@pytest.fixture
def am(tmp_path):
    cfg = AutoMemConfig(db_path=str(tmp_path / "t.db"), embed_dim=DIM)
    llm = FakeLLM()
    mem = AutoMemory(cfg, embedder=FakeEmbedder(), llm=llm, reranker=None)
    mem._fake_llm = llm
    yield mem
    mem.close()


def test_add_extracts_and_stores(am):
    am._fake_llm.queue.append(
        {"memories": [
            {"content": "User loves cheese pizza", "kind": "fact", "importance": 0.6},
            {"content": "User lives in Milan", "kind": "fact", "importance": 0.8},
        ]}
    )
    events = am.add([{"role": "user", "content": "I love cheese pizza! I live in Milan."}])
    assert [e.event for e in events] == ["ADD", "ADD"]
    assert am.stats()["memories"] == 2
    assert events[1].memory.importance == 0.8


def test_add_update_decision_merges(am):
    am._fake_llm.queue.append(
        {"memories": [{"content": "User loves cheese pizza", "kind": "fact", "importance": 0.5}]}
    )
    [first] = am.add([{"role": "user", "content": "I love cheese pizza"}])

    # second fact is similar (cheese+pizza tokens) -> reconcile path -> UPDATE
    am._fake_llm.queue.append(
        {"memories": [{"content": "User loves cheese and chicken pizza", "kind": "fact", "importance": 0.5}]}
    )
    am._fake_llm.queue.append(
        {"op": "UPDATE", "id": first.memory.id, "text": "User loves cheese and chicken pizza"}
    )
    [event] = am.add([{"role": "user", "content": "Actually chicken pizza too"}])
    assert event.event == "UPDATE"
    assert event.previous_content == "User loves cheese pizza"
    # the existing row keeps its id and becomes the merged, current truth
    current = am.get(first.memory.id)
    assert current.content == "User loves cheese and chicken pizza"
    assert current.valid_to is None
    # UPDATE-archiving: the pre-merge version is snapshotted as bi-temporal
    # history (a new row, valid_to set) so the old wording stays answerable
    assert am.stats()["memories"] == 2
    hist = [
        m for m in am.list_memories()
        if m.id != current.id and m.content == "User loves cheese pizza"
    ]
    assert len(hist) == 1
    assert hist[0].valid_to is not None
    assert hist[0].superseded_by == current.id


def test_add_delete_decision_supersedes_and_links_contradiction(am):
    am._fake_llm.queue.append(
        {"memories": [{"content": "User loves cheese pizza", "kind": "fact", "importance": 0.5}]}
    )
    [first] = am.add([{"role": "user", "content": "I love cheese pizza"}])

    am._fake_llm.queue.append(
        {"memories": [{"content": "User no longer eats cheese pizza", "kind": "fact", "importance": 0.6}]}
    )
    am._fake_llm.queue.append({"op": "DELETE", "id": first.memory.id})
    [event] = am.add([{"role": "user", "content": "I stopped eating cheese pizza"}])
    assert event.event == "DELETE"
    new_mem = event.memory

    # bi-temporal invalidation: the old fact is NOT deleted; it is marked
    # historically valid (valid_to set, superseded_by -> new) and stays retrievable.
    old = am.get(first.memory.id)
    assert not old.is_deleted
    assert old.valid_to is not None
    assert old.superseded_by == new_mem.id
    assert new_mem.valid_to is None  # the new fact is current truth
    kinds = {k for _, k, _ in am.store.neighbors(new_mem.id)}
    assert "contradicts" in kinds


def test_superseded_fact_goes_to_history_quota_not_main_budget(am):
    am.config.history_separate_quota = True  # this test exercises the separate quota
    am._fake_llm.queue.append(
        {"memories": [{"content": "User works at Acme as an analyst", "kind": "fact", "importance": 0.5}]}
    )
    [first] = am.add([{"role": "user", "content": "I work at Acme as an analyst"}])
    am._fake_llm.queue.append(
        {"memories": [{"content": "User works at Globex as a manager", "kind": "fact", "importance": 0.5}]}
    )
    am._fake_llm.queue.append({"op": "DELETE", "id": first.memory.id})
    am.add([{"role": "user", "content": "I now work at Globex as a manager"}])

    # the superseded fact never competes for the current top_k budget...
    result = am.recall("where does the user work", include_short_term=False, history_mode="on")
    main_ids = [sm.memory.id for sm in result.long_term]
    assert first.memory.id not in main_ids
    assert all(sm.memory.valid_to is None for sm in result.long_term)
    # ...but it is preserved and retrievable via the separate history quota
    hist_ids = [sm.memory.id for sm in result.history]
    assert first.memory.id in hist_ids


def test_add_stores_source_and_recall_expands_evidence(am):
    am._fake_llm.queue.append(
        {"memories": [{"content": "User loves cheese pizza", "kind": "fact", "importance": 0.6}]}
    )
    [event] = am.add(
        [{"role": "user", "content": "Honestly the best cheese pizza is from Tony's on 5th"}]
    )
    # the extracted fact carries a pointer to the verbatim passage it came from
    assert event.memory.source_id is not None
    raw = am.store.get_sources([event.memory.source_id])
    assert "Tony's on 5th" in raw[event.memory.source_id]

    # recall expands that passage so exact wording is available to the answerer
    result = am.recall("cheese pizza", include_short_term=False)
    assert event.memory.source_id in result.sources
    assert "Tony's on 5th" in result.sources[event.memory.source_id]
    assert "Tony's on 5th" in result.to_prompt()


def test_slot_path_catches_low_similarity_value_change(am):
    am._fake_llm.queue.append(
        {"memories": [{"content": "User lives in Milan", "kind": "fact",
                       "importance": 0.6, "slot": "home_city"}]}
    )
    [first] = am.add([{"role": "user", "content": "I live in Milan"}])
    assert first.memory.slot == "home_city"

    # the new value barely embeds near the old wording (milan vs rainy tokens),
    # so it is below the reconcile floor — only the slot lookup surfaces the old
    # memory as a candidate, letting the LLM decide UPDATE.
    am._fake_llm.queue.append(
        {"memories": [{"content": "User now lives somewhere rainy", "kind": "fact",
                       "importance": 0.6, "slot": "home_city"}]}
    )
    am._fake_llm.queue.append(
        {"op": "UPDATE", "id": first.memory.id, "text": "User lives in a rainy city"}
    )
    [event] = am.add([{"role": "user", "content": "I moved, it's rainy here now"}])
    assert event.event == "UPDATE"  # reconcile ran => the slot candidate was found
    assert am.get(first.memory.id).content == "User lives in a rainy city"


def test_history_mode_gating(am):
    am.config.history_separate_quota = True  # the gate only applies to the separate quota
    am._fake_llm.queue.append(
        {"memories": [{"content": "User works at Acme", "kind": "fact", "importance": 0.5}]}
    )
    [first] = am.add([{"role": "user", "content": "I work at Acme"}])
    am._fake_llm.queue.append(
        {"memories": [{"content": "User works at Globex", "kind": "fact", "importance": 0.5}]}
    )
    am._fake_llm.queue.append({"op": "DELETE", "id": first.memory.id})
    am.add([{"role": "user", "content": "I now work at Globex"}])

    def hist_ids(mode, q="where does the user work"):
        r = am.recall(q, include_short_term=False, history_mode=mode)
        # the current fact is ALWAYS in the main budget regardless of mode
        assert all(sm.memory.valid_to is None for sm in r.long_term)
        return [sm.memory.id for sm in r.history]

    assert hist_ids("off") == []                       # never attach
    assert first.memory.id in hist_ids("on")           # always attach
    assert hist_ids("auto") == []                       # non-retrospective query -> off
    # retrospective language flips the auto gate on
    assert first.memory.id in hist_ids("auto", "where did the user work before")


def test_compete_mode_history_competes_in_main_pool(am):
    am._fake_llm.queue.append(
        {"memories": [{"content": "User works at Acme", "kind": "fact", "importance": 0.5}]}
    )
    [first] = am.add([{"role": "user", "content": "I work at Acme"}])
    am._fake_llm.queue.append(
        {"memories": [{"content": "User works at Globex", "kind": "fact", "importance": 0.5}]}
    )
    am._fake_llm.queue.append({"op": "DELETE", "id": first.memory.id})
    am.add([{"role": "user", "content": "I now work at Globex"}])

    am.config.history_separate_quota = False  # ablation: pre-refactor competing design
    r = am.recall("where does the user work", include_short_term=False, history_mode="on")
    assert r.history == []  # no separate quota in compete mode
    assert first.memory.id in [sm.memory.id for sm in r.long_term]  # competes in top_k


def test_readonly_store_is_structurally_safe(tmp_path):
    import sqlite3
    path = str(tmp_path / "ro.db")
    rw = AutoMemory(AutoMemConfig(db_path=path, embed_dim=DIM),
                    embedder=FakeEmbedder(), llm=FakeLLM(), reranker=None)
    [m] = rw.add_batch(["User loves cheese pizza"])
    rw.close()

    ro = AutoMemory(AutoMemConfig(db_path=path, embed_dim=DIM),
                    embedder=FakeEmbedder(), llm=FakeLLM(), reranker=None, readonly=True)
    try:
        # reads work and do not mutate
        r = ro.recall("cheese pizza", include_short_term=False,
                      as_of="2023-06-01T00:00:00+00:00")
        assert any(sm.memory.id == m.id for sm in r.long_term)
        assert ro.get(m.id).access_count == 0
        # clock is mandatory in readonly (eval) mode
        with pytest.raises(ValueError):
            ro.recall("cheese pizza", include_short_term=False, as_of=None)
        # a forgotten touch=True is forced harmless (no mutation), not a crash
        ro.recall("cheese pizza", include_short_term=False,
                  as_of="2023-06-01T00:00:00+00:00", touch=True)
        assert ro.get(m.id).access_count == 0
        # but a write that BYPASSES recall hits the OS-level read-only wall
        with pytest.raises(sqlite3.OperationalError):
            ro.store.touch_accessed([m.id])
    finally:
        ro.close()


def test_recall_readonly_does_not_mutate_state(am):
    [m] = am.add_batch(["User loves cheese pizza"])
    # read-only recall (evaluation mode): must not reinforce or log
    r = am.recall("cheese pizza", include_short_term=False, touch=False, log=False)
    assert r.retrieval_id == "r-nolog"
    assert am.get(m.id).access_count == 0
    assert am.get(m.id).last_accessed is None
    # default recall (live mode): reinforces access + writes a retrieval row
    r2 = am.recall("cheese pizza", include_short_term=False)
    assert am.get(m.id).access_count >= 1
    assert am.get(m.id).last_accessed is not None
    assert r2.retrieval_id != "r-nolog"


def test_recall_as_of_scores_retention_at_query_time(am):
    old = am.add_batch(
        [{"content": "User loves cheese pizza", "created_at": "2023-01-01T00:00:00+00:00"}]
    )[0]
    new = am.add_batch(
        [{"content": "User loves cheese pizza places", "created_at": "2023-06-01T00:00:00+00:00"}]
    )[0]
    # retention is scored as of the query time, so the fresher fact decays less
    r = am.recall(
        "cheese pizza", include_short_term=False,
        as_of="2023-06-02T00:00:00+00:00", touch=False, log=False,
    )
    sm = {s.memory.id: s for s in r.long_term}
    assert sm[new.id].retention > sm[old.id].retention


def test_reconcile_hallucinated_id_falls_back_to_add(am):
    am._fake_llm.queue.append(
        {"memories": [{"content": "User loves cheese pizza", "kind": "fact", "importance": 0.5}]}
    )
    am.add([{"role": "user", "content": "I love cheese pizza"}])
    am._fake_llm.queue.append(
        {"memories": [{"content": "User loves cheese pizza on Fridays", "kind": "fact", "importance": 0.5}]}
    )
    am._fake_llm.queue.append({"op": "UPDATE", "id": "nonexistent", "text": "x"})
    [event] = am.add([{"role": "user", "content": "cheese pizza on Fridays"}])
    assert event.event == "ADD"  # fail-safe


def test_remember_direct_and_merge(am):
    m1 = am.remember("Lesson: always run tests before committing")
    assert m1.source == "self"
    assert m1.importance == 0.75
    assert m1.memory_kind == "experience"
    # near-identical content merges instead of duplicating
    m2 = am.remember("Lesson: always run tests before committing!")
    assert m2.id == m1.id
    assert am.stats()["memories"] == 1


def test_add_batch_backfills_created_at(am):
    mems = am.add_batch(
        [
            {"content": "User visited Milan", "created_at": "2025-01-15T10:00:00+00:00"},
            "User likes rainy weather",
        ]
    )
    assert len(mems) == 2
    assert mems[0].created_at == "2025-01-15T10:00:00+00:00"
    assert mems[0].source == "import"


def test_auto_link_created_for_related(am):
    a = am.add_batch(["User loves cheese pizza"])[0]
    b = am.add_batch(["User loves chicken pizza"])[0]  # batch import doesn't link
    m = am.remember("User makes pizza at home every weekend", kind="fact")
    nbrs = {n for n, _, _ in am.store.neighbors(m.id)}
    assert nbrs & {a.id, b.id}  # moderately similar memories got linked


def test_observe_consolidate(am):
    cfg = am.config
    cfg.stm_max_items = 2
    for i in range(4):
        am.observe("user", f"message about python number {i}")
    # 2 oldest overflow; extractor called once for the session
    am._fake_llm.queue.append(
        {"memories": [{"content": "User talks about python a lot", "kind": "summary", "importance": 0.4}]}
    )
    events = am.consolidate()
    assert [e.event for e in events] == ["ADD"]
    assert am.consolidate() == []  # idempotent: overflow already consolidated
    window = am.store.get_stm_window(max_items=10)
    assert len(window) == 2


def test_feedback_roundtrip_via_recall(am):
    am.add_batch(["User loves cheese pizza", "User likes rainy weather"])
    result = am.recall("cheese pizza", include_short_term=False)
    assert result.long_term
    top = result.long_term[0].memory
    out = am.report_outcome(retrieval_id=result.retrieval_id, quality=1.0)
    assert out["updated"] >= 1
    updated = am.get(top.id)
    assert updated.utility == pytest.approx(0.3)  # eta * w0 * (1 - 0)
    assert updated.feedback_count == 1
    # idempotent second report
    out2 = am.report_outcome(retrieval_id=result.retrieval_id, quality=0.0)
    assert out2["updated"] == 0
    assert am.get(top.id).utility == pytest.approx(0.3)


def test_feedback_direct_memory_ids(am):
    [m] = am.add_batch(["User loves cheese pizza"])
    am.report_outcome(memory_ids=[m.id], quality=0.0)
    assert am.get(m.id).utility == pytest.approx(-0.3)


def test_feedback_validation(am):
    with pytest.raises(ValueError):
        am.report_outcome(quality=0.5)
    with pytest.raises(ValueError):
        am.report_outcome(retrieval_id="r-x", memory_ids=["m"], quality=0.5)
    with pytest.raises(ValueError):
        am.report_outcome(memory_ids=["m"], quality=1.5)
    with pytest.raises(KeyError):
        am.report_outcome(retrieval_id="r-missing", quality=0.5)


def test_maintain_forgets_stale_useless(am):
    old = am.add_batch(
        [{"content": "ancient trivia", "created_at": "2020-01-01T00:00:00+00:00", "importance": 0.0}]
    )[0]
    fresh = am.add_batch(["fresh fact"])[0]
    result = am.maintain()
    assert result["forgotten"] == 1
    assert am.get(old.id).is_deleted
    assert not am.get(fresh.id).is_deleted


def test_maintain_spares_high_utility(am):
    old = am.add_batch(
        [{"content": "ancient but proven useful", "created_at": "2020-01-01T00:00:00+00:00", "importance": 0.0}]
    )[0]
    am.report_outcome(memory_ids=[old.id], quality=1.0)
    assert am.maintain()["forgotten"] == 0
