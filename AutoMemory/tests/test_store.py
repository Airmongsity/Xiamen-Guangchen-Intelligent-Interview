"""Storage-layer tests with fake embeddings; no API keys required."""

import numpy as np
import pytest

from automem.store import MemoryStore

DIM = 8


def fake_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "test.db"), embed_dim=DIM)
    yield s
    s.close()


def test_insert_and_get(store):
    mem = store.insert_memory(
        "用户最喜欢的语言是 Python", fake_vec(1), memory_kind="fact", importance=0.7
    )
    got = store.get_memory(mem.id)
    assert got is not None
    assert got.content == "用户最喜欢的语言是 Python"
    assert got.importance == 0.7
    assert got.access_count == 0
    assert not got.is_deleted


def test_knn_orders_by_similarity(store):
    base = fake_vec(42)
    near = base + 0.05 * fake_vec(7)
    far = fake_vec(99)
    m_near = store.insert_memory("near", near / np.linalg.norm(near))
    m_far = store.insert_memory("far", far)
    results = store.knn(base, k=2)
    assert results[0][0] == m_near.id
    assert results[0][1] > results[1][1]
    assert {r[0] for r in results} == {m_near.id, m_far.id}


def test_knn_excludes_deleted_and_other_users(store):
    v = fake_vec(1)
    kept = store.insert_memory("kept", v)
    deleted = store.insert_memory("deleted", v)
    other = store.insert_memory("other user", v, user_id="someone_else")
    store.soft_delete(deleted.id)
    ids = [r[0] for r in store.knn(v, k=10)]
    assert kept.id in ids
    assert deleted.id not in ids
    assert other.id not in ids


def test_fts_search(store):
    m1 = store.insert_memory("the user prefers concise Chinese answers", fake_vec(1))
    store.insert_memory("completely unrelated text about weather", fake_vec(2))
    results = store.fts_search("concise Chinese", k=5)
    assert results
    assert results[0][0] == m1.id
    assert results[0][1] > 0  # flipped bm25 is higher-better


def test_fts_chinese_trigram(store):
    m1 = store.insert_memory("用户喜欢简洁的中文回答", fake_vec(1))
    store.insert_memory("天气晴朗适合出门散步", fake_vec(2))
    results = store.fts_search("中文回答", k=5)
    assert results and results[0][0] == m1.id


def test_fts_malformed_query_returns_empty(store):
    store.insert_memory("hello world", fake_vec(1))
    assert store.fts_search('"" AND (', k=5) is not None  # no exception


def test_update_content_syncs_fts(store):
    m = store.insert_memory("old topic: gardening", fake_vec(1))
    store.update_memory_content(m.id, "new topic: astronomy", fake_vec(2))
    assert store.fts_search("gardening", k=5) == []
    hits = store.fts_search("astronomy", k=5)
    assert hits and hits[0][0] == m.id


def test_touch_accessed(store):
    m = store.insert_memory("x", fake_vec(1))
    store.touch_accessed([m.id])
    store.touch_accessed([m.id])
    got = store.get_memory(m.id)
    assert got.access_count == 2
    assert got.last_accessed is not None


def test_links_and_neighbors(store):
    a = store.insert_memory("a", fake_vec(1))
    b = store.insert_memory("b", fake_vec(2))
    c = store.insert_memory("c", fake_vec(3))
    store.add_link(a.id, b.id, weight=0.8)
    store.add_link(c.id, a.id, weight=0.6)
    nbrs = store.neighbors(a.id)
    assert {(n[0], n[2]) for n in nbrs} == {(b.id, 0.8), (c.id, 0.6)}
    store.add_link(a.id, a.id)  # self-link ignored
    assert len(store.neighbors(a.id)) == 2


def test_retrieval_log_roundtrip(store):
    rid = store.log_retrieval(
        user_id="default", query="q", memory_ids=["m1", "m2"], scores=[0.9, 0.5]
    )
    row = store.get_retrieval(rid)
    assert row["feedback"] is None
    store.mark_retrieval_feedback(rid, 0.8)
    row = store.get_retrieval(rid)
    assert row["feedback"] == 0.8


def test_stm_window_and_overflow(store):
    for i in range(7):
        store.add_stm_event("user", f"msg {i}")
    window = store.get_stm_window(max_items=5)
    assert [e.content for e in window] == [f"msg {i}" for i in range(2, 7)]
    overflow = store.get_stm_overflow(max_items=5)
    assert [e.content for e in overflow] == ["msg 0", "msg 1"]
    store.mark_consolidated([e.id for e in overflow])
    assert store.get_stm_overflow(max_items=5) == []


def test_stm_ttl_excludes_old(store):
    store.add_stm_event("user", "ancient", created_at="2020-01-01T00:00:00+00:00")
    store.add_stm_event("user", "fresh")
    window = store.get_stm_window(max_items=10, ttl_hours=6.0)
    assert [e.content for e in window] == ["fresh"]
    # ancient event should appear in overflow for consolidation
    overflow = store.get_stm_overflow(max_items=10, ttl_hours=6.0)
    assert [e.content for e in overflow] == ["ancient"]


def test_update_fields_validation(store):
    m = store.insert_memory("x", fake_vec(1))
    store.update_memory_fields(m.id, utility=0.5, feedback_count=1)
    got = store.get_memory(m.id)
    assert got.utility == 0.5
    with pytest.raises(ValueError):
        store.update_memory_fields(m.id, content="not allowed")


def test_cross_thread_usage(store):
    """Connection must be usable from worker threads (eval runners, MCP)."""
    from concurrent.futures import ThreadPoolExecutor

    for i in range(10):
        store.insert_memory(f"memory {i}", fake_vec(i))
    q = fake_vec(3)

    def work(i):
        store.insert_memory(f"threaded {i}", fake_vec(100 + i))
        return len(store.knn(q, k=5))

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(work, range(8)))
    assert all(r == 5 for r in results)
    assert store.count_memories() == 18


def test_sources_roundtrip(store):
    sid = store.insert_source("the assistant recommended Roscioli bakery", user_id="u1")
    assert sid.startswith("s-")
    got = store.get_sources([sid, "s-missing"])
    assert got[sid] == "the assistant recommended Roscioli bakery"
    assert "s-missing" not in got
    assert store.get_sources([]) == {}


def test_insert_memory_with_source_and_valid_from(store):
    sid = store.insert_source("raw passage", user_id="default")
    mem = store.insert_memory(
        "extracted fact", fake_vec(1), source_id=sid,
        created_at="2025-03-01T00:00:00+00:00",
    )
    got = store.get_memory(mem.id)
    assert got.source_id == sid
    assert got.valid_from == "2025-03-01T00:00:00+00:00"  # defaults to created_at
    assert got.valid_to is None


def test_get_by_slot_returns_current_only(store):
    v = fake_vec(1)
    old = store.insert_memory("user works at Acme", v, slot="employer")
    new = store.insert_memory("user works at Globex", v, slot="employer")
    other = store.insert_memory("user lives in Milan", v, slot="home_city")
    store.supersede_memory(old.id, new.id, valid_to="2025-06-01T00:00:00+00:00")

    current = {m.id for m in store.get_by_slot("employer")}
    assert current == {new.id}  # superseded employer fact excluded by default
    assert other.id not in current  # different slot excluded
    with_history = {m.id for m in store.get_by_slot("employer", current_only=False)}
    assert with_history == {old.id, new.id}


def test_supersede_keeps_memory_retrievable(store):
    v = fake_vec(1)
    old = store.insert_memory("user works at Acme", v)
    new = store.insert_memory("user works at Globex", v)
    store.supersede_memory(old.id, new.id, valid_to="2025-06-01T00:00:00+00:00")
    got = store.get_memory(old.id)
    assert got.valid_to == "2025-06-01T00:00:00+00:00"
    assert got.superseded_by == new.id
    assert not got.is_deleted  # superseded != deleted
    # still surfaces in KNN (history must remain retrievable)
    ids = [r[0] for r in store.knn(v, k=10)]
    assert old.id in ids


def test_stats(store):
    store.insert_memory("x", fake_vec(1), importance=0.4)
    store.insert_memory("y", fake_vec(2), importance=0.8)
    s = store.stats()
    assert s["memories"] == 2
    assert s["avg_importance"] == 0.6
    assert s["vector_backend"] in ("sqlite-vec", "numpy-bruteforce")
