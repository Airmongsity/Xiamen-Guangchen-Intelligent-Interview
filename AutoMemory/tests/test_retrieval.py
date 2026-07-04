"""Retrieval pipeline tests with deterministic fake providers; no API keys."""

import numpy as np
import pytest

from automem.config import AutoMemConfig
from automem.retrieval import Retriever
from automem.store import MemoryStore

DIM = 8

# fixed orthogonal-ish topic vectors
TOPICS = {
    "python": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "cooking": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float),
    "weather": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=float),
}


class FakeEmbedder:
    dim = DIM

    def embed(self, text: str):
        vec = np.full(DIM, 0.01)
        for topic, tvec in TOPICS.items():
            if topic in text.lower():
                vec = vec + tvec
        return (vec / np.linalg.norm(vec)).tolist()

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class FakeReranker:
    """Scores by crude token overlap; deterministic."""

    def rerank(self, query, documents, *, top_n=None):
        q_tokens = set(query.lower().split())
        scored = [
            (i, len(q_tokens & set(doc.lower().split())) / (len(q_tokens) or 1))
            for i, doc in enumerate(documents)
        ]
        return sorted(scored, key=lambda t: t[1], reverse=True)


@pytest.fixture
def setup(tmp_path):
    cfg = AutoMemConfig(db_path=str(tmp_path / "t.db"), embed_dim=DIM, top_k=5)
    store = MemoryStore(cfg.db_path, embed_dim=DIM)
    emb = FakeEmbedder()
    retriever = Retriever(store, emb, FakeReranker(), cfg)
    yield store, emb, retriever
    store.close()


def add(store, emb, content, **kw):
    return store.insert_memory(content, emb.embed(content), **kw)


def test_recall_relevance_ordering(setup):
    store, emb, retriever = setup
    rel = add(store, emb, "user loves python programming")
    add(store, emb, "user enjoys cooking pasta")
    add(store, emb, "weather was rainy last week")
    result = retriever.recall("python tips", include_short_term=False)
    assert result.long_term
    assert result.long_term[0].memory.id == rel.id


def test_recall_side_effects_and_log(setup):
    store, emb, retriever = setup
    m = add(store, emb, "user loves python")
    result = retriever.recall("python")
    assert result.retrieval_id.startswith("r-")
    row = store.get_retrieval(result.retrieval_id)
    assert row is not None and m.id in row["memory_ids"]
    assert store.get_memory(m.id).access_count == 1


def test_spreading_pulls_linked_memory(setup):
    store, emb, retriever = setup
    src = add(store, emb, "user loves python programming")
    # linked memory has NO lexical/vector overlap with the query
    linked = add(store, emb, "user dislikes verbose answers")
    store.add_link(src.id, linked.id, weight=0.9)
    add(store, emb, "weather was rainy")

    result = retriever.recall("python", include_short_term=False)
    ids = [sm.memory.id for sm in result.long_term]
    assert linked.id in ids
    linked_sm = next(sm for sm in result.long_term if sm.memory.id == linked.id)
    # the memory received a spreading contribution on top of its own score
    # (score_pre > relevance * priority), whether or not it was already a candidate
    assert linked_sm.score_pre > linked_sm.relevance * linked_sm.priority + 1e-9


def test_spreading_disabled_by_gamma0(setup, tmp_path):
    store, emb, _ = setup
    cfg = AutoMemConfig(db_path="ignored", embed_dim=DIM, top_k=5)
    cfg.scoring.gamma = 0.0
    retriever = Retriever(store, emb, FakeReranker(), cfg)
    src = add(store, emb, "user loves python programming")
    far = add(store, emb, "completely unrelated cooking topic xyz")
    store.add_link(src.id, far.id, weight=0.9)
    result = retriever.recall("python", include_short_term=False)
    far_sm = [sm for sm in result.long_term if sm.memory.id == far.id]
    assert not any(sm.via_link for sm in far_sm)


def test_short_term_window_included(setup):
    store, emb, retriever = setup
    store.add_stm_event("user", "we just talked about sqlite")
    add(store, emb, "user loves python")
    result = retriever.recall("python")
    assert len(result.short_term) == 1
    prompt = result.to_prompt()
    assert "sqlite" in prompt
    assert result.retrieval_id in prompt
    assert "report_outcome" in prompt


def test_empty_db_recall(setup):
    _, _, retriever = setup
    result = retriever.recall("anything")
    assert result.long_term == []


def test_utility_affects_ranking(setup):
    store, emb, retriever = setup
    # two memories with identical content relevance; one has negative utility
    good = add(store, emb, "user loves python style A")
    bad = add(store, emb, "user loves python style B")
    store.update_memory_fields(bad.id, utility=-0.9)
    store.update_memory_fields(good.id, utility=0.9)
    result = retriever.recall("python style", include_short_term=False)
    ids = [sm.memory.id for sm in result.long_term]
    assert ids.index(good.id) < ids.index(bad.id)
