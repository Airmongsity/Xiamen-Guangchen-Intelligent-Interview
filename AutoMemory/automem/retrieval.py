"""Retrieval pipeline: recall channels -> hybrid scoring -> activation
spreading -> rerank -> final fusion -> retrieval log."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from . import scoring
from .config import AutoMemConfig
from .models import Memory, RecallResult, ScoredMemory, parse_iso
from .providers.embedder import SiliconFlowEmbedder
from .providers.reranker import SiliconFlowReranker
from .store import MemoryStore

# EXPERIMENTAL / ABLATION-ONLY: this gate is consulted ONLY when
# history_separate_quota=True (off by default). In the shipped "compete" config it
# never runs — see config.py. Kept so `--separate-quota --history-mode auto` is
# reproducible. Cheap regex (no LLM): common English markers + 中文 回溯语.
_RETRO_RE = re.compile(
    r"\b(used to|use to|previously|formerly|earlier|before|back then|"
    r"originally|no longer|not anymore|any ?more|in the past|once|former|prior)\b"
    r"|以前|之前|原来|曾经|过去|从前|当初|最初|本来",
    re.IGNORECASE,
)


def is_retrospective(query: str) -> bool:
    return bool(_RETRO_RE.search(query or ""))


class Retriever:
    def __init__(
        self,
        store: MemoryStore,
        embedder: SiliconFlowEmbedder,
        reranker: SiliconFlowReranker | None,
        config: AutoMemConfig,
    ):
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._cfg = config

    def recall(
        self,
        query: str,
        *,
        user_id: str = "default",
        top_k: int | None = None,
        include_short_term: bool = True,
        history_mode: str | None = None,
        as_of: str | None = None,
        log: bool = True,
        touch: bool = True,
    ) -> RecallResult:
        cfg = self._cfg
        top_k = top_k or cfg.top_k
        if self._store.readonly:
            # a read-only store is evaluation/replay mode: never reinforce or log,
            # regardless of what the caller passed. mode=ro still backstops any
            # write that bypasses recall (e.g. a direct touch_accessed) by raising.
            touch = log = False
        mode = (history_mode or cfg.history_mode).lower()
        want_history = mode == "on" or (mode == "auto" and is_retrospective(query))

        # 1. short-term memory: full recent window
        short_term = []
        if include_short_term:
            short_term = self._store.get_stm_window(
                user_id=user_id,
                max_items=cfg.stm_max_items,
                ttl_hours=cfg.stm_ttl_hours,
            )

        # 2. dual-channel recall
        query_vec = self._embedder.embed(query)
        vec_hits = dict(
            self._store.knn(query_vec, cfg.recall_candidates, user_id=user_id)
        )
        bm25_hits = dict(
            self._store.fts_search(query, cfg.recall_candidates, user_id=user_id)
        )

        candidate_ids = list(vec_hits.keys() | bm25_hits.keys())
        if not candidate_ids:
            rid = (
                self._store.log_retrieval(
                    user_id=user_id, query=query, memory_ids=[], scores=[]
                )
                if log
                else "r-empty"
            )
            return RecallResult(retrieval_id=rid, query=query, short_term=short_term)

        memories = {m.id: m for m in self._store.get_memories(candidate_ids)}
        # `as_of` lets the caller score retention at the time the query was actually
        # asked (e.g. an eval question's date), instead of wall-clock now. Without
        # it, replaying a years-old conversation collapses all retention to ~0.
        now = parse_iso(as_of) if as_of else datetime.now(timezone.utc)

        def score(mem: Memory) -> ScoredMemory:
            ret = scoring.retention(mem, cfg.scoring, now=now)
            rel = scoring.hybrid_relevance(
                vec_hits.get(mem.id), bm25_hits.get(mem.id), cfg.scoring
            )
            pri = scoring.priority(mem, ret, cfg.scoring)
            return ScoredMemory(
                memory=mem, relevance=rel, retention=ret, priority=pri,
                score_pre=rel * pri,
            )

        # DEFAULT PATH ("compete", history_separate_quota=False): every memory —
        # current and superseded — competes in one pool; superseded ones are merely
        # down-weighted (superseded_penalty in priority). This is what the shipped
        # config and the published held-out result use.
        # The `if` branch is the EXPERIMENTAL separate-quota ablation (off by
        # default; see config.py): current facts own top_k, history gets its own
        # gated quota below. Kept only for `--separate-quota` reproducibility.
        if cfg.history_separate_quota:
            current = {mid: m for mid, m in memories.items() if not m.valid_to}
            history = {mid: m for mid, m in memories.items() if m.valid_to}
        else:
            current = dict(memories)
            history = {}

        # 3. main pipeline on CURRENT only: score -> pool -> spread -> rerank
        scored = {mid: score(mem) for mid, mem in current.items()}
        pool = sorted(scored.values(), key=lambda s: s.score_pre, reverse=True)
        pool = pool[: cfg.rerank_pool]
        pool = self._spread(pool, scored, now)
        ranked = self._rerank_and_fuse(query, pool)
        result_list = ranked[:top_k]

        # 4. history quota — EXPERIMENTAL, only reachable when separate_quota=True
        # (history is {} otherwise, so this is a no-op in the default config).
        history_list: list[ScoredMemory] = []
        if want_history and history and cfg.history_quota > 0:
            h_pool = sorted(
                (score(m) for m in history.values()),
                key=lambda s: s.score_pre, reverse=True,
            )[: cfg.rerank_pool]
            history_list = self._rerank_and_fuse(query, h_pool)[: cfg.history_quota]

        # 5. side effects: review refresh + retrieval log
        final = result_list + history_list
        final_ids = [sm.memory.id for sm in final]
        if touch and final_ids:
            self._store.touch_accessed(final_ids)
        rid = (
            self._store.log_retrieval(
                user_id=user_id,
                query=query,
                memory_ids=final_ids,
                scores=[sm.final_score for sm in final],
            )
            if log
            else "r-nolog"
        )
        sources = self._expand_sources(final)
        return RecallResult(
            retrieval_id=rid, query=query, short_term=short_term,
            long_term=result_list, history=history_list, sources=sources,
        )

    def _expand_sources(self, result_list: list[ScoredMemory]) -> dict[str, str]:
        """Hybrid granularity: fetch the verbatim passages behind the top facts."""
        cfg = self._cfg
        if not cfg.expand_sources or not result_list:
            return {}
        wanted: list[str] = []
        for sm in result_list[: cfg.expand_top_n]:
            sid = sm.memory.source_id
            if sid and sid not in wanted:
                wanted.append(sid)
        if not wanted:
            return {}
        raw = self._store.get_sources(wanted)
        return {
            sid: raw[sid][: cfg.source_max_chars] for sid in wanted if sid in raw
        }

    # ----------------------------------------------------------- internals

    def _spread(
        self,
        pool: list[ScoredMemory],
        scored: dict[str, ScoredMemory],
        now: datetime,
    ) -> list[ScoredMemory]:
        cfg = self._cfg
        if cfg.scoring.gamma <= 0.0 or not pool:
            return pool

        pool_scores = {sm.memory.id: sm.score_pre for sm in pool}
        neighbors_of = {
            mid: [(nid, w) for nid, _kind, w in self._store.neighbors(mid)]
            for mid in pool_scores
        }
        contribs = scoring.spread_activation(pool_scores, neighbors_of, cfg.scoring)

        for nid, contribution in contribs.items():
            if nid in scored:
                scored[nid].score_pre += contribution
            else:
                mem = self._store.get_memory(nid)
                if mem is None or mem.is_deleted or mem.valid_to:
                    continue  # history is handled by its own quota, not spreading
                ret = scoring.retention(mem, cfg.scoring, now=now)
                pri = scoring.priority(mem, ret, cfg.scoring)
                scored[nid] = ScoredMemory(
                    memory=mem, relevance=0.0, retention=ret, priority=pri,
                    score_pre=contribution * pri, via_link=True,
                )

        pool = sorted(scored.values(), key=lambda s: s.score_pre, reverse=True)
        return pool[: cfg.rerank_pool]

    def _rerank_and_fuse(
        self, query: str, pool: list[ScoredMemory]
    ) -> list[ScoredMemory]:
        cfg = self._cfg
        pre_normed = scoring.minmax_norm([sm.score_pre for sm in pool])

        if self._reranker is None:
            for sm, pre in zip(pool, pre_normed):
                sm.final_score = pre
            return sorted(pool, key=lambda s: s.final_score, reverse=True)

        docs = [sm.memory.content for sm in pool]
        try:
            rerank_results = self._reranker.rerank(query, docs)
        except Exception:
            # reranker outage degrades to pre-score ordering instead of failing
            for sm, pre in zip(pool, pre_normed):
                sm.final_score = pre
            return sorted(pool, key=lambda s: s.final_score, reverse=True)
        rerank_by_idx = dict(rerank_results)
        for i, (sm, pre) in enumerate(zip(pool, pre_normed)):
            sm.rerank_score = rerank_by_idx.get(i, 0.0)
            sm.final_score = scoring.fuse_final(sm.rerank_score, pre, cfg.scoring)
        return sorted(pool, key=lambda s: s.final_score, reverse=True)
