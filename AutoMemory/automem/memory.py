"""AutoMemory facade: the public API of the library."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from . import scoring
from .config import AutoMemConfig
from .extraction import MemoryExtractor
from .feedback import FeedbackHandler
from .models import Memory, MemoryEvent, RecallResult, utcnow_iso
from .providers import SiliconFlowEmbedder, SiliconFlowReranker, make_llm
from .retrieval import Retriever
from .store import MemoryStore

SELF_MEMORY_DEFAULT_IMPORTANCE = 0.75

logger = logging.getLogger(__name__)


class AutoMemory:
    def __init__(
        self,
        config: AutoMemConfig | None = None,
        *,
        embedder=None,
        llm=None,
        reranker=None,
        readonly=False,
    ):
        """Providers can be injected (e.g. fakes in tests); by default they are
        built from the config, and the LLM/reranker are lazy so that purely
        embedding-based usage works without a DeepSeek key. readonly=True opens the
        store read-only (evaluation mode) — any write attempt raises."""
        self.config = config or AutoMemConfig.from_env()
        self.readonly = readonly
        self.store = MemoryStore(
            self.config.db_path, self.config.embed_dim, readonly=readonly
        )
        self._embedder = embedder or SiliconFlowEmbedder(self.config)
        self._llm = llm
        self._extractor = MemoryExtractor(llm) if llm else None
        if reranker is None and self.config.siliconflow_api_key:
            reranker = SiliconFlowReranker(self.config)
        self.retriever = Retriever(self.store, self._embedder, reranker, self.config)
        self._feedback = FeedbackHandler(self.store, self.config.scoring)

    def _get_extractor(self) -> MemoryExtractor:
        if self._extractor is None:
            self._llm = self._llm or make_llm(self.config)
            self._extractor = MemoryExtractor(self._llm)
        return self._extractor

    def close(self) -> None:
        self.store.close()

    # ----------------------------------------------------------------- write

    def add(
        self,
        messages: list[dict],
        *,
        user_id: str = "default",
        agent_id: str | None = None,
        infer: bool = True,
        conversation_date: str | None = None,
        created_at: str | None = None,
        metadata: dict | None = None,
    ) -> list[MemoryEvent]:
        """Extract memories from a conversation and reconcile them into the store."""
        if not infer:
            items = [
                {"content": m.get("content", ""), "created_at": created_at}
                for m in messages
                if m.get("content")
            ]
            added = self.add_batch(items, user_id=user_id, metadata=metadata)
            return [MemoryEvent(event="ADD", memory=m) for m in added]

        extracted = self._get_extractor().extract(
            messages, conversation_date=conversation_date
        )
        if not extracted:
            return []
        # hybrid granularity: persist the verbatim conversation once; every fact
        # extracted from it points back to this passage for evidence expansion.
        raw_passage = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
            if m.get("content")
        )
        source_id = (
            self.store.insert_source(
                raw_passage, user_id=user_id, created_at=created_at
            )
            if raw_passage
            else None
        )
        events: list[MemoryEvent] = []
        for item in extracted:
            events.append(
                self._reconcile_and_store(
                    item.content,
                    kind=item.kind,
                    importance=item.importance,
                    source="extracted",
                    user_id=user_id,
                    agent_id=agent_id,
                    created_at=created_at,
                    metadata=metadata,
                    source_id=source_id,
                    slot=item.slot,
                )
            )
        return events

    def remember(
        self,
        content: str,
        *,
        kind: str = "experience",
        importance: float | None = None,
        user_id: str = "default",
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        """AI self-authored memory: stored directly, no LLM in the loop."""
        embedding = self._embedder.embed(content)
        near = self.store.knn(embedding, 1, user_id=user_id)
        if near and near[0][1] >= self.config.remember_merge_threshold:
            # near-duplicate of an existing memory: refresh it instead of adding
            existing = self.store.get_memory(near[0][0])
            self.store.update_memory_content(existing.id, content, embedding)
            if importance is not None and importance > existing.importance:
                self.store.update_memory_fields(existing.id, importance=importance)
            return self.store.get_memory(existing.id)

        mem = self.store.insert_memory(
            content,
            embedding,
            memory_kind=kind,
            source="self",
            user_id=user_id,
            agent_id=agent_id,
            importance=(
                importance if importance is not None else SELF_MEMORY_DEFAULT_IMPORTANCE
            ),
            metadata=metadata,
        )
        self._auto_link(mem.id, embedding, user_id=user_id)
        return mem

    def add_batch(
        self,
        items: Iterable[str | dict],
        *,
        user_id: str = "default",
        infer: bool = False,
        metadata: dict | None = None,
    ) -> list[Memory]:
        """Bulk import. Items may carry created_at to backfill history (eval needs
        this). infer=False stores items verbatim; infer=True runs each item
        through the extraction pipeline."""
        norm: list[dict] = []
        for it in items:
            if isinstance(it, str):
                norm.append({"content": it})
            else:
                norm.append(dict(it))
        norm = [it for it in norm if it.get("content")]
        if not norm:
            return []

        if infer:
            out: list[Memory] = []
            for it in norm:
                events = self.add(
                    [{"role": "user", "content": it["content"]}],
                    user_id=user_id,
                    created_at=it.get("created_at"),
                    conversation_date=it.get("created_at"),
                    metadata=it.get("metadata") or metadata,
                )
                out.extend(e.memory for e in events if e.memory and e.event == "ADD")
            return out

        embeddings = self._embedder.embed_batch([it["content"] for it in norm])
        out = []
        for it, emb in zip(norm, embeddings):
            out.append(
                self.store.insert_memory(
                    it["content"],
                    emb,
                    memory_kind=it.get("kind", "fact"),
                    source=it.get("source", "import"),
                    user_id=user_id,
                    importance=it.get("importance", 0.5),
                    created_at=it.get("created_at"),
                    metadata=it.get("metadata") or metadata,
                )
            )
        return out

    def observe(
        self,
        role: str,
        content: str,
        *,
        session_id: str = "default",
        user_id: str = "default",
    ) -> None:
        """Append a raw message to short-term memory. No LLM, no embedding."""
        self.store.add_stm_event(role, content, session_id=session_id, user_id=user_id)

    def consolidate(self, *, user_id: str = "default") -> list[MemoryEvent]:
        """Run events that fell out of the STM window through extraction."""
        overflow = self.store.get_stm_overflow(
            user_id=user_id,
            max_items=self.config.stm_max_items,
            ttl_hours=self.config.stm_ttl_hours,
        )
        if not overflow:
            return []
        events: list[MemoryEvent] = []
        by_session: dict[str, list] = {}
        for ev in overflow:
            by_session.setdefault(ev.session_id, []).append(ev)
        for session_events in by_session.values():
            messages = [{"role": e.role, "content": e.content} for e in session_events]
            events.extend(
                self.add(
                    messages,
                    user_id=user_id,
                    conversation_date=session_events[0].created_at[:10],
                )
            )
        self.store.mark_consolidated([e.id for e in overflow])
        return events

    # ----------------------------------------------------------------- read

    def recall(
        self,
        query: str,
        *,
        top_k: int | None = None,
        user_id: str = "default",
        include_short_term: bool = True,
        history_mode: str | None = None,
        as_of: str | None = None,
        touch: bool = True,
        log: bool = True,
    ) -> RecallResult:
        """touch/log default True for live use (retrieval reinforces memories and is
        logged for feedback). Set both False for read-only evaluation so repeated
        runs don't mutate access_count/last_accessed and skew later scoring. as_of
        (ISO timestamp) scores retention as of that moment instead of wall-clock."""
        if self.readonly and as_of is None:
            # evaluation/replay mode: the clock must be injected, never implicit —
            # wall-clock 'now' on archived data collapses retention to a constant.
            raise ValueError(
                "readonly recall requires as_of (inject the query/replay time)"
            )
        return self.retriever.recall(
            query,
            user_id=user_id,
            top_k=top_k,
            include_short_term=include_short_term,
            history_mode=history_mode,
            as_of=as_of,
            touch=touch,
            log=log,
        )

    def get(self, memory_id: str) -> Memory | None:
        return self.store.get_memory(memory_id)

    def list_memories(self, *, user_id: str = "default", limit: int = 100) -> list[Memory]:
        return self.store.list_memories(user_id=user_id, limit=limit)

    # ----------------------------------------------------------- feedback/admin

    def report_outcome(
        self,
        *,
        retrieval_id: str | None = None,
        memory_ids: list[str] | None = None,
        quality: float,
    ) -> dict:
        return self._feedback.report_outcome(
            retrieval_id=retrieval_id, memory_ids=memory_ids, quality=quality
        )

    def forget(self, memory_id: str, *, hard: bool = False) -> None:
        if hard:
            self.store.hard_delete(memory_id)
        else:
            self.store.soft_delete(memory_id)

    def link(self, src_id: str, dst_id: str, *, kind: str = "related", weight: float = 1.0) -> None:
        self.store.add_link(src_id, dst_id, link_kind=kind, weight=weight)

    def maintain(self, *, user_id: str = "default") -> dict:
        """Soft-forget memories whose retention dropped below threshold and that
        never earned positive utility."""
        now = datetime.now(timezone.utc)
        forgotten = 0
        for mem in self.store.list_memories(user_id=user_id, limit=1_000_000):
            ret = scoring.retention(mem, self.config.scoring, now=now)
            if ret < self.config.scoring.forget_retention_threshold and mem.utility <= 0:
                self.store.soft_delete(mem.id)
                forgotten += 1
        return {"forgotten": forgotten}

    def stats(self, *, user_id: str = "default") -> dict:
        return self.store.stats(user_id=user_id)

    # ----------------------------------------------------------------- internal

    def _reconcile_and_store(
        self,
        content: str,
        *,
        kind: str,
        importance: float,
        source: str,
        user_id: str,
        agent_id: str | None,
        created_at: str | None,
        metadata: dict | None,
        source_id: str | None = None,
        slot: str | None = None,
    ) -> MemoryEvent:
        embedding = self._embedder.embed(content)
        sp = self.config.scoring
        near = self.store.knn(embedding, sp.reconcile_top_n, user_id=user_id)
        # Reconcile candidates: a fixed top-N quota above a similarity floor (not a
        # hard high threshold), PLUS any current memory already filling the same
        # attribute slot. The slot path catches value changes that only embed
        # moderately, e.g. "works at Acme" vs "works at Globex".
        cand_ids: list[str] = [mid for mid, cos in near if cos >= sp.reconcile_floor]
        if slot:
            for m in self.store.get_by_slot(slot, user_id=user_id):
                if m.id not in cand_ids:
                    cand_ids.append(m.id)

        # Only CURRENT memories are reconcile targets. Superseded history rows stay
        # retrievable (is_deleted=0) so KNN can surface them, but reconciling against
        # them would tangle the supersession chains (a stale value could "win").
        existing = [
            (mid, mem.content)
            for mid in cand_ids
            if (mem := self.store.get_memory(mid)) is not None and not mem.valid_to
        ]
        decision = (
            self._get_extractor().reconcile(content, existing) if existing else None
        )

        if decision and decision.op == "NONE":
            return MemoryEvent(event="NONE")

        if decision and decision.op == "UPDATE":
            return self._apply_update(
                decision, content, importance=importance,
                created_at=created_at, source_id=source_id, slot=slot,
            )

        if decision and decision.op == "DELETE":
            # Knowledge update: the old fact is no longer current. We do NOT delete
            # it (bi-temporal invalidation) — it stays retrievable as history while
            # the candidate becomes the new current truth.
            old = self.store.get_memory(decision.target_id)
            new_slot = slot or old.slot
            mem = self.store.insert_memory(
                content, embedding,
                memory_kind=kind, source=source, user_id=user_id, agent_id=agent_id,
                importance=importance, created_at=created_at, metadata=metadata,
                source_id=source_id, slot=new_slot,
            )
            valid_to = mem.valid_from or mem.created_at
            self.store.supersede_memory(old.id, mem.id, valid_to=valid_to)
            # auto-link first, then assert the contradicts edge LAST so it isn't
            # downgraded to "related": the superseded memory is still visible
            # (is_deleted=0), so it is a valid auto-link neighbor of the new fact.
            self._auto_link(mem.id, embedding, user_id=user_id)
            self.store.add_link(mem.id, old.id, link_kind="contradicts", weight=1.0)
            self._enforce_single_slot(mem.id, new_slot, user_id=user_id, valid_to=valid_to)
            return MemoryEvent(event="DELETE", memory=mem, previous_content=old.content)

        mem = self.store.insert_memory(
            content, embedding,
            memory_kind=kind, source=source, user_id=user_id, agent_id=agent_id,
            importance=importance, created_at=created_at, metadata=metadata,
            source_id=source_id, slot=slot,
        )
        self._auto_link(mem.id, embedding, user_id=user_id, knn_results=near)
        # single-valued attribute invariant: a brand-new value for a slot retires
        # any prior current value of that same slot, even when the LLM judged this
        # an ADD rather than a contradiction.
        self._enforce_single_slot(
            mem.id, slot, user_id=user_id, valid_to=mem.valid_from or mem.created_at
        )
        return MemoryEvent(event="ADD", memory=mem)

    def _apply_update(
        self, decision, content: str, *, importance: float,
        created_at: str | None, source_id: str | None, slot: str | None,
    ) -> MemoryEvent:
        """Merge into the existing memory, archiving its pre-merge version as
        bi-temporal history so the old value stays answerable. The current row
        keeps its id (links/feedback persist) and becomes the merged truth."""
        old = self.store.get_memory(decision.target_id)
        new_text = decision.text or content
        # 1. snapshot the old version (reuse its stored embedding; no re-embed)
        old_emb = self.store.index.get(old.id)
        if old_emb is not None:
            hist = self.store.insert_memory(
                old.content, old_emb,
                memory_kind=old.memory_kind, source=old.source, user_id=old.user_id,
                agent_id=old.agent_id, importance=old.importance,
                created_at=old.created_at, source_id=old.source_id,
                valid_from=old.valid_from, slot=old.slot,
            )
            self.store.supersede_memory(
                hist.id, old.id, valid_to=created_at or utcnow_iso()
            )
        else:
            # bi-temporal invariant breach: we are about to overwrite the current
            # value but cannot archive its prior version (no stored embedding).
            # Do it loudly rather than silently losing the old fact.
            logger.warning(
                "UPDATE on memory %s: no stored embedding; old value NOT archived "
                "to history before overwrite. old_content=%r",
                old.id, (old.content or "")[:120],
            )
        # 2. the existing row becomes the merged, current truth
        new_emb = self._embedder.embed(new_text)
        self.store.update_memory_content(old.id, new_text, new_emb)
        fields: dict = {}
        if importance > old.importance:
            fields["importance"] = importance
        if source_id:  # repoint to the freshest passage the merge drew from
            fields["source_id"] = source_id
        if slot and not old.slot:
            fields["slot"] = slot
        if fields:
            self.store.update_memory_fields(old.id, **fields)
        # the merged row is now the single current value of its slot; retire any
        # other current memory sharing that slot.
        self._enforce_single_slot(
            old.id, slot or old.slot, user_id=old.user_id,
            valid_to=created_at or utcnow_iso(),
        )
        return MemoryEvent(
            event="UPDATE",
            memory=self.store.get_memory(old.id),
            previous_content=old.content,
        )

    def _enforce_single_slot(
        self, current_id: str, slot: str | None, *, user_id: str, valid_to: str
    ) -> None:
        """A slot names a single-valued attribute, so it can have only one current
        value. Supersede every OTHER current memory filling the same slot, pointing
        it at the winner and asserting a contradicts edge."""
        if not slot:
            return
        for m in self.store.get_by_slot(slot, user_id=user_id):  # current_only
            if m.id == current_id or m.valid_to:
                continue
            self.store.supersede_memory(m.id, current_id, valid_to=valid_to)
            self.store.add_link(current_id, m.id, link_kind="contradicts", weight=1.0)

    def _auto_link(
        self,
        memory_id: str,
        embedding,
        *,
        user_id: str,
        knn_results: list[tuple[str, float]] | None = None,
    ) -> None:
        """Link a new memory to moderately-similar neighbors (the logic chain)."""
        cfg = self.config
        near = (
            knn_results
            if knn_results is not None
            else self.store.knn(embedding, cfg.link_max_count + 3, user_id=user_id)
        )
        linked = 0
        for nid, cos in near:
            if nid == memory_id:
                continue
            if cfg.link_min_cos <= cos < cfg.link_max_cos:
                nbr = self.store.get_memory(nid)
                if nbr is None or nbr.valid_to:  # don't chain onto stale history
                    continue
                self.store.add_link(memory_id, nid, link_kind="related", weight=cos)
                linked += 1
                if linked >= cfg.link_max_count:
                    break
