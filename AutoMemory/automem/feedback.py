"""Outcome feedback: attribute a quality signal back to retrieved memories.

quality in [0,1] maps to signal s = 2q-1 in [-1,1]. Memories in the retrieval
are updated with a DCG-style rank discount (earlier ranks presumably influenced
the answer more) through a bounded EMA. Reporting twice on the same
retrieval_id is a no-op (idempotent).
"""

from __future__ import annotations

import json

from . import scoring
from .config import ScoringParams
from .models import utcnow_iso
from .store import MemoryStore


class FeedbackHandler:
    def __init__(self, store: MemoryStore, params: ScoringParams):
        self._store = store
        self._params = params

    def report_outcome(
        self,
        *,
        retrieval_id: str | None = None,
        memory_ids: list[str] | None = None,
        quality: float,
    ) -> dict:
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if (retrieval_id is None) == (memory_ids is None):
            raise ValueError("Provide exactly one of retrieval_id or memory_ids")
        signal = 2.0 * quality - 1.0

        if retrieval_id is not None:
            row = self._store.get_retrieval(retrieval_id)
            if row is None:
                raise KeyError(f"Unknown retrieval_id: {retrieval_id}")
            if row["feedback"] is not None:
                return {"updated": 0, "skipped": "already reported"}
            ranked_ids = json.loads(row["memory_ids"])
            updates = [
                (mid, scoring.feedback_rank_weight(rank))
                for rank, mid in enumerate(ranked_ids)
            ]
            self._store.mark_retrieval_feedback(retrieval_id, quality)
        else:
            updates = [(mid, 1.0) for mid in memory_ids]

        updated = 0
        for mid, rank_weight in updates:
            mem = self._store.get_memory(mid)
            if mem is None:
                continue
            new_utility = scoring.update_utility(
                mem.utility, signal, rank_weight, self._params
            )
            self._store.update_memory_fields(
                mid,
                utility=new_utility,
                feedback_count=mem.feedback_count + 1,
                updated_at=utcnow_iso(),
            )
            updated += 1
        return {"updated": updated, "signal": signal}
