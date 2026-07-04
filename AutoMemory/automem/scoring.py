"""Pure scoring functions: Ebbinghaus decay, hybrid relevance, priority,
activation spreading, and final fusion. No I/O here — everything is driven by
ScoringParams so ablation experiments only need to tweak the config.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import ScoringParams
from .models import Memory, parse_iso


def effective_strength(mem: Memory, params: ScoringParams) -> float:
    """Memory strength S_eff in days; larger = slower forgetting."""
    s_base = params.s_base.get(mem.memory_kind, params.s_base["fact"])
    if mem.source == "self":
        s_base *= params.self_source_multiplier
    reinforcement = 1.0 + params.rho * math.log1p(mem.access_count)
    consolidation = 1.0 + params.a * mem.importance + params.b * max(mem.utility, 0.0)
    return s_base * reinforcement * consolidation


def retention(mem: Memory, params: ScoringParams, *, now: datetime | None = None) -> float:
    """Ebbinghaus retention exp(-dt/S). Anchor = max(created_at, last_accessed),
    so each retrieval acts as a review that resets the forgetting curve."""
    now = now or datetime.now(timezone.utc)
    anchor = parse_iso(mem.created_at)
    if mem.last_accessed:
        anchor = max(anchor, parse_iso(mem.last_accessed))
    dt_days = max((now - anchor).total_seconds() / 86400.0, 0.0)
    return math.exp(-dt_days / effective_strength(mem, params))


def cos_norm(cosine: float) -> float:
    """Map cosine [-1,1] to [0,1]."""
    return (1.0 + cosine) / 2.0


def bm25_sigmoid(bm25_raw: float, params: ScoringParams) -> float:
    """Squash a raw (higher-better) BM25 score into [0,1], mem0-style."""
    z = (bm25_raw - params.bm25_sigmoid_center) / params.bm25_sigmoid_scale
    return 1.0 / (1.0 + math.exp(-z))


def hybrid_relevance(
    cosine: float | None, bm25_raw: float | None, params: ScoringParams
) -> float:
    """Weighted blend of the two channels; a missing channel scores 0."""
    vec_part = cos_norm(cosine) if cosine is not None else 0.0
    bm25_part = bm25_sigmoid(bm25_raw, params) if bm25_raw is not None else 0.0
    return params.w_vec * vec_part + params.w_bm25 * bm25_part


def priority(mem: Memory, ret: float, params: ScoringParams) -> float:
    """Decay-floored retention times the importance/utility multiplier.
    utility < 0 actively suppresses a memory. A superseded fact (valid_to set)
    is down-weighted but not removed, so current truth ranks above history."""
    decay_term = params.lam + (1.0 - params.lam) * ret
    weight_term = 1.0 + params.alpha * mem.importance + params.beta * mem.utility
    pri = decay_term * max(weight_term, 0.0)
    if mem.valid_to:
        pri *= params.superseded_penalty
    return pri


def spread_activation(
    scores: dict[str, float],
    neighbors_of: dict[str, list[tuple[str, float]]],
    params: ScoringParams,
) -> dict[str, float]:
    """One-hop activation spreading over memory links.

    scores: {memory_id: score_pre} for the current candidate set (the sources).
    neighbors_of: {source_id: [(neighbor_id, link_weight), ...]}.
    Returns {memory_id: contribution} — additions for existing candidates and
    seed scores for newly pulled-in neighbors. Per neighbor only the top
    `spread_max_contribs` contributions count, each capped at
    `spread_cap_frac * source_score` to keep hub nodes from flooding.
    """
    if params.gamma <= 0.0:
        return {}
    contribs: dict[str, list[float]] = {}
    for src_id, s_m in scores.items():
        for nbr_id, weight in neighbors_of.get(src_id, []):
            if nbr_id == src_id:
                continue
            c = min(params.gamma * weight * s_m, params.spread_cap_frac * s_m)
            if c > 0.0:
                contribs.setdefault(nbr_id, []).append(c)
    return {
        nid: sum(sorted(cs, reverse=True)[: params.spread_max_contribs])
        for nid, cs in contribs.items()
    }


def fuse_final(
    rerank_score: float, score_pre_normed: float, params: ScoringParams
) -> float:
    return params.w_rerank * rerank_score + params.w_pre * score_pre_normed


def minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def feedback_rank_weight(rank: int) -> float:
    """DCG-style discount: rank 0 -> 1.0, 1 -> 0.63, 2 -> 0.5, ..."""
    return 1.0 / math.log2(rank + 2)


def update_utility(current: float, signal: float, rank_weight: float, params: ScoringParams) -> float:
    """EMA update, bounded in [-1, 1] since signal is."""
    return current + params.eta * rank_weight * (signal - current)
