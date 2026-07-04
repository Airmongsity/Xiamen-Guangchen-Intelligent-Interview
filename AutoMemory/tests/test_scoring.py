"""Numeric assertions for the pure scoring functions; no API keys required."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from automem import scoring
from automem.config import ScoringParams
from automem.models import Memory

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_mem(**kw) -> Memory:
    defaults = dict(
        id="m1", content="x", memory_kind="fact", source="extracted",
        importance=0.5, utility=0.0, access_count=0,
        created_at=(NOW - timedelta(days=7)).isoformat(),
        updated_at=NOW.isoformat(),
    )
    defaults.update(kw)
    return Memory(**defaults)


@pytest.fixture
def params():
    return ScoringParams()


# ----------------------------------------------------------------- decay

def test_retention_decays_over_time(params):
    fresh = make_mem(created_at=NOW.isoformat())
    old = make_mem(created_at=(NOW - timedelta(days=60)).isoformat())
    assert scoring.retention(fresh, params, now=NOW) == pytest.approx(1.0)
    assert scoring.retention(old, params, now=NOW) < scoring.retention(
        make_mem(), params, now=NOW
    )


def test_retention_formula_exact(params):
    # fact, importance=0.5, utility=0, access=0: S_eff = 14 * 1 * 1.5 = 21
    mem = make_mem(created_at=(NOW - timedelta(days=21)).isoformat())
    assert scoring.effective_strength(mem, params) == pytest.approx(21.0)
    assert scoring.retention(mem, params, now=NOW) == pytest.approx(math.exp(-1.0))


def test_access_refreshes_anchor(params):
    mem = make_mem(
        created_at=(NOW - timedelta(days=60)).isoformat(),
        last_accessed=(NOW - timedelta(days=1)).isoformat(),
    )
    stale = make_mem(created_at=(NOW - timedelta(days=60)).isoformat())
    assert scoring.retention(mem, params, now=NOW) > scoring.retention(
        stale, params, now=NOW
    )


def test_strength_modifiers(params):
    base = make_mem()
    assert scoring.effective_strength(
        make_mem(memory_kind="experience"), params
    ) > scoring.effective_strength(base, params)
    assert scoring.effective_strength(
        make_mem(source="self"), params
    ) == pytest.approx(scoring.effective_strength(base, params) * 1.5)
    assert scoring.effective_strength(
        make_mem(access_count=5), params
    ) > scoring.effective_strength(base, params)
    assert scoring.effective_strength(
        make_mem(utility=0.5), params
    ) > scoring.effective_strength(base, params)
    # negative utility must not weaken strength below neutral (b*max(u,0))
    assert scoring.effective_strength(
        make_mem(utility=-0.5), params
    ) == pytest.approx(scoring.effective_strength(base, params))


# ----------------------------------------------------------------- relevance

def test_hybrid_relevance_channels(params):
    both = scoring.hybrid_relevance(0.8, 8.0, params)
    vec_only = scoring.hybrid_relevance(0.8, None, params)
    bm25_only = scoring.hybrid_relevance(None, 8.0, params)
    assert both == pytest.approx(vec_only + bm25_only)
    assert vec_only == pytest.approx(0.7 * 0.9)
    assert 0 < bm25_only < 0.3


def test_bm25_sigmoid_monotonic(params):
    lo = scoring.bm25_sigmoid(1.0, params)
    hi = scoring.bm25_sigmoid(10.0, params)
    assert 0 < lo < hi < 1


# ----------------------------------------------------------------- priority

def test_priority_floor_keeps_ancient_relevant(params):
    ancient = make_mem(importance=0.0, utility=0.0)
    p = scoring.priority(ancient, 0.0, params)  # fully decayed
    assert p == pytest.approx(0.2)  # lam floor


def test_priority_utility_dominates_importance(params):
    high_util = scoring.priority(make_mem(importance=0.0, utility=1.0), 1.0, params)
    high_imp = scoring.priority(make_mem(importance=1.0, utility=0.0), 1.0, params)
    assert high_util > high_imp  # beta=0.8 > alpha=0.6


def test_priority_negative_utility_suppresses(params):
    neutral = scoring.priority(make_mem(utility=0.0), 1.0, params)
    bad = scoring.priority(make_mem(utility=-1.0), 1.0, params)
    assert bad < neutral
    assert bad >= 0.0


def test_priority_ablation_lam1_disables_decay():
    params = ScoringParams(lam=1.0)
    mem = make_mem()
    assert scoring.priority(mem, 0.0, params) == scoring.priority(mem, 1.0, params)


def test_priority_ablation_beta0_ignores_utility():
    params = ScoringParams(beta=0.0)
    assert scoring.priority(
        make_mem(utility=1.0), 1.0, params
    ) == scoring.priority(make_mem(utility=0.0), 1.0, params)


def test_priority_superseded_is_downweighted(params):
    current = make_mem(valid_to=None)
    historical = make_mem(valid_to=NOW.isoformat())
    p_cur = scoring.priority(current, 1.0, params)
    p_hist = scoring.priority(historical, 1.0, params)
    assert p_hist == pytest.approx(p_cur * params.superseded_penalty)
    assert p_hist < p_cur


# ----------------------------------------------------------------- spreading

def test_spread_basic(params):
    contribs = scoring.spread_activation(
        {"a": 1.0}, {"a": [("b", 0.8)]}, params
    )
    assert contribs == {"b": pytest.approx(0.35 * 0.8 * 1.0)}


def test_spread_cap_and_topk(params):
    # hub neighbor receives from 3 sources; only top-2 contributions count
    contribs = scoring.spread_activation(
        {"a": 1.0, "b": 0.9, "c": 0.5},
        {"a": [("hub", 1.0)], "b": [("hub", 1.0)], "c": [("hub", 1.0)]},
        params,
    )
    expected = 0.35 * 1.0 + 0.35 * 0.9  # c's contribution dropped
    assert contribs["hub"] == pytest.approx(expected)


def test_spread_contribution_capped(params):
    # weight 2.0 would give 0.7*s, capped at 0.5*s
    contribs = scoring.spread_activation({"a": 1.0}, {"a": [("b", 2.0)]}, params)
    assert contribs["b"] == pytest.approx(0.5)


def test_spread_gamma0_disabled():
    params = ScoringParams(gamma=0.0)
    assert scoring.spread_activation({"a": 1.0}, {"a": [("b", 1.0)]}, params) == {}


# ----------------------------------------------------------------- feedback

def test_feedback_rank_weights():
    assert scoring.feedback_rank_weight(0) == pytest.approx(1.0)
    assert scoring.feedback_rank_weight(1) == pytest.approx(1 / math.log2(3))
    assert scoring.feedback_rank_weight(2) == pytest.approx(0.5)


def test_update_utility_ema_bounded(params):
    u = 0.0
    for _ in range(100):
        u = scoring.update_utility(u, 1.0, 1.0, params)
    assert u == pytest.approx(1.0, abs=1e-6)
    for _ in range(100):
        u = scoring.update_utility(u, -1.0, 1.0, params)
    assert u == pytest.approx(-1.0, abs=1e-6)


def test_update_utility_single_step(params):
    # 0 + 0.3 * 1.0 * (1 - 0) = 0.3
    assert scoring.update_utility(0.0, 1.0, 1.0, params) == pytest.approx(0.3)
    # rank-1 weight applied
    assert scoring.update_utility(0.0, 1.0, 0.63, params) == pytest.approx(0.189)


# ----------------------------------------------------------------- misc

def test_minmax_norm():
    assert scoring.minmax_norm([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert scoring.minmax_norm([2.0, 2.0]) == [1.0, 1.0]
    assert scoring.minmax_norm([]) == []


def test_fuse_final(params):
    assert scoring.fuse_final(1.0, 0.0, params) == pytest.approx(0.7)
    assert scoring.fuse_final(0.0, 1.0, params) == pytest.approx(0.3)
