"""Shared helpers for the LongMemEval (V1) evaluation runners.

LongMemEval-S item schema (list[dict]):
  question_id        str   (abstention questions end with "_abs")
  question_type      str   single-session-user | single-session-assistant |
                           single-session-preference | multi-session |
                           temporal-reasoning | knowledge-update
  question           str
  answer             str
  question_date      str
  haystack_dates     list[str]
  haystack_session_ids  list[str]
  haystack_sessions  list[list[turn]]   turn = {role, content, has_answer?}
  answer_session_ids list[str]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from automem import AutoMemConfig, AutoMemory  # noqa: E402

HERE = os.path.dirname(__file__)
ENV_FILE = os.path.join(HERE, "..", "..", ".env")
DEFAULT_DATASET = os.path.join(HERE, "data", "longmemeval_s")


def load_dataset(path: str | None = None) -> list[dict]:
    with open(path or DEFAULT_DATASET, encoding="utf-8") as f:
        return json.load(f)


def is_abstention(item: dict) -> bool:
    return str(item.get("question_id", "")).endswith("_abs")


def parse_lme_date(ts: str) -> str:
    """'2023/05/07 (Sun) 14:30' -> ISO 8601. Falls back to date-only forms."""
    ts = (ts or "").strip()
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def stratified_subset(data: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """Pick `limit` questions keeping the per-type proportion stable, seeded."""
    import random
    from collections import defaultdict

    if limit <= 0 or limit >= len(data):
        return data
    by_type: dict[str, list[dict]] = defaultdict(list)
    for x in data:
        by_type[x["question_type"]].append(x)
    rng = random.Random(seed)
    picked: list[dict] = []
    for t, items in sorted(by_type.items()):
        k = max(1, round(limit * len(items) / len(data)))
        pool = items[:]
        rng.shuffle(pool)
        picked.extend(pool[:k])
    rng.shuffle(picked)
    return picked[:limit] if len(picked) > limit else picked


def make_memory(
    db_path: str, *, scoring_overrides: dict | None = None, readonly: bool = False
) -> AutoMemory:
    cfg = AutoMemConfig.from_env(env_file=ENV_FILE, db_path=db_path)
    if scoring_overrides:
        for k, v in scoring_overrides.items():
            setattr(cfg.scoring, k, v)
    return AutoMemory(cfg, readonly=readonly)


def config() -> AutoMemConfig:
    return AutoMemConfig.from_env(env_file=ENV_FILE)


def dump_run_config(path: str, am, **extra) -> None:
    """Record the ablation-relevant config of a run next to its output, so a run's
    provenance (reconcile_floor, separate_quota, as_of, ...) is auditable and not
    inferred after the fact."""
    sp = am.config.scoring
    meta = {
        "reconcile_floor": sp.reconcile_floor,
        "reconcile_top_n": sp.reconcile_top_n,
        "superseded_penalty": sp.superseded_penalty,
        "history_mode": am.config.history_mode,
        "history_separate_quota": am.config.history_separate_quota,
        "history_quota": am.config.history_quota,
        "top_k": am.config.top_k,
        "llm_model": am.config.llm_model,
        "embed_model": am.config.embed_model,
        "rerank_model": am.config.rerank_model,
        **extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# The reader prompt mirrors the official LongMemEval QA setup: answer strictly
# from retrieved memories, and say so when the evidence is absent (this is what
# the abstention questions test).
ANSWER_PROMPT = """\
You are a memory assistant. Answer the question using ONLY the retrieved \
memories below. Each memory is prefixed with its date.

# INSTRUCTIONS
1. Use the dated memories to resolve relative time references to absolute dates.
2. Memories under "EARLIER VALUES" are prior values that were later replaced, each \
shown as "[<from> until <until>]" with the date span it was true — they are not the \
current state, but their date ranges ARE valid evidence for past-state and timeline \
questions ("before", "how many days between X and Y"). For a present-tense question, \
rely on the current facts instead.
3. Be concise: give the specific fact, name, date, or number.
4. If verbatim source passages are provided, prefer their exact names, wording, \
and numbers over the paraphrased facts above them.
5. If the memories do not contain enough information to answer, reply exactly: \
"I don't know / no information available".

# RETRIEVED MEMORIES
{memories}

# QUESTION (asked on {question_date})
{question}

Answer:\
"""

# Official LongMemEval uses GPT-4o as judge with type-specific rubrics. We
# reproduce the rubric distinctions (abstention is correct only when the model
# declines; others use generous semantic match) with DeepSeek as judge.
JUDGE_PROMPT_DEFAULT = """\
Decide whether the generated answer is correct given the question and the gold \
answer. Be generous about phrasing: if it conveys the same fact / refers to the \
same entity, date, or quantity as the gold answer, it is CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON: {{"label": "CORRECT" or "WRONG"}}\
"""

JUDGE_PROMPT_ABSTENTION = """\
This question is UNANSWERABLE from the conversation history; a correct system \
must abstain (say it doesn't know / there's no information / it cannot answer). \
Label CORRECT if the generated answer abstains or declines to answer. Label \
WRONG if it fabricates a specific answer.

Question: {question}
Generated answer: {generated_answer}

Return JSON: {{"label": "CORRECT" or "WRONG"}}\
"""


def recall_fields(gold, retrieved) -> dict:
    """Session-level retrieval recall vs LongMemEval's answer_session_ids. Shared by
    the AutoMem and RAG runners so the metric is identical across systems."""
    gold = set(gold or [])
    retrieved = set(retrieved)
    if not gold:  # unanswerable / no evidence labels -> not scored
        return {"recall_at_k": None, "recall_hit": None,
                "n_gold_sessions": 0, "n_retrieved_sessions": len(retrieved)}
    covered = len(retrieved & gold)
    return {
        "recall_at_k": covered / len(gold),
        "recall_hit": 1 if covered else 0,
        "n_gold_sessions": len(gold),
        "n_retrieved_sessions": len(retrieved),
    }


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum(
        min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(pred_tokens)
    )
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
