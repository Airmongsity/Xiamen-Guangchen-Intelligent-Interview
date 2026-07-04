"""Shared helpers for the LoCoMo evaluation runners."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from automem import AutoMemConfig, AutoMemory  # noqa: E402

DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "powermem", "benchmark", "locomo", "dataset", "locomo10.json",
)


def load_dataset(path: str | None = None) -> list[dict]:
    with open(path or DEFAULT_DATASET, encoding="utf-8") as f:
        return json.load(f)


def parse_locomo_timestamp(ts: str) -> str:
    """LoCoMo session timestamps look like '1:56 pm on 8 May, 2023'."""
    try:
        dt = datetime.strptime(ts.strip(), "%I:%M %p on %d %B, %Y")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return datetime.now(timezone.utc).isoformat()


def speaker_user_ids(item: dict, idx: int) -> tuple[str, str]:
    conv = item["conversation"]
    return f"{conv['speaker_a']}_{idx}", f"{conv['speaker_b']}_{idx}"


def iter_sessions(conversation: dict):
    """Yield (session_key, date_time_str, chats) in order."""
    n = 1
    while f"session_{n}" in conversation:
        chats = conversation[f"session_{n}"]
        if chats:
            yield f"session_{n}", conversation.get(f"session_{n}_date_time", ""), chats
        n += 1


def make_memory(db_path: str, *, scoring_overrides: dict | None = None) -> AutoMemory:
    cfg = AutoMemConfig.from_env(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        db_path=db_path,
    )
    if scoring_overrides:
        for k, v in scoring_overrides.items():
            setattr(cfg.scoring, k, v)
    return AutoMemory(cfg)


ANSWER_PROMPT = """\
You are an intelligent memory assistant answering a question from retrieved \
conversation memories of two speakers.

# INSTRUCTIONS:
1. Analyze the timestamped memories from both speakers carefully.
2. Convert relative time references to absolute dates using the memory \
timestamps (a memory from 4 May 2022 saying "last year" means 2021).
3. If memories contradict, trust the most recent one.
4. Answer in fewer than 6 words; give the specific fact, date, or detail.
5. If the memories truly contain no answer, reply "No information available".

Memories for {speaker_1}:
{speaker_1_memories}

Memories for {speaker_2}:
{speaker_2_memories}

Question: {question}
Answer:\
"""

JUDGE_PROMPT = """\
Your task is to label a generated answer as CORRECT or WRONG given a question \
and the gold answer. Be generous: if the generated answer touches the same \
topic / refers to the same date or time period as the gold answer, label it \
CORRECT even if phrased differently or longer.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON: {{"label": "CORRECT" or "WRONG"}}\
"""


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    common = {}
    for t in pred_tokens:
        if t in gold_tokens:
            common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
