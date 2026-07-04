"""Score a LongMemEval results JSON (DeepSeek LLM-judge), per type and overall.

Abstention questions (question_id endswith "_abs") are judged with a dedicated
rubric: correct == the system declined to answer. All others use generous
semantic matching against the gold answer.

Usage:
  python evaluation/longmemeval/judge.py --input ../results/lme_automem.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import (
    JUDGE_PROMPT_ABSTENTION,
    JUDGE_PROMPT_DEFAULT,
    config,
    token_f1,
)

from automem.providers import DeepSeekLLM


def is_abs(row: dict) -> bool:
    return str(row.get("question_id", "")).endswith("_abs")


def judge_one(llm, row: dict) -> dict:
    if is_abs(row):
        prompt = JUDGE_PROMPT_ABSTENTION.format(
            question=row["question"], generated_answer=row["response"]
        )
    else:
        prompt = JUDGE_PROMPT_DEFAULT.format(
            question=row["question"],
            gold_answer=row["answer"],
            generated_answer=row["response"],
        )
    label = llm.complete_json("You label answers. JSON only.", prompt, temperature=0.0)
    correct = 1 if str(label.get("label", "")).upper() == "CORRECT" else 0
    return {
        "question_id": row["question_id"],
        "question_type": row.get("question_type", "?"),
        "abstention": is_abs(row),
        "llm_score": correct,
        "f1_score": 0.0 if is_abs(row) else token_f1(str(row["response"]), str(row["answer"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    rows = list(data.values()) if isinstance(data, dict) else data

    llm = DeepSeekLLM(config())
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        scored = list(ex.map(lambda r: judge_one(llm, r), rows))

    by_type: dict[str, list] = defaultdict(list)
    for s in scored:
        key = "abstention" if s["abstention"] else s["question_type"]
        by_type[key].append(s)

    print(f"\n{'question_type':<26}{'n':>5}{'LLM acc':>10}")
    tl = []
    for t in sorted(by_type):
        items = by_type[t]
        acc = sum(i["llm_score"] for i in items) / len(items)
        tl.extend(i["llm_score"] for i in items)
        print(f"{t:<26}{len(items):>5}{acc:>10.4f}")
    print(f"{'OVERALL':<26}{len(tl):>5}{sum(tl)/len(tl):>10.4f}")

    out_path = args.output or os.path.splitext(args.input)[0] + "_scored.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
