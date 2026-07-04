"""Score a results JSON (token F1 + DeepSeek LLM-judge), per category and overall.

Self-contained replacement for mem0/evaluation/evals.py + metrics/llm_judge.py
so no OpenAI key is needed. The input format is identical, so mem0's scripts
also work on our results files.

Usage:
  python evaluation/locomo/judge.py --input results/automem_results.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import JUDGE_PROMPT, token_f1

from automem import AutoMemConfig
from automem.providers import DeepSeekLLM


def judge_one(llm, row: dict) -> dict:
    label_data = llm.complete_json(
        "You label answers. JSON only.",
        JUDGE_PROMPT.format(
            question=row["question"],
            gold_answer=row["answer"],
            generated_answer=row["response"],
        ),
        temperature=0.0,
    )
    llm_score = 1 if str(label_data.get("label", "")).upper() == "CORRECT" else 0
    return {
        "question": row["question"],
        "answer": row["answer"],
        "response": row["response"],
        "category": str(row["category"]),
        "f1_score": token_f1(str(row["response"]), str(row["answer"])),
        "llm_score": llm_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    llm = DeepSeekLLM(AutoMemConfig.from_env(env_file=env_file))

    rows = [
        row
        for conv_rows in data.values()
        for row in conv_rows
        if str(row.get("category")) != "5"
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        scored = list(ex.map(lambda r: judge_one(llm, r), rows))

    by_cat: dict[str, list] = defaultdict(list)
    for s in scored:
        by_cat[s["category"]].append(s)

    print(f"\n{'category':<10}{'n':>5}{'LLM acc':>10}{'F1':>8}")
    total_llm, total_f1 = [], []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        llm_acc = sum(i["llm_score"] for i in items) / len(items)
        f1 = sum(i["f1_score"] for i in items) / len(items)
        total_llm.extend(i["llm_score"] for i in items)
        total_f1.extend(i["f1_score"] for i in items)
        print(f"{cat:<10}{len(items):>5}{llm_acc:>10.4f}{f1:>8.4f}")
    print(
        f"{'overall':<10}{len(total_llm):>5}"
        f"{sum(total_llm) / len(total_llm):>10.4f}"
        f"{sum(total_f1) / len(total_f1):>8.4f}"
    )

    out_path = args.output or os.path.splitext(args.input)[0] + "_scored.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
