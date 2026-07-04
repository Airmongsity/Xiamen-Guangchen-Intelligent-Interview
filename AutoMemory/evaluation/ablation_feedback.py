"""Feedback-loop ablation on LoCoMo.

Per conversation the QA set is split 50/50 into train/test. Phase 1 answers the
train questions, judges them, and feeds the outcome back via report_outcome —
writing utility into the memories each variant is allowed to use. Phase 2
answers the held-out test questions and reports accuracy.

Variants:
  full         everything on
  no_feedback  beta=0, b=0  (utility ignored in ranking and decay)
  no_spread    gamma=0      (activation spreading off)
  no_decay     lam=1        (time decay off)

Each variant works on its own copy of the ingested DB (phase 1 mutates state).

Usage:
  python evaluation/ablation_feedback.py --db evaluation/results/locomo.db \
      --conv-ids 0,1 --variants full,no_feedback
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "locomo"))

from common import JUDGE_PROMPT, load_dataset, make_memory, speaker_user_ids  # noqa: E402
from search_runner import answer_question, conversation_as_of  # noqa: E402

from automem.providers import DeepSeekLLM  # noqa: E402

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_feedback": {"beta": 0.0, "b": 0.0},
    "no_spread": {"gamma": 0.0},
    "no_decay": {"lam": 1.0},
}


def judge_correct(llm, row: dict) -> bool:
    data = llm.complete_json(
        "You label answers. JSON only.",
        JUDGE_PROMPT.format(
            question=row["question"],
            gold_answer=row["answer"],
            generated_answer=row["response"],
        ),
        temperature=0.0,
    )
    return str(data.get("label", "")).upper() == "CORRECT"


def run_variant(
    variant: str, base_db: str, data: list, conv_ids: list[int],
    *, top_k: int, workers: int, seed: int,
) -> dict:
    db = os.path.join(os.path.dirname(base_db), f"ablation_{variant}.db")
    shutil.copyfile(base_db, db)
    for suffix in ("-wal", "-shm"):
        try:
            os.remove(db + suffix)
        except FileNotFoundError:
            pass

    am = make_memory(db, scoring_overrides=VARIANTS[variant])
    llm = DeepSeekLLM(am.config)
    rng = random.Random(seed)
    train_results, test_results = [], []

    for idx in conv_ids:
        item = data[idx]
        user_a, user_b = speaker_user_ids(item, idx)
        as_of = conversation_as_of(item)
        qa = [q for q in item["qa"] if int(q.get("category", -1)) != 5]
        qa = sorted(qa, key=lambda q: q["question"])
        rng.shuffle(qa)
        split = len(qa) // 2
        train, test = qa[:split], qa[split:]

        # Phase 1: answer train questions, judge, write feedback. log=True so the
        # retrieval rows exist for report_outcome; touch=False so ONLY the utility
        # feedback (not access reinforcement) changes between phase 1 and phase 2.
        def phase1(q):
            row = answer_question(am, llm, q, user_a, user_b, top_k, as_of=as_of, log=True)
            correct = judge_correct(llm, row)
            quality = 1.0 if correct else 0.2
            for rid in row["retrieval_ids"]:
                try:
                    am.report_outcome(retrieval_id=rid, quality=quality)
                except KeyError:
                    pass
            row["llm_score"] = int(correct)
            return row

        with ThreadPoolExecutor(max_workers=workers) as ex:
            train_results.extend(ex.map(phase1, train))

        # Phase 2: held-out test questions, read-only (no feedback written)
        def phase2(q):
            row = answer_question(am, llm, q, user_a, user_b, top_k, as_of=as_of)
            row["llm_score"] = int(judge_correct(llm, row))
            return row

        with ThreadPoolExecutor(max_workers=workers) as ex:
            test_results.extend(ex.map(phase2, test))
        print(f"  [{variant}] conv {idx} done")

    am.close()

    def acc(rows):
        return sum(r["llm_score"] for r in rows) / len(rows) if rows else 0.0

    return {
        "variant": variant,
        "train_n": len(train_results),
        "train_acc": acc(train_results),
        "test_n": len(test_results),
        "test_acc": acc(test_results),
        "test_rows": test_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db", required=True, help="DB produced by add_runner.py")
    parser.add_argument("--conv-ids", default=None)
    parser.add_argument("--variants", default="full,no_feedback")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="evaluation/results/ablation.json")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    conv_ids = (
        [int(x) for x in args.conv_ids.split(",")]
        if args.conv_ids
        else list(range(len(data)))
    )
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")

    summary = []
    for variant in variants:
        print(f"=== variant: {variant} ===")
        result = run_variant(
            variant, args.db, data, conv_ids,
            top_k=args.top_k, workers=args.workers, seed=args.seed,
        )
        summary.append(result)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'variant':<14}{'train acc':>11}{'test acc':>10}{'test n':>8}")
    for r in summary:
        print(f"{r['variant']:<14}{r['train_acc']:>11.4f}{r['test_acc']:>10.4f}{r['test_n']:>8}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
