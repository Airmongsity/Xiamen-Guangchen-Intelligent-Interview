"""Answer LoCoMo questions from AutoMemory recall (the SEARCH phase).

Output JSON is compatible with mem0/evaluation/evals.py:
  {conv_idx: [{question, answer, response, category, evidence, ...}]}

Usage:
  python evaluation/locomo/search_runner.py --db results/locomo.db \
      --out results/automem_results.json --conv-ids 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from common import (
    ANSWER_PROMPT, iter_sessions, load_dataset, make_memory,
    parse_locomo_timestamp, speaker_user_ids,
)

from automem.providers import DeepSeekLLM


def conversation_as_of(item: dict) -> str | None:
    """Latest session timestamp of a conversation — the moment 'now' for scoring
    retention, instead of wall-clock (which would collapse all retention to ~0)."""
    dates = [parse_locomo_timestamp(dt) for _, dt, _ in iter_sessions(item["conversation"]) if dt]
    return max(dates) if dates else None


def format_memories(result) -> str:
    return json.dumps(
        [
            f"{sm.memory.created_at[:10]}: {sm.memory.content}"
            for sm in result.long_term
        ],
        ensure_ascii=False,
        indent=2,
    )


def answer_question(
    am, llm, item_qa: dict, user_a: str, user_b: str, top_k: int,
    *, as_of: str | None = None, touch: bool = False, log: bool = False,
) -> dict:
    question = item_qa["question"]
    t0 = time.time()
    # Read-only by default (touch=False: do NOT reinforce memories during eval;
    # log=False) and retention scored as-of the conversation date. The feedback
    # ablation overrides log=True because report_outcome needs the retrieval rows.
    kw = dict(top_k=top_k, include_short_term=False, as_of=as_of, touch=touch, log=log)
    res_a = am.recall(question, user_id=user_a, **kw)
    res_b = am.recall(question, user_id=user_b, **kw)
    search_time = time.time() - t0

    prompt = ANSWER_PROMPT.format(
        speaker_1=user_a.rsplit("_", 1)[0],
        speaker_2=user_b.rsplit("_", 1)[0],
        speaker_1_memories=format_memories(res_a),
        speaker_2_memories=format_memories(res_b),
        question=question,
    )
    t1 = time.time()
    response = llm.complete("You answer questions from memories.", prompt, temperature=0.0)
    return {
        "question": question,
        "answer": item_qa.get("answer", item_qa.get("adversarial_answer", "")),
        "category": item_qa.get("category", -1),
        "evidence": item_qa.get("evidence", []),
        "response": response.strip(),
        "speaker_1_memories": [sm.memory.content for sm in res_a.long_term],
        "speaker_2_memories": [sm.memory.content for sm in res_b.long_term],
        "retrieval_ids": [res_a.retrieval_id, res_b.retrieval_id],
        "search_time": search_time,
        "response_time": time.time() - t1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db", default="evaluation/results/locomo.db")
    parser.add_argument("--out", default="evaluation/results/automem_results.json")
    parser.add_argument("--conv-ids", default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    conv_ids = (
        [int(x) for x in args.conv_ids.split(",")] if args.conv_ids else range(len(data))
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    am = make_memory(args.db)
    llm = DeepSeekLLM(am.config)
    results: dict[str, list] = {}

    for idx in conv_ids:
        item = data[idx]
        user_a, user_b = speaker_user_ids(item, idx)
        as_of = conversation_as_of(item)
        qa = [q for q in item["qa"] if int(q.get("category", -1)) != 5]
        print(f"=== conversation {idx}: {len(qa)} questions ===")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            conv_results = list(
                ex.map(
                    lambda q: answer_question(
                        am, llm, q, user_a, user_b, args.top_k, as_of=as_of
                    ),
                    qa,
                )
            )
        results[str(idx)] = conv_results
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved {args.out}")

    am.close()


if __name__ == "__main__":
    main()
