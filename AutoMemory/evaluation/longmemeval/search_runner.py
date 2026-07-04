"""Answer LongMemEval questions from AutoMemory recall (the SEARCH phase).

Reads the manifest written by add_runner so it scores exactly the ingested
subset. Output is judge.py-compatible: {question_id: row}.

Usage:
  python evaluation/longmemeval/search_runner.py --db ../results/lme.db \
      --out ../results/lme_automem.json --top-k 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from common import (
    ANSWER_PROMPT, dump_run_config, load_dataset, make_memory, parse_lme_date,
    recall_fields,
)

from automem.providers import DeepSeekLLM


def format_memories(result) -> str:
    if not result.long_term and not result.history:
        return "(no memories retrieved)"
    lines = []
    for sm in result.long_term:
        m = sm.memory
        src = f" {{src:{m.source_id[:6]}}}" if m.source_id else ""
        # only happens in the compete ablation; normally history is split out below
        tag = f" (SUPERSEDED, not current, until {m.valid_to[:10]})" if m.valid_to else ""
        lines.append(f"- [{m.created_at[:10]}]{tag}{src} {m.content}")
    if result.history:
        # only reached under the experimental --separate-quota ablation (default
        # compete config leaves result.history empty).
        # superseded earlier values, on a separate quota; NOT current. The explicit
        # [SUPERSEDED valid_from -> valid_to] marker preserves the date anchors that
        # timeline questions need, while signalling loudly that the value is stale so
        # it isn't used to answer a present-tense question.
        # Section-level usage instruction (NOT a per-line pejorative): the lines stay
        # neutral dated intervals so timeline questions keep their date anchors, while
        # the header carries the "not current" guidance once, for both reading modes.
        lines.append(
            "\n# EARLIER VALUES — these are prior values that were later replaced, "
            "shown with the date range each was true. NOT the current state. For a "
            "past-state or timeline question ('before', 'how many days between'), use "
            "a line's date range as evidence; for a present-tense question, use the "
            "current facts above instead."
        )
        for sm in result.history:
            m = sm.memory
            frm = (m.valid_from or m.created_at or "")[:10]
            to = (m.valid_to or "")[:10]
            src = f" {{src:{m.source_id[:6]}}}" if m.source_id else ""
            lines.append(f"- [{frm} until {to}]{src} {m.content}")
    if result.sources:
        # hybrid granularity: verbatim passages for exact names / wording / dates.
        lines.append("\n# VERBATIM SOURCE PASSAGES (for exact wording)")
        for sid, text in result.sources.items():
            lines.append(f"[src:{sid[:6]}]\n{text}")
    return "\n".join(lines)


def recall_at_k(res, item: dict) -> dict:
    """Retrieval Recall@k against LongMemEval's evidence labels: did the top-k
    retrieved memories come from the gold answer session(s)? Separates retrieval
    misses from reasoning errors. Uses the session_id tagged onto each memory at
    ingest. Returns nulls when the question carries no answer_session_ids."""
    retrieved = set()
    for sm in list(res.long_term) + list(getattr(res, "history", [])):
        sid = (sm.memory.metadata or {}).get("session_id")
        if sid:
            retrieved.add(sid)
    return recall_fields(item.get("answer_session_ids"), retrieved)


def answer(am, llm, item: dict, top_k: int, history_mode: str = "auto",
           use_asof: bool = True) -> dict:
    qid = item["question_id"]
    question = item["question"]
    t0 = time.time()
    # score retention as of when the question was asked (~2023), not wall-clock now
    as_of = (
        parse_lme_date(item.get("question_date", ""))
        if use_asof and item.get("question_date") else None
    )
    res = am.recall(
        question, user_id=qid, top_k=top_k,
        include_short_term=False, history_mode=history_mode, as_of=as_of,
        touch=False, log=False,  # read-only: evaluation must not mutate memory state
    )
    search_time = time.time() - t0
    recall = recall_at_k(res, item)
    prompt = ANSWER_PROMPT.format(
        memories=format_memories(res),
        question_date=item.get("question_date", ""),
        question=question,
    )
    response = llm.complete(
        "You answer questions strictly from the provided memories.",
        prompt,
        temperature=0.0,
    )
    return {
        "question_id": qid,
        "question_type": item["question_type"],
        "question": question,
        "answer": item.get("answer", ""),
        "response": response.strip(),
        "n_retrieved": len(res.long_term),
        "search_time": search_time,
        **recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db", default="evaluation/results/lme.db")
    parser.add_argument("--out", default="evaluation/results/lme_automem.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--history-mode", choices=["auto", "on", "off"], default="auto",
        help="superseded-history quota: off | on (always) | auto (query-aware gate)",
    )
    parser.add_argument(
        "--separate-quota", action="store_true",
        help="ablation: segregate history into its own quota + gate (default: compete in-pool)",
    )
    parser.add_argument(
        "--no-asof", action="store_true",
        help="ablation: score retention at wall-clock now instead of question_date",
    )
    args = parser.parse_args()

    data = {x["question_id"]: x for x in load_dataset(args.dataset)}
    manifest = args.manifest or os.path.splitext(args.db)[0] + "_manifest.json"
    with open(manifest, encoding="utf-8") as f:
        qids = json.load(f)
    items = [data[q] for q in qids if q in data]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # read-only store: any stray write (forgotten touch/log) raises instead of
    # silently polluting the measured state. as_of is also enforced (see recall).
    am = make_memory(args.db, readonly=True)
    if args.separate_quota:
        am.config.history_separate_quota = True  # ablation: segregated quota + gate
    dump_run_config(
        os.path.splitext(args.out)[0] + "_config.json", am,
        phase="search", top_k=args.top_k, use_asof=not args.no_asof,
        history_mode=args.history_mode,
    )
    llm = DeepSeekLLM(am.config)
    print(
        f"answering {len(items)} questions (top_k={args.top_k}, "
        f"{args.workers} workers, history={args.history_mode}, "
        f"separate_quota={am.config.history_separate_quota})..."
    )
    use_asof = not args.no_asof
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(
            ex.map(
                lambda it: answer(am, llm, it, args.top_k, args.history_mode, use_asof),
                items,
            )
        )
    out = {r["question_id"]: r for r in rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {args.out}")
    am.close()


if __name__ == "__main__":
    main()
