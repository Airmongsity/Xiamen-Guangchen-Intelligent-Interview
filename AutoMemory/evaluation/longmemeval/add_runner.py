"""Ingest LongMemEval haystacks into AutoMemory (the ADD phase).

Each question carries its OWN haystack of ~50 sessions, so every question gets
its own user_id namespace (= question_id) inside one shared DB. Sessions within
a question are ingested sequentially (dedup/reconcile is order-sensitive);
different questions run in parallel.

Usage:
  python evaluation/longmemeval/add_runner.py --db ../results/lme.db \
      --limit 60 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import dump_run_config, load_dataset, make_memory, parse_lme_date, stratified_subset


def ingest_question(am, item: dict) -> tuple[str, int, int]:
    qid = item["question_id"]
    sessions = item["haystack_sessions"]
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    n_mem = 0
    failures = 0
    for i, session in enumerate(sessions):
        msgs = [
            {"role": t.get("role", "user"), "content": t.get("content", "")}
            for t in session
            if t.get("content")
        ]
        if not msgs:
            continue
        date_str = dates[i] if i < len(dates) else ""
        iso = parse_lme_date(date_str)
        # tag every memory with the session it came from, so the search phase can
        # score retrieval Recall@k against LongMemEval's answer_session_ids labels
        sid = sids[i] if i < len(sids) else None
        try:
            events = am.add(
                msgs,
                user_id=qid,
                infer=True,
                conversation_date=date_str or None,
                created_at=iso,
                metadata={"session_id": sid} if sid else None,
            )
            n_mem += len(events)
        except Exception as e:  # one bad session must not abort the whole run
            failures += 1
            print(f"  ! {qid} session {i} failed: {type(e).__name__}: {e}")
    return qid, n_mem, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db", default="evaluation/results/lme.db")
    parser.add_argument("--limit", type=int, default=60, help="stratified subset; 0=all 500")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", default=None, help="where to write ingested question_ids")
    parser.add_argument(
        "--types", default=None,
        help="comma-separated question_type filter (e.g. knowledge-update)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip questions already in the manifest (continue a long/crashed run)",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        data = [x for x in data if x["question_type"] in wanted]
    subset = stratified_subset(data, args.limit)
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    manifest = args.manifest or os.path.splitext(args.db)[0] + "_manifest.json"

    # resume: a fully-ingested question is recorded in the manifest only after it
    # completes, so skipping those lets a crashed multi-hour run continue without
    # re-ingesting (and re-duplicating) what already landed.
    ingested: list[str] = []
    if args.resume and os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            ingested = list(json.load(f))
        done_set = set(ingested)
        before = len(subset)
        subset = [it for it in subset if it["question_id"] not in done_set]
        print(f"resume: {len(ingested)} already done, {len(subset)}/{before} remaining")

    am = make_memory(args.db)
    dump_run_config(
        os.path.splitext(args.db)[0] + "_config.json", am,
        phase="ingest", limit=args.limit, n_questions=len(subset) + len(ingested),
    )
    print(f"ingesting {len(subset)} questions x ~50 sessions ({args.workers} workers)...")
    print(f"  reconcile_floor={am.config.scoring.reconcile_floor} "
          f"separate_quota={am.config.history_separate_quota}")
    t0 = time.time()
    done = 0
    total_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(ingest_question, am, it): it["question_id"] for it in subset}
        for fut in as_completed(futures):
            try:
                qid, n_mem, failures = fut.result()
            except Exception as e:  # whole-question failure: log, keep going
                qid = futures[fut]
                print(f"[skip] {qid} crashed: {type(e).__name__}: {e}")
                done += 1
                continue
            ingested.append(qid)
            total_fail += failures
            done += 1
            tag = f"  ({failures} session fails)" if failures else ""
            print(f"[{done}/{len(subset)}] {qid}: {n_mem} memories{tag}  ({time.time()-t0:.0f}s)")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(sorted(ingested), f, indent=2)
    if total_fail:
        print(f"WARNING: {total_fail} sessions failed across the run")

    print(f"done in {time.time()-t0:.0f}s; manifest -> {manifest}")
    print("stats:", am.store.stats())
    am.close()


if __name__ == "__main__":
    main()
