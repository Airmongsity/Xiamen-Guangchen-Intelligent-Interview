"""Measure the real ingest cost drivers: per-call breakdown (extraction vs
reconcile), DeepSeek cache-hit rate, and projected full-500 cost — at a given
reconcile_floor, so we can see what actually drives the bill.

  python measure_cost.py --n 3 --floor 0.4
  python measure_cost.py --n 3 --floor 0.65
"""

from __future__ import annotations

import argparse
import os
import tempfile

from common import make_memory, parse_lme_date, stratified_subset, load_dataset

from automem.providers.llm import DeepSeekLLM

# approximate DeepSeek deepseek-chat rates (USD per 1M tokens); adjust to current
RATE_MISS = 0.27   # input, cache miss
RATE_HIT = 0.07    # input, cache hit
RATE_OUT = 1.10    # output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--floor", type=float, default=0.4)
    args = ap.parse_args()

    DeepSeekLLM.track_usage = True
    db = os.path.join(tempfile.gettempdir(), f"automem_cost_{args.floor}.db")
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass
    am = make_memory(db, scoring_overrides={"reconcile_floor": args.floor})

    data = stratified_subset(load_dataset(), args.n)
    n_sessions = 0
    for it in data:
        dates = it.get("haystack_dates", [])
        for i, session in enumerate(it["haystack_sessions"]):
            msgs = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                    for t in session if t.get("content")]
            if not msgs:
                continue
            n_sessions += 1
            ds = dates[i] if i < len(dates) else ""
            try:
                am.add(msgs, user_id=it["question_id"], infer=True,
                       conversation_date=ds or None, created_at=parse_lme_date(ds))
            except Exception as e:
                print("  ! session failed:", type(e).__name__)

    usage = am._llm.usage if am._llm else {}
    print(f"\n=== floor={args.floor}  questions={len(data)}  sessions={n_sessions} ===")
    tot_calls = tot_in = tot_hit = tot_out = 0
    cost = 0.0
    for tag, b in sorted(usage.items()):
        miss = b["prompt"] - b["cache_hit"]
        c = miss / 1e6 * RATE_MISS + b["cache_hit"] / 1e6 * RATE_HIT + b["completion"] / 1e6 * RATE_OUT
        cost += c
        hit_rate = b["cache_hit"] / b["prompt"] * 100 if b["prompt"] else 0
        print(f"  [{tag:<33}] calls={b['calls']:>4}  in={b['prompt']:>8}  "
              f"cache_hit={hit_rate:4.0f}%  out={b['completion']:>7}  ${c:.4f}")
        tot_calls += b["calls"]; tot_in += b["prompt"]; tot_hit += b["cache_hit"]; tot_out += b["completion"]
    overall_hit = tot_hit / tot_in * 100 if tot_in else 0
    print(f"  TOTAL calls={tot_calls}  in={tot_in}  cache_hit={overall_hit:.0f}%  out={tot_out}  ${cost:.4f}")
    per_q = cost / len(data)
    print(f"  per-question ${per_q:.4f}  ->  projected full-500 ingest  ${per_q*500:.2f}")
    am.close()


if __name__ == "__main__":
    main()
