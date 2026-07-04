"""Measure the SUPERSESSION TRIGGER RATE as a function of reconcile_floor, on a
fixed set of knowledge-update haystacks. The point: raising the floor to save
cost must not quietly suppress the bi-temporal mechanism (DELETE/UPDATE that
invalidate an old value). We compare floors directly on the mechanism, not on the
overfit-prone dev-set overall.

  python probe_supersession.py --n 5 --floors 0.4,0.65,0.8
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import load_dataset, make_memory, parse_lme_date

from automem.providers.llm import DeepSeekLLM


def pick_ku(n: int) -> list[dict]:
    data = load_dataset()
    ku = sorted(
        (x for x in data if x["question_type"] == "knowledge-update"),
        key=lambda x: x["question_id"],
    )
    return ku[:n]


def run_floor(items: list[dict], floor: float) -> dict:
    db = os.path.join(tempfile.gettempdir(), f"automem_probe_{floor}.db")
    for ext in ("", "-wal", "-shm"):
        try:
            os.remove(db + ext)
        except OSError:
            pass
    am = make_memory(db, scoring_overrides={"reconcile_floor": floor})
    ops: Counter = Counter()

    def ingest_one(it) -> Counter:
        # sessions sequential per question (supersession is order-dependent)
        local: Counter = Counter()
        dates = it.get("haystack_dates", [])
        for i, session in enumerate(it["haystack_sessions"]):
            msgs = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                    for t in session if t.get("content")]
            if not msgs:
                continue
            ds = dates[i] if i < len(dates) else ""
            try:
                for ev in am.add(msgs, user_id=it["question_id"], infer=True,
                                 conversation_date=ds or None, created_at=parse_lme_date(ds)):
                    local[ev.event] += 1
            except Exception:
                local["FAILED"] += 1
        return local

    with ThreadPoolExecutor(max_workers=len(items)) as ex:  # parallel across questions
        for local in ex.map(ingest_one, items):
            ops.update(local)
    # supersession footprint in the final store: rows stamped with valid_to
    superseded = sum(
        len([m for m in am.store.list_memories(user_id=it["question_id"], limit=100000)
             if m.valid_to])
        for it in items
    )
    contradicts = am.store.conn.execute(
        "SELECT COUNT(*) FROM memory_links WHERE link_kind='contradicts'"
    ).fetchone()[0]
    am.close()
    return {"ops": ops, "superseded": superseded, "contradicts": contradicts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--floors", default="0.4,0.65,0.8")
    args = ap.parse_args()
    DeepSeekLLM.track_usage = True
    items = pick_ku(args.n)
    print("probe questions:", [it["question_id"] for it in items])

    rows = []
    for f in [float(x) for x in args.floors.split(",")]:
        r = run_floor(items, f)
        rows.append((f, r))

    print(f"\n{'floor':>6}{'ADD':>7}{'UPDATE':>8}{'DELETE':>8}{'NONE':>7}"
          f"{'superseded':>12}{'contradicts':>13}")
    for f, r in rows:
        o = r["ops"]
        print(f"{f:>6}{o['ADD']:>7}{o['UPDATE']:>8}{o['DELETE']:>8}{o['NONE']:>7}"
              f"{r['superseded']:>12}{r['contradicts']:>13}")
    print("\nUPDATE+DELETE = supersession triggers; 'superseded' = rows left with valid_to.")
    print("If 0.65 holds these near 0.4, the floor is recall-safe for knowledge-update.")


if __name__ == "__main__":
    main()
