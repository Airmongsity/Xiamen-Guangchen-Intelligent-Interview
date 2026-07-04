"""Ingest LoCoMo conversations into AutoMemory (the ADD phase).

Mirrors mem0/evaluation/src/memzero/add.py: each conversation gets two users
({speaker}_{conv_idx}); each session runs through the extraction pipeline with
the session timestamp backfilled as created_at, which exercises our time-decay
anchor against months-old memories.

Usage:
  python evaluation/locomo/add_runner.py --db results/locomo.db --conv-ids 0,1
"""

from __future__ import annotations

import argparse
import os
import time

from common import (
    iter_sessions,
    load_dataset,
    make_memory,
    parse_locomo_timestamp,
    speaker_user_ids,
)


def add_conversation(am, item: dict, idx: int, *, batch_size: int = 20) -> None:
    conv = item["conversation"]
    user_a, user_b = speaker_user_ids(item, idx)

    for session_key, date_str, chats in iter_sessions(conv):
        created_at = parse_locomo_timestamp(date_str)
        messages = [
            {"role": "user", "content": f"{c['speaker']}: {c['text']}"} for c in chats
        ]
        # both speakers share the session content but store it under their own
        # user_id so per-speaker recall works like mem0's setup
        for user_id in (user_a, user_b):
            for i in range(0, len(messages), batch_size):
                chunk = messages[i : i + batch_size]
                for attempt in range(3):
                    try:
                        events = am.add(
                            chunk,
                            user_id=user_id,
                            conversation_date=created_at[:10],
                            created_at=created_at,
                        )
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        print(f"  retry {session_key} ({e})")
                        time.sleep(2)
                added = sum(1 for e in events if e.event == "ADD")
                print(f"  conv {idx} {user_id} {session_key}: +{added} memories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--db", default="evaluation/results/locomo.db")
    parser.add_argument("--conv-ids", default=None, help="comma-separated subset, e.g. 0,1")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    data = load_dataset(args.dataset)
    conv_ids = (
        [int(x) for x in args.conv_ids.split(",")] if args.conv_ids else range(len(data))
    )

    am = make_memory(args.db)
    t0 = time.time()
    for idx in conv_ids:
        print(f"=== conversation {idx} ===")
        add_conversation(am, data[idx], idx)
    print(f"done in {time.time() - t0:.0f}s; stats: {am.stats()}")
    am.close()


if __name__ == "__main__":
    main()
