"""Pin down the evidence rank for the two genuine retrieval misses."""
from __future__ import annotations
import json, re
from common import DEFAULT_DATASET, make_memory

DB = "../results/lme3.db"
PROBES = {
    "853b0a1d": ["necklace", "grandma", "grandmother", "birthday", "silver", "18"],
    "75f70248": ["luna", "cat", "shed", "dust", "clean", "vacuum", "living room", "sneez"],
}


def main():
    data = {x["question_id"]: x for x in json.load(open(DEFAULT_DATASET, encoding="utf-8"))}
    am = make_memory(DB)
    for qid, terms in PROBES.items():
        item = data[qid]
        q = item["question"]
        res = am.recall(q, user_id=qid, top_k=30, include_short_term=False)
        ranks = {sm.memory.id: i for i, sm in enumerate(res.long_term)}
        print(f"\n{'='*70}\n{qid}: {q}")
        print(f"top30 superseded count: {sum(1 for sm in res.long_term if sm.memory.valid_to)}")
        # all db memories mentioning any probe term
        rx = re.compile("|".join(re.escape(t) for t in terms), re.I)
        hits = [m for m in am.store.list_memories(user_id=qid, limit=10_000) if rx.search(m.content)]
        print(f"{len(hits)} memories in DB mention probe terms; their top-30 ranks:")
        for m in hits:
            r = ranks.get(m.id)
            tag = f"rank {r}" if r is not None else "—— OUT of top30"
            v = " [SUPERSEDED]" if m.valid_to else ""
            print(f"  [{tag:<14}]{v} {m.content[:120]}")
    am.close()


if __name__ == "__main__":
    main()
