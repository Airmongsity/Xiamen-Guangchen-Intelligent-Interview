"""Diagnose ssu/ssp questions that flipped correct(V1)->wrong(V2): is the gold
evidence dropping out of top-30 (dilution), or retrieved-but-answered-wrong
(interference)? Also measures how many top-30 slots superseded history ate."""

from __future__ import annotations

import json
import re
import sys

from common import DEFAULT_DATASET, make_memory

FLIPS = ["66f24dbb", "75f70248", "853b0a1d", "b0479f84"]
V2_DB = "../results/lme3.db"
V1_DB = "../results/lme2.db"


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3}


def gold_evidence(item: dict) -> list[str]:
    """has_answer turn texts from the answer sessions = the gold evidence."""
    ev = []
    ans_ids = set(item.get("answer_session_ids", []))
    sids = item.get("haystack_session_ids", [])
    for i, session in enumerate(item["haystack_sessions"]):
        sid = sids[i] if i < len(sids) else None
        for t in session:
            if t.get("has_answer") or (sid in ans_ids):
                if t.get("content"):
                    ev.append(t["content"])
    return ev


def find_in_db(am, qid, gtoks):
    """Any ingested memory whose content covers the gold tokens (proxy: >=60%)."""
    hits = []
    for m in am.store.list_memories(user_id=qid, limit=10_000):
        mt = toks(m.content)
        if gtoks and len(gtoks & mt) / len(gtoks) >= 0.5:
            hits.append(m)
    return hits


def diagnose(am, item, label):
    qid = item["question_id"]
    q = item["question"]
    gold = item.get("answer", "")
    gtoks = toks(gold)
    ev = gold_evidence(item)
    evtoks = set().union(*[toks(e) for e in ev]) if ev else set()

    res = am.recall(q, user_id=qid, top_k=30, include_short_term=False)
    retrieved = res.long_term
    n_hist = sum(1 for sm in retrieved if sm.memory.valid_to)
    # gold present among retrieved memory contents?
    ret_text = " ".join(sm.memory.content for sm in retrieved)
    ret_src = " ".join(res.sources.values())
    gold_in_ret = gtoks and len(gtoks & toks(ret_text)) / len(gtoks) >= 0.5
    gold_in_src = gtoks and len(gtoks & toks(ret_src)) / len(gtoks) >= 0.5
    # is the gold evidence even in the db (was it ingested)?
    db_hits = find_in_db(am, qid, gtoks if gtoks else evtoks)

    print(f"\n{'='*70}\n[{label}] {qid}  ({item['question_type']})")
    print(f"Q: {q}")
    print(f"GOLD: {gold}")
    print(f"retrieved={len(retrieved)}  superseded_in_top30={n_hist}  "
          f"gold_in_retrieved={gold_in_ret}  gold_in_sources={gold_in_src}")
    print(f"gold-bearing memories in DB (ingested at all): {len(db_hits)}")
    for m in db_hits[:4]:
        rank = next((i for i, sm in enumerate(retrieved) if sm.memory.id == m.id), None)
        tag = f"top30 rank {rank}" if rank is not None else "NOT in top30"
        v = " [SUPERSEDED]" if m.valid_to else ""
        print(f"    - ({tag}){v} {m.content[:110]}")
    if not db_hits:
        print("    (no clearly gold-bearing memory found by token proxy)")


def main():
    data = {x["question_id"]: x for x in json.load(open(DEFAULT_DATASET, encoding="utf-8"))}
    v2res = json.load(open("../results/lme3_automem_k30.json", encoding="utf-8"))
    v1res = json.load(open("../results/lme2_automem_k30.json", encoding="utf-8"))
    am2 = make_memory(V2_DB)
    for qid in FLIPS:
        item = data[qid]
        print(f"\n#### {qid}: V1 ans = {v1res[qid]['response'][:140]!r}")
        print(f"####        V2 ans = {v2res[qid]['response'][:140]!r}")
        diagnose(am2, item, "V2")
    am2.close()


if __name__ == "__main__":
    main()
