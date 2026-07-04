"""Summarize a held-out run: per-category answer accuracy for AutoMem / RAG / None
side by side, plus session-level Recall@k (retrieval quality) for AutoMem and RAG —
so retrieval misses are separable from reasoning errors.

  python summarize.py --tag lme_full
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lme_full")
    args = ap.parse_args()

    systems = ["automem", "rag", "none"]
    acc = {}   # system -> {qid: llm_score}
    cat = {}   # qid -> category (abstention bucketed)
    recall = {}  # system -> {qid: row with recall fields}
    for s in systems:
        scored = load(os.path.join(RESULTS, f"{args.tag}_{s}_scored.json"))
        raw = load(os.path.join(RESULTS, f"{args.tag}_{s}.json"))
        if scored:
            acc[s] = {r["question_id"]: r["llm_score"] for r in scored}
            for r in scored:
                cat[r["question_id"]] = "abstention" if r.get("abstention") else r["question_type"]
        if raw:
            recall[s] = raw  # dict keyed by qid

    cats = sorted(set(cat.values()))
    by = lambda d, qids: [d[q] for q in qids if q in d]

    # ---- answer accuracy (3 systems) ----
    print(f"\n{'category':<26}{'n':>4}{'AutoMem':>10}{'RAG':>10}{'None':>10}")
    qids_by_cat = defaultdict(list)
    for q, c in cat.items():
        qids_by_cat[c].append(q)
    tot = {s: [] for s in systems}
    for c in cats:
        qids = qids_by_cat[c]
        line = f"{c:<26}{len(qids):>4}"
        for s in systems:
            vals = by(acc.get(s, {}), qids)
            tot[s].extend(vals)
            line += f"{(sum(vals)/len(vals) if vals else float('nan')):>10.3f}"
        print(line)
    line = f"{'OVERALL':<26}{len(cat):>4}"
    for s in systems:
        v = tot[s]
        line += f"{(sum(v)/len(v) if v else float('nan')):>10.3f}"
    print(line)

    # ---- session-level Recall@k (AutoMem vs RAG; None has no retrieval) ----
    print(f"\nRecall@k (gold answer-session retrieved; questions with evidence labels)")
    print(f"{'category':<26}{'n':>4}{'AM hit':>9}{'AM rec':>9}{'RAG hit':>9}{'RAG rec':>9}")
    for c in cats:
        qids = qids_by_cat[c]
        line = f"{c:<26}"
        scored_n = None
        cells = ""
        for s in ("automem", "rag"):
            rows = recall.get(s, {})
            hits = [rows[q]["recall_hit"] for q in qids if q in rows and rows[q].get("recall_hit") is not None]
            recs = [rows[q]["recall_at_k"] for q in qids if q in rows and rows[q].get("recall_at_k") is not None]
            scored_n = len(hits) if scored_n is None else scored_n
            cells += f"{(sum(hits)/len(hits) if hits else float('nan')):>9.3f}{(sum(recs)/len(recs) if recs else float('nan')):>9.3f}"
        print(f"{c:<26}{scored_n:>4}{cells}")
    # overall recall
    line = f"{'OVERALL':<26}"
    n_lab = None
    cells = ""
    for s in ("automem", "rag"):
        rows = recall.get(s, {})
        hits = [r["recall_hit"] for r in rows.values() if r.get("recall_hit") is not None]
        recs = [r["recall_at_k"] for r in rows.values() if r.get("recall_at_k") is not None]
        n_lab = len(hits) if n_lab is None else n_lab
        cells += f"{(sum(hits)/len(hits) if hits else float('nan')):>9.3f}{(sum(recs)/len(recs) if recs else float('nan')):>9.3f}"
    print(f"{'OVERALL':<26}{n_lab:>4}{cells}")
    print("\nAM hit = fraction of questions whose gold session was in AutoMem top-k;")
    print("AM rec = mean fraction of gold sessions covered. Same for RAG.")


if __name__ == "__main__":
    main()
