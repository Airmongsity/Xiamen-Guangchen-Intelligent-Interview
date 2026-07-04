#!/usr/bin/env bash
# A/B: old extraction prompt vs new (mem0-style) extraction prompt, on the same
# seeded 60-q dev slice, same top-k, both DeepSeek. Only ingest (add_runner) uses
# the extraction prompt, so we revert prompts.py to HEAD (old) for just the old
# ingest via `git stash`, then restore the working-tree (new) prompt immediately.
set -u

ROOT="D:/Users/cjy24/Documents/autoMem"
LME="$ROOT/evaluation/longmemeval"
RES="$ROOT/evaluation/results"
LIMIT=60
TOPK=30
WORKERS=8

cd "$LME"

echo "=== A/B dev run (limit=$LIMIT top_k=$TOPK) $(date) ==="

# ---- OLD prompt arm: stash the prompts.py edit so the tree is at HEAD ----
echo "--- [OLD] reverting prompts.py to HEAD and ingesting ---"
git -C "$ROOT" stash push -- automem/extraction/prompts.py
python add_runner.py --db "$RES/dev_ab_old.db" --limit "$LIMIT" --workers "$WORKERS"
rc=$?
# ALWAYS restore the new prompt, even if ingest failed
git -C "$ROOT" stash pop
if [ $rc -ne 0 ]; then echo "OLD ingest failed (rc=$rc)"; exit $rc; fi

set -e
python search_runner.py --db "$RES/dev_ab_old.db" --out "$RES/dev_ab_old_automem.json" --top-k "$TOPK" --workers "$WORKERS"
python judge.py --input "$RES/dev_ab_old_automem.json"

# ---- NEW prompt arm: working tree already has the new prompt ----
echo "--- [NEW] ingesting with the mem0-style prompt ---"
python add_runner.py --db "$RES/dev_ab_new.db" --limit "$LIMIT" --workers "$WORKERS"
python search_runner.py --db "$RES/dev_ab_new.db" --out "$RES/dev_ab_new_automem.json" --top-k "$TOPK" --workers "$WORKERS"
python judge.py --input "$RES/dev_ab_new_automem.json"

# ---- compare ----
python - "$RES/dev_ab_old_automem_scored.json" "$RES/dev_ab_new_automem_scored.json" <<'PY'
import json, sys
from collections import defaultdict

def load(p):
    rows = json.load(open(p, encoding="utf-8"))
    by = defaultdict(list)
    for r in rows:
        key = "abstention" if r["abstention"] else r["question_type"]
        by[key].append(r["llm_score"])
    return by

old, new = load(sys.argv[1]), load(sys.argv[2])
types = sorted(set(old) | set(new))
print(f"\n{'question_type':<26}{'n':>4}{'OLD':>9}{'NEW':>9}{'delta':>9}")
def acc(xs): return sum(xs)/len(xs) if xs else float('nan')
allo, alln = [], []
for t in types:
    o, n = old.get(t, []), new.get(t, [])
    allo += o; alln += n
    print(f"{t:<26}{len(n):>4}{acc(o):>9.4f}{acc(n):>9.4f}{acc(n)-acc(o):>+9.4f}")
print(f"{'OVERALL':<26}{len(alln):>4}{acc(allo):>9.4f}{acc(alln):>9.4f}{acc(alln)-acc(allo):>+9.4f}")
PY

echo "=== A/B done $(date) ==="
