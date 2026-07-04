"""Unattended held-out run: ingest once, then evaluate all THREE systems on the
SAME question set with the SAME reader prompt and judge — AutoMem, naive RAG, and
no-memory — so the numbers are directly comparable. Each search phase also records
per-question Recall@k against the evidence labels.

Phases (sequential, each logged with timestamps; ingest is --resume-able):
  1. ingest        add_runner  -> {tag}.db (+ {tag}_manifest.json, session-tagged)
  2. automem       search_runner -> {tag}_automem.json   (Recall@k)
  3. rag           baseline_runner rag -> {tag}_rag.json (Recall@k, low concurrency)
  4. none          baseline_runner none -> {tag}_none.json
  5. judge x3      judge.py on each output

  python run_full.py --tag lme_full --limit 150 --workers 8
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def phase(name: str, cmd: list[str], required: bool = True) -> bool:
    log(f"START {name}: {' '.join(cmd)}")
    r = subprocess.run([sys.executable, *cmd], cwd=HERE)
    if r.returncode != 0:
        log(f"FAILED {name} (exit {r.returncode})")
        if required:
            sys.exit(r.returncode)
        return False
    log(f"DONE {name}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lme_full")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rag-workers", type=int, default=2, help="RAG re-embeds every turn; keep low for TPM")
    args = ap.parse_args()

    def p(name: str) -> str:
        return os.path.join(RESULTS, name)

    db = p(f"{args.tag}.db")
    manifest = p(f"{args.tag}_manifest.json")
    out_am, out_rag, out_none = (p(f"{args.tag}_{s}.json") for s in ("automem", "rag", "none"))

    log(f"=== held-out run tag={args.tag} limit={args.limit} ===")
    phase("ingest", ["add_runner.py", "--db", db, "--limit", str(args.limit),
                     "--workers", str(args.workers), "--resume"])
    phase("automem-search", ["search_runner.py", "--db", db, "--out", out_am,
                             "--top-k", str(args.top_k), "--workers", str(args.workers)])
    phase("rag", ["baseline_runner.py", "--mode", "rag", "--manifest", manifest,
                  "--out", out_rag, "--top-k", str(args.top_k), "--workers", str(args.rag_workers)])
    phase("none", ["baseline_runner.py", "--mode", "none", "--manifest", manifest,
                   "--out", out_none, "--workers", str(args.workers)])
    for label, out in (("automem", out_am), ("rag", out_rag), ("none", out_none)):
        phase(f"judge-{label}", ["judge.py", "--input", out], required=False)
    log("=== run complete; summarize with: python summarize.py --tag " + args.tag + " ===")


if __name__ == "__main__":
    main()
