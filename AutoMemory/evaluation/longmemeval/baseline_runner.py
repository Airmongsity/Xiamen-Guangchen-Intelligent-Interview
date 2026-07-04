"""External baselines for LongMemEval, scored on the same manifest subset.

  none  -- no memory; the LLM answers each question blind (the floor).
  rag   -- naive RAG: every raw haystack turn is a chunk, embedded with bge-m3,
           top-k retrieved per question by cosine, answered with the same reader
           prompt. No extraction / scoring / decay -- isolates what AutoMemory's
           machinery adds over plain retrieval on a LONG history.

Usage:
  python evaluation/longmemeval/baseline_runner.py --mode rag \
      --manifest ../results/lme_manifest.json --out ../results/lme_rag.json
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from common import ANSWER_PROMPT, config, load_dataset, parse_lme_date, recall_fields

from automem.providers import DeepSeekLLM, SiliconFlowEmbedder

NONE_PROMPT = """\
You are answering with NO access to any conversation history. Answer only if \
certain from general knowledge; otherwise reply exactly "I don't know / no \
information available". Be concise.

Question: {question}
Answer:\
"""


# bge-m3 accepts 8192 tokens; cap each chunk well under that (~1 token/4 chars)
MAX_CHUNK_CHARS = 6000


def build_chunks(item: dict) -> tuple[list[str], list[str], list]:
    texts, dates, sids = [], [], []
    sessions = item["haystack_sessions"]
    hdates = item.get("haystack_dates", [])
    hsids = item.get("haystack_session_ids", [])
    for i, session in enumerate(sessions):
        iso = parse_lme_date(hdates[i] if i < len(hdates) else "")[:10]
        sid = hsids[i] if i < len(hsids) else None
        for t in session:
            c = t.get("content", "")
            if c:
                texts.append(f"{t.get('role','user')}: {c}"[:MAX_CHUNK_CHARS])
                dates.append(iso)
                sids.append(sid)
    return texts, dates, sids


def answer_rag(llm, embedder, item, top_k) -> dict:
    question = item["question"]
    texts, dates, sids = build_chunks(item)
    t0 = time.time()
    mat = np.asarray(embedder.embed_batch(texts), dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
    qv = np.asarray(embedder.embed(question), dtype=np.float32)
    qv /= np.linalg.norm(qv) or 1.0
    order = np.argsort(-(mat @ qv))[:top_k]
    mems = "\n".join(f"- [{dates[i]}] {texts[i]}" for i in order)
    search_time = time.time() - t0
    # same session-level Recall@k metric as the AutoMem runner, on the same labels
    recall = recall_fields(
        item.get("answer_session_ids"), {sids[i] for i in order if sids[i]}
    )
    prompt = ANSWER_PROMPT.format(
        memories=mems, question_date=item.get("question_date", ""), question=question
    )
    response = llm.complete(
        "You answer questions strictly from the provided memories.", prompt, temperature=0.0
    )
    return _row(item, response, search_time, len(order), recall)


def answer_none(llm, item) -> dict:
    response = llm.complete(
        "You answer questions.",
        NONE_PROMPT.format(question=item["question"]),
        temperature=0.0,
    )
    return _row(item, response, 0.0, 0, recall_fields(item.get("answer_session_ids"), set()))


def _row(item, response, search_time, n_ret, recall=None) -> dict:
    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "answer": item.get("answer", ""),
        "response": response.strip(),
        "n_retrieved": n_ret,
        "search_time": search_time,
        **(recall or {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "rag"], required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    data = {x["question_id"]: x for x in load_dataset(args.dataset)}
    with open(args.manifest, encoding="utf-8") as f:
        qids = json.load(f)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # resume: keep any answers already saved, only (re)do the missing ones
    out: dict[str, dict] = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            out = json.load(f)
    items = [data[q] for q in qids if q in data and q not in out]

    cfg = config()
    llm = DeepSeekLLM(cfg)
    embedder = SiliconFlowEmbedder(cfg) if args.mode == "rag" else None
    print(f"{args.mode}: {len(items)} to do, {len(out)} already done ({args.workers} workers)...")

    lock = threading.Lock()
    done = 0

    def run_one(it):
        nonlocal done
        row = answer_none(llm, it) if args.mode == "none" else answer_rag(
            llm, embedder, it, args.top_k
        )
        with lock:
            out[row["question_id"]] = row
            done += 1
            # checkpoint every answer so a 429 crash never loses progress
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            if done % 5 == 0 or done == len(items):
                print(f"  {done}/{len(items)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run_one, items))
    print(f"saved {args.out} ({len(out)} total)")


if __name__ == "__main__":
    main()
