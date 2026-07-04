"""External baselines for the LoCoMo benchmark, to contextualize AutoMemory.

Two modes, both emitting judge.py-compatible JSON ({conv_idx: [rows]}):

  none  -- no memory at all; the LLM answers each question blind. This is the
           floor: how much can DeepSeek answer from parametric knowledge / the
           question alone, with zero conversation context.

  rag   -- naive retrieval-augmented baseline: NO LLM extraction, NO scoring,
           NO decay/feedback. Every raw dialogue turn becomes a chunk, embedded
           with the same bge-m3 model; each question retrieves the top-k chunks
           per speaker by cosine and answers with the same ANSWER_PROMPT.
           Isolates "what does the AutoMemory machinery add over plain RAG".

Usage:
  python evaluation/locomo/baseline_runner.py --mode rag  --conv-ids 0 \
      --out evaluation/results/rag_results.json
  python evaluation/locomo/baseline_runner.py --mode none --conv-ids 0 \
      --out evaluation/results/none_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from common import (
    ANSWER_PROMPT,
    iter_sessions,
    load_dataset,
    parse_locomo_timestamp,
    speaker_user_ids,
)

from automem import AutoMemConfig
from automem.providers import DeepSeekLLM, SiliconFlowEmbedder

NONE_PROMPT = """\
You are answering a question with NO access to any conversation history or \
memories. Answer only if you are certain from general knowledge; otherwise \
reply "No information available". Answer in fewer than 6 words.

Question: {question}
Answer:\
"""


def build_chunks(item: dict, idx: int) -> dict[str, dict]:
    """Per-speaker raw-turn chunks: {user_id: {"texts": [...], "dates": [...]}}."""
    user_a, user_b = speaker_user_ids(item, idx)
    by_user: dict[str, dict] = {
        user_a: {"texts": [], "dates": []},
        user_b: {"texts": [], "dates": []},
    }
    name_to_user = {user_a.rsplit("_", 1)[0]: user_a, user_b.rsplit("_", 1)[0]: user_b}
    for _key, date_str, chats in iter_sessions(item["conversation"]):
        iso = parse_locomo_timestamp(date_str)
        for turn in chats:
            speaker = turn.get("speaker")
            text = turn.get("text", "")
            uid = name_to_user.get(speaker)
            if uid is None or not text:
                continue
            by_user[uid]["texts"].append(f"{speaker}: {text}")
            by_user[uid]["dates"].append(iso[:10])
    return by_user


def embed_chunks(embedder, by_user: dict[str, dict]) -> dict[str, dict]:
    for uid, bucket in by_user.items():
        if bucket["texts"]:
            vecs = embedder.embed_batch(bucket["texts"])
            mat = np.asarray(vecs, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-9)
            bucket["mat"] = mat
        else:
            bucket["mat"] = np.zeros((0, embedder.dim), dtype=np.float32)
    return by_user


def top_k_chunks(bucket: dict, qvec: np.ndarray, k: int) -> list[str]:
    mat = bucket["mat"]
    if mat.shape[0] == 0:
        return []
    sims = mat @ qvec
    order = np.argsort(-sims)[:k]
    return [f"{bucket['dates'][i]}: {bucket['texts'][i]}" for i in order]


def answer_rag(llm, embedder, by_user, user_a, user_b, qa, top_k) -> dict:
    question = qa["question"]
    t0 = time.time()
    qvec = np.asarray(embedder.embed(question), dtype=np.float32)
    qvec /= np.linalg.norm(qvec) or 1.0
    mem_a = top_k_chunks(by_user[user_a], qvec, top_k)
    mem_b = top_k_chunks(by_user[user_b], qvec, top_k)
    search_time = time.time() - t0
    prompt = ANSWER_PROMPT.format(
        speaker_1=user_a.rsplit("_", 1)[0],
        speaker_2=user_b.rsplit("_", 1)[0],
        speaker_1_memories=json.dumps(mem_a, ensure_ascii=False, indent=2),
        speaker_2_memories=json.dumps(mem_b, ensure_ascii=False, indent=2),
        question=question,
    )
    response = llm.complete("You answer questions from memories.", prompt, temperature=0.0)
    return _row(qa, response, search_time)


def answer_none(llm, qa) -> dict:
    question = qa["question"]
    response = llm.complete(
        "You answer questions.",
        NONE_PROMPT.format(question=question),
        temperature=0.0,
    )
    return _row(qa, response, 0.0)


def _row(qa: dict, response: str, search_time: float) -> dict:
    return {
        "question": qa["question"],
        "answer": qa.get("answer", qa.get("adversarial_answer", "")),
        "category": qa.get("category", -1),
        "evidence": qa.get("evidence", []),
        "response": response.strip(),
        "search_time": search_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "rag"], required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conv-ids", default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    conv_ids = (
        [int(x) for x in args.conv_ids.split(",")] if args.conv_ids else range(len(data))
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    cfg = AutoMemConfig.from_env(env_file=env_file)
    llm = DeepSeekLLM(cfg)
    embedder = SiliconFlowEmbedder(cfg) if args.mode == "rag" else None

    results: dict[str, list] = {}
    for idx in conv_ids:
        item = data[idx]
        qa = [q for q in item["qa"] if int(q.get("category", -1)) != 5]
        print(f"=== conversation {idx}: {len(qa)} questions ({args.mode}) ===")
        if args.mode == "none":
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(lambda q: answer_none(llm, q), qa))
        else:
            user_a, user_b = speaker_user_ids(item, idx)
            print("  embedding raw turns...")
            by_user = embed_chunks(embedder, build_chunks(item, idx))
            n = sum(len(b["texts"]) for b in by_user.values())
            print(f"  {n} chunks embedded; answering...")
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(
                    ex.map(
                        lambda q: answer_rag(
                            llm, embedder, by_user, user_a, user_b, q, args.top_k
                        ),
                        qa,
                    )
                )
        results[str(idx)] = rows
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
