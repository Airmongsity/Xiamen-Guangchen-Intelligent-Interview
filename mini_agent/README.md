# mini_agent — a minimal, from-scratch LLM agent

A minimal but complete **Agent Runtime** built from scratch (no langgraph /
openhands / openclaw). It implements the core loop, a tool-registration
mechanism, LLM-output parsing, isolated sessions, basic context compression,
tracing, and error handling — driven by a **real** OpenAI-compatible LLM API.

```
receive user input
  → LLM decides: answer directly, or call tool(s)?
  → run tool(s), feed results back
  → loop until the LLM gives a final answer (or a safety limit trips)
```

Only the runtime is hand-written; the OpenAI SDK is used purely as the HTTP
client to an OpenAI-compatible endpoint (it is not an agent framework).

## Setup

```bash
pip install -r mini_agent/requirements.txt
# Put credentials in a root .env (see mini_agent/.env.example)
```

The config reads either convention from `.env` (`CHAT_*` wins):

| Purpose   | Generic keys      | DeepSeek keys        |
|-----------|-------------------|----------------------|
| API key   | `CHAT_API_KEY`    | `DEEPSEEK_API_KEY`   |
| Base URL  | `CHAT_API_URL`\*  | `DEEPSEEK_BASE_URL`  |
| Model     | `CHAT_MODEL`      | `MINI_AGENT_MODEL`   |

\* `CHAT_API_URL` may be a full endpoint (e.g. `…/v1/chat/completions`); the
trailing `/chat/completions` is stripped to form the SDK `base_url`.

> **⚠️ Model note — must support function calling.**
> The loop relies on OpenAI-style `tool_calls`. Small/weak models fail here:
> `Qwen/Qwen2.5-7B-Instruct` on SiliconFlow, in testing, produced degenerate
> output and **timed out** on tool requests. Verified working, ~2s per call:
> **`deepseek-ai/DeepSeek-V3`** and **`Qwen/Qwen2.5-72B-Instruct`**. Set e.g.
> `CHAT_MODEL=deepseek-ai/DeepSeek-V3` for the demo/recording.

## Run

```bash
python -m mini_agent.cli            # interactive REPL
python -m mini_agent.examples.demo  # scripted two-window demo (for recording)
```

REPL commands: `/session <id>`, `/sessions`, `/todos`, `/trace`, `/help`, `/quit`.

## System design

| Module | Responsibility |
|---|---|
| `runtime.py` | **The core loop.** Decide → call tools → loop → finalize. Owns all orchestration and the safety limits. |
| `llm_client.py` | Thin OpenAI-compatible client: credentials, model, retry with backoff on transient errors. |
| `tools/registry.py` | **Tool registration**: each tool = `name` + `description` + JSON-Schema `parameters` + a Python `func`. `schemas()` produces the `tools=[…]` payload the LLM sees. |
| `tools/*.py` | `calculator` (safe AST eval), `search` (mock, deterministic), `weather` (mock, deterministic), `todo` (**user-scoped** — shared across a user's windows). |
| `parser.py` | Extract **thought / tool calls / final answer** from a response; salvage malformed or truncated argument JSON instead of crashing. |
| `session.py` | Per-`session_id` conversation (history, rolling summary) — **isolated** between windows; **todos are per-user**, shared across a user's sessions. Thread-safe `SessionManager`. |
| `context_manager.py` | Assemble the per-step message list; **basic compression** of old turns into a rolling summary at safe (user-turn) boundaries. |
| `trace.py` | Per-call execution trace (tool, args, ok/error, latency) + logging. |

### The loop and its safety limits

- **Terminates** when the model returns a message with no tool call (final answer).
- **`max_steps`** (default 8) caps tool-calling iterations; on hit, the runtime
  forces one tool-free finalize call so the user still gets an answer
  (`stop_reason="max_steps"`).
- **Repeated failure guard**: after 3 consecutive steps where every tool call
  failed, it stops gracefully (`stop_reason="tool_failures"`).
- **Parallel tool calls** in one step are supported (the model often batches them).

### Error handling

Tool exceptions, unknown tools, and un-parseable arguments are **caught and
returned to the model as a tool result** (`{"error": …}`), so it can recover on
the next step rather than the process crashing. LLM transport errors are retried
in `llm_client`; if the LLM still fails, the turn ends with a polite message.

## Context management & memory: what goes in, when, and where

This is the heart of the "context の有効管理" requirement.

**What is put into context** (assembled fresh every step, in this order):

1. **system prompt** — role/instructions.
2. **rolling summary** (a second `system` message) — *only if* older turns have
   been compressed. Placed high so it frames the recent turns.
3. **recent verbatim turns** — the tail of the conversation kept in full:
   user inputs, the assistant's tool-call messages, and each `tool` result.

So the model always sees: durable instructions → compressed older context →
exact recent exchange → (within the loop) the fresh tool results it just got.

**When compression happens (recall/placement timing).** On each new user turn,
before calling the LLM, `maybe_compress` checks the history length. Past
`max_history_messages` (default 20), the oldest turns are summarized by the LLM
(preserving names, numbers, dates, decisions, pending todos) into the rolling
summary, and only the last `keep_recent_messages` (default 8) are kept verbatim.
Compression only cuts at a **user-turn boundary**, never between an assistant
`tool_calls` message and its `tool` results — cutting there would produce an
orphaned tool message the API rejects. This is *basic* compression by design
(the task scopes complex/hierarchical compression out).

**Follow-ups** — pure or tool-using — work because the recent turns + rolling
summary are in context, so the model can answer "what did I ask you to remember?"
or re-run a tool against remembered state.

**Scope note (deliberate):** conversation context is per **session** (windows do
not bleed into each other), but the **todo list is per user** — a reminder
belongs to the person, so adding a todo in one window and asking for it in
another returns it. The task only requires the *sessions/conversations* to be
independent, not the todo list; a user-global reminder is the product-correct
choice.

### Long-term memory (AutoMemory) — optional, wired via a flag

The compression above is *within-session* working memory. For **cross-session,
long-term** recall, the sibling `AutoMemory` project (hybrid retrieval +
forgetting curve + outcome feedback) plugs in behind the `MemoryBackend`
protocol in `memory.py`. It is **off by default** (the runtime uses `NullMemory`
so it stays self-contained); enable it with:

```bash
python -m mini_agent.cli --memory          # REPL with long-term memory
```
```python
from mini_agent import build_agent, AutoMemoryBackend
agent = build_agent(memory=AutoMemoryBackend(db_path="mini_agent_memory.db"))
```

**Recall timing & placement** (`runtime.Agent.chat`):

- **Recall** runs *once per turn*, before the tool loop:
  `memory.recall(user_input, user_id)` → an injectable `<memory>` block.
- **Placement**: the block is a `system` message placed *after* the base
  prompt/summary and *before* the current turn (see `context_manager.build`),
  then reused on every step of that turn (recall is not repeated per step).
- **Record**: after the final answer, `memory.record(session_id, user, answer)`
  runs the exchange through AutoMemory's extraction so it is recallable later.
- **Feedback**: `memory.report(user_id, quality)` is exposed to close
  AutoMemory's outcome-feedback loop (not auto-called, since "did it help?" is a
  caller judgement).

**Scope**: long-term memory is namespaced by `user_id` in AutoMemory (like
todos), so personal facts a user shares in one window are recallable from any of
their windows — the "the agent gets to know *you*" goal. Conversation *context*
stays per-session. Verified live: a *fresh* agent (empty working memory) sharing
the same DB recalls the user's facts from a prior session.

**.env note**: AutoMemory reads `DEEPSEEK_API_KEY` (extraction LLM) and
`SILICONFLOW_API_KEY` (embedding `BAAI/bge-m3` 1024-dim + reranker). This repo's
`.env` names the SiliconFlow key `EMBEDDING_API_KEY`; the adapter passes it
through, so no `.env` edit is needed. AutoMemory's embedder/reranker are fixed to
its evaluated defaults (bge-m3), *not* the repo's `EMBEDDING_MODEL` — swapping
them would invalidate AutoMemory's benchmark numbers and needs a full re-index,
and retrieval is not its bottleneck (session Recall@k ≈ 0.993), so it is left
stable on purpose.

## Tests

```bash
python -m pytest tests/ -q      # 33 offline tests, no network, time-invariant
```

Coverage: calculator (incl. rejecting non-arithmetic), parser + JSON salvage,
tool registration, session isolation, user-scoped todos, deterministic
search/weather, context compression (incl. the safe-boundary rule), and the full
loop via a scripted `FakeLLM` — single-tool, multi-step, session isolation,
max-steps finalize, repeated-failure stop, and unknown-tool handling. Mock tools
are clock-free, so tests are time-invariant per the project's testing rule.

## Project structure

```
mini_agent/
├── runtime.py          # core loop (self-implemented)
├── llm_client.py       # OpenAI-compatible client + retry
├── config.py           # .env loader + Settings (CHAT_* / DEEPSEEK_*)
├── parser.py           # thought / tool-call / final-answer + JSON salvage
├── session.py          # Session + thread-safe SessionManager (isolation)
├── context_manager.py  # context assembly + basic compression
├── trace.py            # tool-call execution trace
├── tools/              # registry + calculator / search / weather / todo
├── examples/demo.py    # scripted two-window demo (recording)
└── cli.py              # interactive REPL
tests/                  # 33 offline tests + FakeLLM fixtures
```
