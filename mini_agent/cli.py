"""Interactive terminal REPL -- the entry point for the demo recording.

Run:  python -m mini_agent.cli

Commands:
    /session <id>   switch to (or create) a session  -- demonstrates isolation
    /sessions       list active session ids
    /todos          show the current session's todo list
    /trace          show tool-call trace for the current session
    /help           show this help
    /quit           exit
Anything else is sent to the agent as a message.
"""

from __future__ import annotations

import logging
import sys

from .runtime import build_agent

BANNER = """\
mini_agent REPL. Type /help for commands, /quit to exit.
Two sessions demo:  /session w1  -> chat ; /session w2 -> chat ; switch back anytime.
"""

HELP = __doc__


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    use_memory = "--memory" in argv

    # Models often emit emoji; the Windows console defaults to GBK and would crash
    # on them. Force UTF-8 output where supported.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Keep the transcript readable: tool traces still print via the logger above.
    logging.getLogger("mini_agent.llm").setLevel(logging.WARNING)

    try:
        agent = build_agent(use_memory=use_memory)
    except RuntimeError as err:
        print(f"Startup error: {err}", file=sys.stderr)
        return 1
    if use_memory:
        print("[long-term memory: AutoMemory enabled]")

    print(BANNER)
    current = "default"
    print(f"[session: {current}]")

    while True:
        try:
            line = input(f"({current}) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(HELP)
            continue
        if line == "/sessions":
            print("sessions:", agent.sessions.list_ids() or "(none yet)")
            continue
        if line.startswith("/session"):
            parts = line.split(maxsplit=1)
            current = parts[1].strip() if len(parts) > 1 else current
            agent.sessions.get_or_create(current)
            print(f"[switched to session: {current}]")
            continue
        if line == "/todos":
            session = agent.sessions.get(current)
            print("todos:", session.todos if session else "(empty)")
            continue
        if line == "/trace":
            if agent.tracer is None:
                print("(tracing disabled)")
            else:
                for e in agent.tracer.for_session(current):
                    status = "ok" if e.ok else f"ERR:{e.error}"
                    print(f"  step {e.step} {e.tool}({e.args}) [{status}] {e.latency_ms}ms")
            continue

        result = agent.chat(line, session_id=current)
        print(f"\n{result.answer}\n  (steps={result.steps}, stop={result.stop_reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
