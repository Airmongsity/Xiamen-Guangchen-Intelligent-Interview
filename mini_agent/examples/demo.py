"""Scripted demo for the screen recording.

Reproduces the task's two-window scenario and shows the two scopes:

    window 1: check weather   + add a todo (bring umbrella)
    window 2: weekly report   + add a todo (finish report)
    window 1 (resumed): its CONVERSATION only remembers weather, not the report
                        -> conversations are isolated per session
    window 2 (resumed): asking "all my todos" shows BOTH items
                        -> todos are per-USER, shared across windows

Run:  python -m mini_agent.examples.demo
Requires a function-calling-capable CHAT_MODEL (see README's model note).
"""

from __future__ import annotations

from mini_agent.runtime import build_agent


def _turn(agent, label, text, session_id):
    print(f"\n=== {label}  [session={session_id}] ===")
    print(f"user> {text}")
    result = agent.chat(text, session_id=session_id)
    print(f"agent> {result.answer}")
    print(f"       (steps={result.steps}, stop={result.stop_reason})")


def main() -> None:
    agent = build_agent()

    _turn(agent, "window 1", "What's the weather in Xiamen? Add a todo to bring an umbrella.", "window-1")
    _turn(agent, "window 2", "Help me outline a weekly report, and add a todo to finish it by Friday.", "window-2")
    # Conversation isolation: window-1 only knows its own thread (weather), not the report.
    _turn(agent, "window 1 resumed", "What was the topic we just discussed in THIS window?", "window-1")
    # User-scoped todos: window-2 sees the umbrella added in window-1 too.
    _turn(agent, "window 2 resumed", "List ALL my todos.", "window-2")

    print("\n--- final state ---")
    print("window-1 messages:", len(agent.sessions.get("window-1").messages), "(isolated conversation)")
    print("window-2 messages:", len(agent.sessions.get("window-2").messages), "(isolated conversation)")
    print("shared user todos:", agent.sessions.todos_for("default"))


if __name__ == "__main__":
    main()
