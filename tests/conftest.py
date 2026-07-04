"""Make the repo root importable and expose LLM fakes to all tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_message(content: str = "", tool_calls: list | None = None) -> SimpleNamespace:
    calls = []
    for i, (name, args) in enumerate(tool_calls or []):
        arguments = args if isinstance(args, str) else json.dumps(args)
        calls.append(SimpleNamespace(
            id=f"call_{i}",
            function=SimpleNamespace(name=name, arguments=arguments),
        ))
    return SimpleNamespace(content=content, tool_calls=calls or None)


def _make_response(content: str = "", tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=_make_message(content, tool_calls))])


class FakeLLM:
    """Scripted LLM.

    ``turns`` is a list of dicts, each either ``{"content": "..."}`` (final answer)
    or ``{"tool_calls": [(name, args_dict), ...]}``. Each ``.chat`` call pops the
    next turn. ``calls`` records the messages/tools it was given, for assertions.
    """

    def __init__(self, turns: list[dict]):
        self._turns = list(turns)
        self.calls: list[dict] = []

    def chat(self, messages, *, tools=None, tool_choice="auto", temperature=0.2):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._turns:
            return _make_response(content="(no more scripted turns)")
        turn = self._turns.pop(0)
        return _make_response(turn.get("content", ""), turn.get("tool_calls"))


@pytest.fixture
def make_message():
    return _make_message


@pytest.fixture
def make_response():
    return _make_response


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
