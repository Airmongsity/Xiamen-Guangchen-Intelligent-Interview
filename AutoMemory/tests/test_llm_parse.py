"""Lenient/salvage JSON parsing for LLM responses (no API key needed)."""

from automem.providers.llm import _parse_json_lenient


def test_parses_plain_json():
    assert _parse_json_lenient('{"memories": [{"content": "a"}]}') == {
        "memories": [{"content": "a"}]
    }


def test_strips_code_fence():
    text = '```json\n{"op": "ADD"}\n```'
    assert _parse_json_lenient(text) == {"op": "ADD"}


def test_salvages_truncated_memories_array():
    # response cut off mid-string after two complete objects
    text = (
        '{"memories": ['
        '{"content": "UserA lives in Milan", "kind": "fact", "importance": 0.6, "slot": "home_city"},'
        '{"content": "UserA likes pizza", "kind": "fact", "importance": 0.5, "slot": null},'
        '{"content": "UserA started a very long story that got cut o'
    )
    out = _parse_json_lenient(text)
    contents = [m["content"] for m in out["memories"]]
    assert contents == ["UserA lives in Milan", "UserA likes pizza"]


def test_unrecoverable_returns_empty_dict():
    assert _parse_json_lenient("total garbage not json") == {}
