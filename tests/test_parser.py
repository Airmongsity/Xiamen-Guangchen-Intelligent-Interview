from mini_agent.parser import PARSE_ERROR_KEY, parse_message, salvage_json


def test_final_answer(make_message):
    parsed = parse_message(make_message(content="Hello there"))
    assert parsed.final_answer == "Hello there"
    assert parsed.tool_calls == []


def test_tool_call_extraction(make_message):
    msg = make_message(content="let me compute", tool_calls=[("calculator", {"expression": "1+1"})])
    parsed = parse_message(msg)
    assert parsed.final_answer is None
    assert parsed.thought == "let me compute"
    assert parsed.tool_calls[0]["name"] == "calculator"
    assert parsed.tool_calls[0]["arguments"] == {"expression": "1+1"}


def test_salvage_valid_json():
    assert salvage_json('{"a": 1}') == {"a": 1}


def test_salvage_empty():
    assert salvage_json("") == {}
    assert salvage_json(None) == {}


def test_salvage_truncated_json():
    # Missing closing brace/quote -- should be recovered, not crash.
    assert salvage_json('{"city": "Xiamen') == {"city": "Xiamen"}


def test_salvage_gives_up_gracefully():
    out = salvage_json("this is not json at all !!!")
    assert out.get(PARSE_ERROR_KEY) is True
