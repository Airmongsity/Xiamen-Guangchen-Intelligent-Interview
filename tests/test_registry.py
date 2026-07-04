import pytest

from mini_agent.tools import default_registry
from mini_agent.tools.registry import Tool, ToolRegistry


def _dummy(ctx, x: int = 0):
    return {"x": x}


def test_register_and_schema():
    reg = ToolRegistry()
    reg.register(Tool("dummy", "a dummy tool", {"type": "object", "properties": {}}, _dummy))
    assert "dummy" in reg
    schema = reg.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy"
    assert schema["function"]["description"] == "a dummy tool"


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(Tool("dummy", "d", {"type": "object", "properties": {}}, _dummy))
    with pytest.raises(ValueError):
        reg.register(Tool("dummy", "d2", {"type": "object", "properties": {}}, _dummy))


def test_default_registry_has_four_tools():
    reg = default_registry()
    assert set(reg.names()) == {"calculator", "search", "weather", "todo"}
    # Every tool exposes a valid JSON-Schema object for its parameters.
    for schema in reg.schemas():
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
