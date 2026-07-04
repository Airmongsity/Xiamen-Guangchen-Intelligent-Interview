"""weather tool (mocked).

Deterministic, clock-free canned data so tests stay time-invariant (per the
project's testing rule). A real implementation would call a weather API inside
``_lookup`` and keep the same schema.
"""

from __future__ import annotations

from .registry import Tool, ToolContext

_CANNED: dict[str, dict] = {
    "xiamen": {"condition": "sunny", "temp_c": 29, "humidity": 70},
    "beijing": {"condition": "cloudy", "temp_c": 24, "humidity": 45},
    "shanghai": {"condition": "light rain", "temp_c": 26, "humidity": 80},
    "shenzhen": {"condition": "thunderstorm", "temp_c": 30, "humidity": 85},
}


def get_weather(ctx: ToolContext, city: str) -> dict:
    data = _CANNED.get(city.strip().lower(), {"condition": "clear", "temp_c": 22, "humidity": 55})
    return {"city": city, **data}


WEATHER = Tool(
    name="weather",
    description="Get the current (mocked) weather for a city: condition, temperature, humidity.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Xiamen'."}
        },
        "required": ["city"],
    },
    func=get_weather,
)
