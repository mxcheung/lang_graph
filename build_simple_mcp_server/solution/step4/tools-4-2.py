from .models import Tool, ToolParameter


def get_weather(location: str) -> str:
    """Gets the current weather for a specified location."""
    return f"The weather in {location} is sunny."


GET_WEATHER_TOOL = Tool(
    name="get_weather",
    description="Gets the current weather for a specified location.",
    parameters=[
        ToolParameter(name="location", type="string")
    ]
)
