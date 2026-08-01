from .models import Tool, ToolParameter


def get_weather(location: str) -> str:
    """Gets the current weather for a specified location."""
    return f"The weather in {location} is sunny."


# TODO: Task 4.2 - Define the Tool Schema for get_weather
