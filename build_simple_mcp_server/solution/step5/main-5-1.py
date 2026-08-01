from fastapi import FastAPI, HTTPException

from app.models import ModelContextRequest, ModelContextResponse
from app.tools import GET_WEATHER_TOOL, get_weather


app = FastAPI()


tool_registry = {
    "get_weather": get_weather
}


@app.post("/mcp")
async def handle_mcp_request(request: ModelContextRequest):
    if request.verb == "discovery":
        return ModelContextResponse(tools=[GET_WEATHER_TOOL])

    # TODO: Task 5.2 - Implement the execution handler

    # TODO: Task 6.1 - Add error handling for unknown tools
    pass
