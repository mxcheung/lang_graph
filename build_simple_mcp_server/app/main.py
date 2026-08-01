from fastapi import FastAPI, HTTPException

from app.models import ModelContextRequest, ModelContextResponse
from app.tools import GET_WEATHER_TOOL, get_weather


app = FastAPI()


tool_registry = {
    "get_weather": get_weather
}


@app.post("/mcp", response_model=ModelContextResponse)
async def handle_mcp_request(request: ModelContextRequest):
    if request.verb == "discovery":
        return ModelContextResponse(tools=[GET_WEATHER_TOOL])
    
    elif request.verb == "execute":
        try:
            tool_function = tool_registry[request.tool_name]
            arguments = request.arguments or {}
            result = tool_function(**arguments)
            return ModelContextResponse(result=result)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found.")

    raise HTTPException(status_code=400, detail=f"Unknown verb '{request.verb}'")
