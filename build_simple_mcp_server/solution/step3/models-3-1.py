from pydantic import BaseModel
from typing import Optional, List, Any, Dict


class ToolParameter(BaseModel):
    name: str
    type: str


class Tool(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]


# TODO: Task 3.2 - Create the ModelContextRequest model


# TODO: Task 3.2 - Create the ModelContextResponse model
