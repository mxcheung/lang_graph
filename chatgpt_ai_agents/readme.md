```python
import os
import json
import math
from openai import AzureOpenAI
```


```python
os.environ['AZURE_OPENAI_API_KEY'] = '869f60fe-c046-4bb4-838f-63f07c767620'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'http://pluralsight.openai.azure.com'
os.environ['AZURE_OPENAI_API_VERSION'] = '2024-06-01'
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
)
response = client.chat.completions.create( 
    model="gpt-4o-mini",
    messages=[ 
        { 
            "role": "user", 
            "content": "What are the Olympics?"
        }
    ],
    max_tokens=100,
) 
print(response.choices[0].message.content)
```

    The Olympics, officially known as the Olympic Games, is a major international multi-sport event featuring summer and winter sports competitions. The games are held every four years, with the Summer and Winter Olympics alternating every two years. 
    
    The modern Olympic Games were founded by Pierre de Coubertin in 1896 and are based on the ancient Olympic Games that were held in Olympia, Greece, from 776 BC. The modern Olympics encompass a wide variety of sports, where athletes from around the world compete for


# RTCF system prompt


```python
system_prompt = """
Role:
You are an AI agent tutor and autonomous task assistant.

Task:
Use reasoning and tools to complete the user's task accurately.

Context:
You can use tools for calculation and AI-agent concept lookup.
Follow a plan → act → observe → refine process internally.
Do not reveal private chain-of-thought. Summarize your approach briefly.

Format:
Return:
1. Brief answer
2. Tool use summary
3. Reflection question for the learner
""".strip()
```

# Local tools


```python
def calculator(expression: str) -> str:
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": pow,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


KNOWLEDGE_BASE = {
    "agent": "An AI agent is a system that uses a model to decide actions toward a goal, often using tools.",
    "tool calling": "Tool calling lets a model request that the application execute a function and return the result.",
    "memory": "Agent memory stores useful prior context, task state, or observations.",
    "planning": "Planning is when an agent breaks a goal into smaller steps before acting.",
    "rtcf": "RTCF stands for Role, Task, Context, and Format. It helps structure prompts clearly.",
}


def kb_lookup(query: str) -> str:
    query_lower = query.lower()

    matches = {
        key: value
        for key, value in KNOWLEDGE_BASE.items()
        if key in query_lower
    }

    if not matches:
        return json.dumps(
            {
                "not_found": "No exact matching entry found.",
                "available_topics": list(KNOWLEDGE_BASE.keys()),
            },
            indent=2,
        )

    return json.dumps(matches, indent=2)


TOOLS = {
    "calculator": calculator,
    "kb_lookup": kb_lookup,
}
```

# Azure OpenAI tool schemas


```python
tool_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a safe math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, such as sqrt(144).",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_lookup",
            "description": "Look up short definitions about AI agents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The AI-agent concept or topic to look up.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]
```

# Agent loop


```python
def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"

    try:
        return TOOLS[tool_name](**tool_args)
    except Exception as e:
        return f"Tool execution error: {e}"


# -----------------------------
# Azure Chat Completions agent loop
# -----------------------------

def run_azure_agent(user_task: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task},
    ]

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tool_schema,
        tool_choice="auto",
        max_tokens=800,
    )

    assistant_message = first_response.choices[0].message

    # If the model does not request tools, return the direct response.
    if not assistant_message.tool_calls:
        return assistant_message.content

    # Add the assistant message that requested tool calls.
    messages.append(assistant_message)

    # Execute each requested tool call.
    for tool_call in assistant_message.tool_calls:
#         print("tool was called")
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool_result = run_tool(tool_name, tool_args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

    # Send the tool results back to the model for final synthesis.
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=800,
    )

    return final_response.choices[0].message.content


```

# Agent Tasks


```python
test_tasks = [
    "Explain what an AI agent is.",
    "What is RTCF prompt design?",
    "Explain tool calling and calculate sqrt(625).",
    "Explain how memory helps an agent complete multi-step tasks.",
]

for task in test_tasks:
    print("=" * 80)
    print("TASK:")
    print(task)
    print("\nAGENT RESPONSE:")
    print(run_azure_agent(task))
    print()
```

    ================================================================================
    TASK:
    Explain what an AI agent is.
    
    AGENT RESPONSE:
    1. An AI agent is a system that utilizes a model to determine actions to achieve a specific goal, often employing various tools to assist in this process.
    
    2. I used a knowledge base lookup to define an AI agent accurately.
    
    3. What aspects of AI agents do you find most intriguing or want to understand better?
    
    ================================================================================
    TASK:
    What is RTCF prompt design?
    
    AGENT RESPONSE:
    1. RTCF prompt design stands for Role, Task, Context, and Format. It is a framework used to structure prompts clearly and effectively.
    
    2. Tool use summary: I used an AI-agent concept lookup tool to retrieve the definition of RTCF prompt design.
    
    3. Reflection question for the learner: How do you think structuring prompts with the RTCF framework can improve communication or responses in AI interactions?
    
    ================================================================================
    TASK:
    Explain tool calling and calculate sqrt(625).
    
    AGENT RESPONSE:
    1. The square root of 625 is 25. Tool calling allows the model to request a function's execution and get the result.
    
    2. I used a calculator tool to compute the square root and a knowledge base lookup to explain tool calling.
    
    3. What do you think is the importance of using tools in problem-solving?
    
    ================================================================================
    TASK:
    Explain how memory helps an agent complete multi-step tasks.
    
    AGENT RESPONSE:
    1. Memory enables an agent to retain information about prior steps, contexts, or observations, allowing it to make informed decisions and adjust actions as needed to successfully complete multi-step tasks.
    
    2. I utilized the AI-agent concept lookup tool to find relevant information on how memory functions within AI agents.
    
    3. How do you think improving an agent's memory could enhance its performance in complex tasks?
    



```python

```


```python

```
