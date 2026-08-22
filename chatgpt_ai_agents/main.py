import os
import json
import math
from openai import AzureOpenAI

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

