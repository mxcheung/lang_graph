show me how to write lang graph from scratch to get openning shop hours

# 🏪 Shop Hours Assistant

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20-green.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A powerful LangGraph-powered application that retrieves and displays shop opening hours using natural language queries. Built with LangChain and OpenAI.

## ✨ Features

- 🔍 **Natural Language Processing** - Ask about shop hours in plain English
- 🤖 **LLM-Powered** - Uses OpenAI's GPT for intelligent parsing
- 📊 **Structured Responses** - Beautifully formatted output with current day highlight
- 💬 **Interactive CLI** - User-friendly command-line interface
- 🔄 **Extensible Workflow** - Built on LangGraph for easy customization
- 🎯 **Smart Extraction** - Automatically identifies shop names from queries
- ⚡ **Error Handling** - Graceful failure recovery

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shop-hours-langgraph.git
cd shop-hours-langgraph

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```


```
shop-hours-langgraph/
├── requirements.txt
├── setup.sh
├── .env
├── .gitignore
├── README.md
├── main.py                 # Main application
├── src/
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflow
│   ├── nodes.py           # Node implementations
│   ├── models.py          # Pydantic models
│   ├── state.py           # State definitions
│   └── utils.py           # Utility functions
├── data/
│   └── mock_hours.json    # Mock data for testing
└── tests/
    ├── __init__.py
    ├── test_graph.py
    └── test_nodes.py
```    
