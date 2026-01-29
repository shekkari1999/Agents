# AI Agent Framework

A flexible framework for building AI agents with tool support, MCP integration, and multi-step reasoning.

## Features

- **Agent System**: Multi-step reasoning with tool execution
- **Tool Framework**: Easy tool creation and integration
- **MCP Integration**: Load tools from MCP servers
- **LLM Client**: Unified interface for LLM API calls via LiteLLM
- **Modular Design**: Clean, organized package structure

## Installation

### Prerequisites

Install `uv` (recommended package manager):
- macOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Homebrew (macOS):
```bash
brew install uv
```

### Setup

1. Create a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys (OPENAI_API_KEY, TAVILY_API_KEY, etc.)
```

## Quick Start

```python
from agent_framework import Agent, LlmClient, FunctionTool

# Define a tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

# Create the agent
agent = Agent(
    model=LlmClient(model="gpt-5-mini"),
    tools=[FunctionTool(calculator)],
    instructions="You are a helpful assistant.",
)

# Run the agent
result = await agent.run("What is 1234 * 5678?")
print(result.output)  # "7006652"
```

## Package Structure

```
agent_framework/
├── __init__.py      # Package exports
├── models.py        # Core data models (Message, ToolCall, Event, ExecutionContext)
├── tools.py         # BaseTool and FunctionTool classes
├── llm.py           # LlmClient and request/response models
├── agent.py         # Agent and AgentResult classes
├── mcp.py           # MCP tool loading utilities
└── utils.py         # Helper functions for tool definitions
```

## Usage Examples

### Using the @tool Decorator

```python
from agent_framework import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# multiply is now a FunctionTool instance
```

### MCP Tool Integration

```python
from agent_framework import load_mcp_tools
import os

connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
}

mcp_tools = await load_mcp_tools(connection)
agent = Agent(
    model=LlmClient(model="gpt-5-mini"),
    tools=mcp_tools,
)
```

## Documentation

See `agent_framework/README.md` for detailed API documentation.

## License

See LICENSE file for details.
