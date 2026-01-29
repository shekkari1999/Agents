# Agent Framework

A flexible framework for building AI agents with tool support, MCP integration, and multi-step reasoning.

## Structure

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

## Components

### Models (`models.py`)
- `Message`: Text messages in conversations
- `ToolCall`: LLM's request to execute a tool
- `ToolResult`: Result from tool execution
- `Event`: Recorded occurrence during agent execution
- `ExecutionContext`: Central storage for execution state

### Tools (`tools.py`)
- `BaseTool`: Abstract base class for all tools
- `FunctionTool`: Wraps Python functions as tools

### LLM (`llm.py`)
- `LlmClient`: Client for LLM API calls using LiteLLM
- `LlmRequest`: Request object for LLM calls
- `LlmResponse`: Response object from LLM calls

### Agent (`agent.py`)
- `Agent`: Main agent class that orchestrates reasoning and tool execution
- `AgentResult`: Result of an agent execution

### MCP (`mcp.py`)
- `load_mcp_tools()`: Load tools from MCP servers

### Utils (`utils.py`)
- `function_to_input_schema()`: Convert function signature to JSON Schema
- `format_tool_definition()`: Format tool definition in OpenAI format
- `tool`: Decorator to convert functions to tools

## Usage Examples

### Basic Tool Usage

```python
from agent_framework import FunctionTool

def my_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

tool = FunctionTool(my_function)
result = await tool.execute(context, x=5, y=3)  # 8
```

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

## Installation

The framework uses:
- `pydantic` for data validation
- `litellm` for LLM API calls
- `mcp` for MCP server integration

Install dependencies:
```bash
pip install pydantic litellm mcp
```

