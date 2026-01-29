"""Agent Framework - A flexible framework for building AI agents with tool support."""

from .models import (
    Message,
    ToolCall,
    ToolResult,
    ContentItem,
    Event,
    ExecutionContext,
)
from .tools import BaseTool, FunctionTool, tool
from .llm import LlmClient, LlmRequest, LlmResponse
from .agent import Agent, AgentResult
from .mcp import load_mcp_tools
from .utils import (
    function_to_input_schema,
    format_tool_definition,
    function_to_tool_definition,
    mcp_tools_to_openai_format,
    display_trace,
)

__all__ = [
    # Models
    "Message",
    "ToolCall",
    "ToolResult",
    "ContentItem",
    "Event",
    "ExecutionContext",
    # Tools
    "BaseTool",
    "FunctionTool",
    "tool",
    # LLM
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    # Agent
    "Agent",
    "AgentResult",
    # MCP
    "load_mcp_tools",
    # Utils
    "function_to_input_schema",
    "format_tool_definition",
    "function_to_tool_definition",
    "mcp_tools_to_openai_format",
    "display_trace",
]

__version__ = "0.1.0"

