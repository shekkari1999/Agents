"""Agent Framework - A flexible framework for building AI agents with tool support."""

from .models import (
    Message,
    ToolCall,
    ToolResult,
    ContentItem,
    Event,
    ExecutionContext,
    Session,
    ToolConfirmation,
    PendingToolCall,
    BaseSessionManager,
    InMemorySessionManager,
)
from .tools import BaseTool, FunctionTool, tool
from .llm import LlmClient, LlmRequest, LlmResponse, build_messages
from .memory import (
    count_tokens,
    apply_sliding_window,
    apply_compaction,
    apply_summarization,
    ContextOptimizer,
)
from .callbacks import create_optimizer_callback
from .agent import Agent, AgentResult
from .mcp import load_mcp_tools
from .utils import (
    function_to_input_schema,
    format_tool_definition,
    function_to_tool_definition,
    mcp_tools_to_openai_format,
    display_trace,
    format_trace,
)

__all__ = [
    # Models
    "Message",
    "ToolCall",
    "ToolResult",
    "ContentItem",
    "Event",
    "ExecutionContext",
    "Session",
    "ToolConfirmation",
    "PendingToolCall",
    "BaseSessionManager",
    "InMemorySessionManager",
    # Tools
    "BaseTool",
    "FunctionTool",
    "tool",
    # LLM
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    "build_messages",
    # Agent
    "Agent",
    "AgentResult",
    # MCP
    "load_mcp_tools",
    # Memory
    "count_tokens",
    "apply_sliding_window",
    "apply_compaction",
    "apply_summarization",
    "ContextOptimizer",
    "create_optimizer_callback",
    # Utils
    "function_to_input_schema",
    "format_tool_definition",
    "function_to_tool_definition",
    "mcp_tools_to_openai_format",
    "display_trace",
    "format_trace",
]

__version__ = "0.1.0"

