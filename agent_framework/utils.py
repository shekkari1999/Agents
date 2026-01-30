"""Utility functions for the agent framework."""

import inspect
from typing import Dict, Any


def function_to_input_schema(func) -> dict:
    """Convert a function signature to JSON Schema input format."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
    
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]
    
    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }


def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    """Format a tool definition in OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def function_to_tool_definition(func) -> dict:
    """Convert a function to OpenAI tool definition format."""
    return format_tool_definition(
        func.__name__,
        func.__doc__ or "",
        function_to_input_schema(func)
    )


def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
    """Convert MCP tool definitions to OpenAI tool format."""
    return [
        format_tool_definition(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
        )
        for tool in mcp_tools.tools
    ]


def format_trace(context) -> str:
    """Format execution trace as a string.
    
    Args:
        context: ExecutionContext to format
        
    Returns:
        Formatted trace string
    """
    from .models import Message, ToolCall, ToolResult
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"Execution Trace (ID: {context.execution_id})")
    lines.append("=" * 60)
    lines.append("")
    
    for i, event in enumerate(context.events, 1):
        lines.append(f"Step {i} - {event.author.upper()} ({event.timestamp:.2f})")
        lines.append("-" * 60)
        
        for item in event.content:
            if isinstance(item, Message):
                content_preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
                lines.append(f"  [Message] ({item.role}): {content_preview}")
            elif isinstance(item, ToolCall):
                lines.append(f"  [Tool Call] {item.name}")
                lines.append(f"     Arguments: {item.arguments}")
            elif isinstance(item, ToolResult):
                status_marker = "[SUCCESS]" if item.status == "success" else "[ERROR]"
                lines.append(f"  {status_marker} Tool Result: {item.name} ({item.status})")
                if item.content:
                    content_preview = str(item.content[0])[:100]
                    if len(str(item.content[0])) > 100:
                        content_preview += "..."
                    lines.append(f"     Output: {content_preview}")
        
        lines.append("")
    
    lines.append("=" * 60)
    lines.append(f"Final Result: {context.final_result}")
    lines.append(f"Total Steps: {context.current_step}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def display_trace(context):
    """Display the execution trace of an agent run.
    
    Args:
        context: ExecutionContext to display
    """
    print(format_trace(context))
