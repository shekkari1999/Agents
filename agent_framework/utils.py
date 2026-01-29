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


def display_trace(context):
    """Display the execution trace of an agent run.
    
    Args:
        context: ExecutionContext to display
    """
    from .models import Event, Message, ToolCall, ToolResult
    
    print(f"\n{'='*60}")
    print(f"Execution Trace (ID: {context.execution_id})")
    print(f"{'='*60}\n")
    
    for i, event in enumerate(context.events, 1):
        print(f"Step {i} - {event.author.upper()} ({event.timestamp:.2f})")
        print(f"{'-'*60}")
        
        for item in event.content:
            if isinstance(item, Message):
                content_preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
                print(f"  [Message] ({item.role}): {content_preview}")
            elif isinstance(item, ToolCall):
                print(f"  [Tool Call] {item.name}")
                print(f"     Arguments: {item.arguments}")
            elif isinstance(item, ToolResult):
                status_marker = "[SUCCESS]" if item.status == "success" else "[ERROR]"
                print(f"  {status_marker} Tool Result: {item.name} ({item.status})")
                if item.content:
                    content_preview = str(item.content[0])[:100]
                    if len(str(item.content[0])) > 100:
                        content_preview += "..."
                    print(f"     Output: {content_preview}")
        
        print()
    
    print(f"{'='*60}")
    print(f"Final Result: {context.final_result}")
    print(f"Total Steps: {context.current_step}")
    print(f"{'='*60}\n")
