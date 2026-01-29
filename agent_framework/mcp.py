"""MCP (Model Context Protocol) tool integration."""

import os
from typing import Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools import BaseTool, FunctionTool


def _extract_text_content(result) -> str:
    """Extract text content from MCP tool result."""
    if not hasattr(result, 'content'):
        return str(result)
    
    texts = []
    for item in result.content:
        if hasattr(item, 'text'):
            texts.append(item.text)
        else:
            texts.append(str(item))
    
    return "\n\n".join(texts)


async def load_mcp_tools(connection: Dict) -> List[BaseTool]:
    """Load tools from an MCP server and convert to FunctionTools.
    
    Args:
        connection: Dictionary with connection parameters:
            - command: Command to run the MCP server
            - args: Arguments for the command
            - env: Environment variables (optional)
    
    Returns:
        List of BaseTool instances wrapping MCP tools
    
    Example:
        connection = {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
        }
        tools = await load_mcp_tools(connection)
    """
    tools = []
    
    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            
            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)
    
    return tools


def _create_mcp_tool(mcp_tool, connection: Dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""
    
    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)
    
    tool_definition = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        }
    }
    
    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition
    )
