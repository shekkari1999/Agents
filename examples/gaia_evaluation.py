"""GAIA benchmark evaluation example using the agent framework."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import agent_framework
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework import Agent, LlmClient, load_mcp_tools, display_trace, tool


# Calculator tool
@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions. Supports basic math operations like +, -, *, /, **, etc."""
    return eval(expression)

async def main():
    """Example usage of GAIA evaluation."""
    # Load MCP tools
    tavily_connection = {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
    }
    
    mcp_tools = await load_mcp_tools(tavily_connection)
    
    # Combine all tools: calculator (already wrapped by @tool decorator) + MCP tools
    all_tools = [calculator] + mcp_tools
    
    # Show available tools
    print(f"\n{'='*60}")
    print(f"Available Tools: {len(all_tools)}")
    print(f"{'='*60}")
    for i, tool_obj in enumerate(all_tools, 1):
        print(f"{i}. {tool_obj.name}")
        if hasattr(tool_obj, 'description'):
            desc = tool_obj.description[:80] + "..." if len(tool_obj.description) > 80 else tool_obj.description
            print(f"   {desc}")
    print(f"{'='*60}\n")
    
    # Create agent with instructions to use web search
    agent = Agent(
        model=LlmClient(model="gpt-5-mini"),
        tools=all_tools,
        instructions="""You are a helpful assistant. You have access to tools.

Do NOT rely solely on your training data. Use the tools when necessary to present accurate information.
Instead of assumptions, use websearch for the questions you don't know exact answer to
""",
        max_steps=10,
    )
    
    # Solve a problem
    result = await agent.run(
        'If A is usain bolt\'s world record in 100 meters, B is usain bolt\'s fastest time in 200 meters, what is A x B ?'
    )
    
    print(f"\n{'='*60}")
    print(f"Final Answer: {result.output}")
    print(f"Steps: {result.context.current_step}")
    print(f"{'='*60}\n")
    
    # Display execution trace
    display_trace(result.context)


if __name__ == "__main__":
    asyncio.run(main())

