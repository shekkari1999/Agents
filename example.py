"""Example usage of the agent framework."""

import asyncio
import os
from dotenv import load_dotenv
from agent_framework import Agent, LlmClient, FunctionTool, load_mcp_tools

load_dotenv()


# Example 1: Simple calculator tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)


# Example 2: Using the @tool decorator
from agent_framework import tool

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # This is a placeholder - in real usage, you'd call an actual search API
    return f"Search results for: {query}"


async def main():
    # Create a calculator tool
    calc_tool = FunctionTool(calculator)
    
    # Create the agent
    agent = Agent(
        model=LlmClient(model="gpt-5-mini"),
        tools=[calc_tool, search_web],
        instructions="You are a helpful assistant that can calculate and search the web.",
    )
    
    # Run the agent
    result = await agent.run("What is 1234 * 5678?")
    print(f"Result: {result.output}")
    print(f"Steps taken: {result.context.current_step}")
    
    # Example with MCP tools
    if os.getenv("TAVILY_API_KEY"):
        connection = {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
        }
        mcp_tools = await load_mcp_tools(connection)
        
        agent_with_mcp = Agent(
            model=LlmClient(model="gpt-5-mini"),
            tools=[calc_tool, *mcp_tools],
            instructions="You are a helpful assistant with web search capabilities.",
        )
        
        result = await agent_with_mcp.run("What is the capital of France?")
        print(f"Result: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())

