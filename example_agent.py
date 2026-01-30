"""Simple example to test the agent framework.

This script demonstrates basic agent usage with tools.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_framework import Agent, LlmClient, display_trace
from agent_tools import calculator, search_web


async def main():
    """Run a simple agent example."""
    
    print("=" * 60)
    print("Agent Framework - Simple Test")
    print("=" * 60)
    print()
    
    # Create agent with calculator and web search tools
    agent = Agent(
        model=LlmClient(model="gpt-4o-mini"),  # Use a cost-effective model for testing
        tools=[calculator, search_web],
        instructions="You are a helpful assistant. Use websearch tool to search web for sure.",
        max_steps=10
    )
 
    result1 = await agent.run("What are the finalists of australian open 2026 mens singles")
    print(f"\nAnswer: {result1.output}")
    print(f"Steps taken: {result1.context}")
    
  

if __name__ == "__main__":
    asyncio.run(main())

