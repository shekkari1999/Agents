"""Demo script showing agent usage with structured output."""

import asyncio
import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add parent directory to path so we can import agent_framework
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework import Agent, LlmClient, display_trace
from dotenv import load_dotenv

load_dotenv()


# Define output structure
class AnswerOutput(BaseModel):
    """Structured output for the answer."""
    final_answer: str = Field(description="The final answer to the question")


async def main():
    # Create agent with structured output and verbose mode enabled
    agent = Agent(
        model=LlmClient(model="gpt-5-mini"),
        tools=[],
        instructions="You are a helpful assistant that answers questions accurately.",
        output_type=AnswerOutput,
        verbose=True,  # Enable verbose mode to see thinking process
    )
    
    print("Starting agent execution...")
    print("=" * 60)
    
    result = await agent.run(
        "If Eliud Kipchoge could maintain his marathon pace, "
        "how many thousand hours to reach the Moon?"
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Answer: {result.output.final_answer}")
    print(f"Steps taken: {result.context.current_step}")
    print("=" * 60)
    
    # Optionally show full trace
    print("\nFull Execution Trace:")
    display_trace(result.context)


if __name__ == "__main__":
    asyncio.run(main())