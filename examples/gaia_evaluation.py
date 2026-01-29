"""GAIA benchmark evaluation example using the agent framework."""

import asyncio
import os
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

# Add parent directory to path so we can import agent_framework
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework import Agent, LlmClient, AgentResult, load_mcp_tools, display_trace


# GAIA output model
class GaiaOutput(BaseModel):
    """Structured output for GAIA benchmark responses."""
    is_solvable: bool = Field(description="Whether the problem can be solved with available tools")
    unsolvable_reason: str = Field(default="", description="Reason if problem is unsolvable")
    final_answer: str = Field(description="The final answer to the problem")


# GAIA system prompt
gaia_prompt = """

You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending on whether the element is a number or a string.

"""


def create_gaia_agent(model: str, tools: List) -> Agent:
    """Create an agent configured for GAIA benchmark evaluation.
    
    Args:
        model: LLM model name (e.g., "gpt-5", "gpt-5-mini")
        tools: List of tools to provide to the agent
    
    Returns:
        Configured Agent instance
    """
    return Agent(
        model=LlmClient(model=model),
        tools=tools,
        instructions=gaia_prompt,
        output_type=GaiaOutput,
        max_steps=15,
    )


# Semaphore for rate limiting
SEMAPHORE = asyncio.Semaphore(3)


async def solve_problem(agent: Agent, question: str) -> AgentResult:
    """Solve a single GAIA problem with rate limiting.
    
    Args:
        agent: Configured agent instance
        question: Problem question to solve
    
    Returns:
        AgentResult with structured output
    """
    async with SEMAPHORE:
        return await agent.run(question)


async def run_experiment(
    problems: List[dict],
    models: List[str],
    tools: List = None,
) -> dict:
    """Run GAIA evaluation experiment across multiple models.
    
    Args:
        problems: List of problem dictionaries with 'Question' and 'Final answer' keys
        models: List of model names to evaluate
        tools: List of tools to provide to agents
    
    Returns:
        Dictionary mapping model names to lists of results
    """
    tools = tools or []
    results = {model: [] for model in models}
    
    tasks = []
    for problem in problems:
        for model in models:
            agent = create_gaia_agent(model, tools)
            task = solve_problem(agent, problem.get("Question", problem.get("question", "")))
            tasks.append((model, problem, task))
    
    # Execute all tasks
    task_results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
    # Organize results
    for (model, problem, _), result in zip(tasks, task_results):
        if isinstance(result, Exception):
            results[model].append({
                "task_id": problem.get("task_id", problem.get("id", "")),
                "model": model,
                "error": str(result),
            })
        else:
            output = result.output if isinstance(result.output, GaiaOutput) else None
            results[model].append({
                "task_id": problem.get("task_id", problem.get("id", "")),
                "model": model,
                "is_solvable": output.is_solvable if output else None,
                "final_answer": output.final_answer if output else None,
                "unsolvable_reason": output.unsolvable_reason if output else None,
                "correct": (
                    output.final_answer.strip().lower() == problem.get("Final answer", "").strip().lower()
                    if output and "Final answer" in problem
                    else None
                ),
                "steps": result.context.current_step,
            })
    
    return results


async def main():
    """Example usage of GAIA evaluation."""
    # Load MCP tools
    tavily_connection = {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
    }
    
    mcp_tools = await load_mcp_tools(tavily_connection)
    
    # Create agent
    agent = create_gaia_agent("gpt-5-mini", mcp_tools)
    
    # Solve a problem
    result = await agent.run(
        "If Eliud Kipchoge could maintain his marathon pace, "
        "how many thousand hours to reach the Moon?"
    )
    
    if isinstance(result.output, GaiaOutput):
        print(f"Answer: {result.output.final_answer}")
        print(f"Solvable: {result.output.is_solvable}")
        print(f"Steps: {result.context.current_step}")
    else:
        print(f"Answer: {result.output}")
        print(f"Steps: {result.context.current_step}")
    
    # Display execution trace
    display_trace(result.context)


if __name__ == "__main__":
    asyncio.run(main())

