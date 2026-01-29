# Examples

This directory contains example scripts demonstrating how to use the agent framework.

## GAIA Evaluation

`gaia_evaluation.py` - Example of running GAIA benchmark evaluation with the agent framework.

### Usage

```python
from examples.gaia_evaluation import (
    create_gaia_agent,
    run_experiment,
    solve_problem,
    display_trace
)
from agent_framework import load_mcp_tools
import os

# Load tools
tavily_connection = {
    "command": "npx",
    "args": ["-y", "tavily-mcp@latest"],
    "env": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
}
mcp_tools = await load_mcp_tools(tavily_connection)

# Run experiment
results = await run_experiment(
    problems=list(problems),
    models=["gpt-5", "gpt-5-mini"],
    tools=mcp_tools,
)

# Or solve a single problem
agent = create_gaia_agent("gpt-5-mini", mcp_tools)
result = await agent.run("Your question here...")
display_trace(result.context)
```
