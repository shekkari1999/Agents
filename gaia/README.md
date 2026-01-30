# GAIA Benchmark Evaluation

This directory contains code for evaluating models on the GAIA benchmark.

## Structure

```
gaia/
├── models.py          # Data models (GaiaOutput)
├── config.py         # Configuration (prompt, semaphores, default models)
├── evaluator.py      # Core evaluation functions
├── run.py            # Main script to run full evaluation
├── example.py        # Simple example usage
├── gaia_evaluation.py # Legacy example using agent framework
└── README.md         # This file
```

## Quick Start

### Basic Usage

```python
from datasets import load_dataset
from gaia import run_experiment, DEFAULT_MODELS

# Load dataset
level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")

# Select subset
subset = level1_problems.select(range(20))

# Run evaluation
results = await run_experiment(subset, DEFAULT_MODELS)
```

### Using Custom Models

```python
from gaia import run_experiment

models = ["gpt-5", "gpt-5-mini"]
results = await run_experiment(subset, models)
```

### Running the Main Script

```bash
python -m gaia.run
```

### Running the Example

```bash
python gaia/example.py
```

## Components

### Models (`models.py`)
- `GaiaOutput`: Pydantic model for structured output
  - `is_solvable`: Whether the problem can be solved
  - `unsolvable_reason`: Explanation if unsolvable
  - `final_answer`: The answer if solvable

### Config (`config.py`)
- `GAIA_PROMPT`: System prompt for GAIA evaluation
- `PROVIDER_SEMAPHORES`: Rate limiting semaphores for API providers
- `DEFAULT_MODELS`: Default list of models to evaluate

### Evaluator (`evaluator.py`)
- `solve_problem()`: Solve a single problem with structured output
- `evaluate_gaia_single()`: Evaluate one problem-model pair
- `run_experiment()`: Run full evaluation on multiple problems and models
- `is_correct()`: Check if prediction matches answer
- `get_provider()`: Extract provider from model name

## Example Output

The evaluation returns a dictionary mapping model names to lists of results:

```python
{
    "gpt-5": [
        {
            "task_id": "...",
            "model": "gpt-5",
            "correct": True,
            "is_solvable": True,
            "prediction": "17",
            "answer": "17",
            "unsolvable_reason": ""
        },
        ...
    ],
    "gpt-5-mini": [...]
}
```

## Rate Limiting

The code uses semaphores to limit concurrent API calls:
- OpenAI: 30 concurrent requests
- Anthropic: 10 concurrent requests

Adjust these in `config.py` if needed.
