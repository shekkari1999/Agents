"""Example: Running GAIA evaluation with a subset of problems."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from gaia import run_experiment, DEFAULT_MODELS


async def main():
    """Example: Run GAIA evaluation on a small subset."""
    # Load GAIA dataset
    level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    print(f"Number of Level 1 problems: {len(level1_problems)}")
    
    # Select a subset (first 20 problems)
    subset = level1_problems.select(range(20))
    
    # Run experiment with default models
    results = await run_experiment(subset, DEFAULT_MODELS)
    
    # Print results
    print("\nResults:")
    for model, model_results in results.items():
        total = len(model_results)
        correct = sum(1 for r in model_results if r.get("correct", False))
        print(f"{model}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

