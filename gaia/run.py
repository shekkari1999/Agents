"""Main script to run GAIA benchmark evaluation."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from .evaluator import run_experiment
from .config import DEFAULT_MODELS


async def main():
    """Run GAIA benchmark evaluation."""
    # Load GAIA dataset
    print("Loading GAIA dataset...")
    level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    print(f"Number of Level 1 problems: {len(level1_problems)}")
    
    # Select subset for testing (adjust range as needed)
    subset = level1_problems.select(range(20))
    print(f"Evaluating on {len(subset)} problems...")
    
    # Run experiment
    print(f"Models: {DEFAULT_MODELS}")
    results = await run_experiment(subset, DEFAULT_MODELS)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    for model, model_results in results.items():
        total = len(model_results)
        correct = sum(1 for r in model_results if r.get("correct", False))
        solvable = sum(1 for r in model_results if r.get("is_solvable", False))
        accuracy = (correct / total * 100) if total > 0 else 0
        solvable_rate = (solvable / total * 100) if total > 0 else 0
        
        print(f"\n{model}:")
        print(f"  Total problems: {total}")
        print(f"  Correct: {correct} ({accuracy:.1f}%)")
        print(f"  Solvable: {solvable} ({solvable_rate:.1f}%)")
    
    print("\n" + "="*60)
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())

