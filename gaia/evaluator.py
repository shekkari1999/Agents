"""GAIA benchmark evaluation functions."""

import asyncio
from typing import Dict, List, Optional
from litellm import acompletion
from tqdm.asyncio import tqdm as tqdm_asyncio

from .models import GaiaOutput
from .config import GAIA_PROMPT, PROVIDER_SEMAPHORES


def get_provider(model: str) -> str:
    """Extract provider name from model string.
    
    Args:
        model: Model name (e.g., "gpt-5", "anthropic/claude-sonnet-4-5")
    
    Returns:
        Provider name ("anthropic" or "openai")
    """
    return "anthropic" if model.startswith("anthropic/") else "openai"


async def solve_problem(model: str, question: str) -> GaiaOutput:
    """Solve a single problem and return structured output.
    
    Args:
        model: Model name to use for solving
        question: The GAIA problem question
    
    Returns:
        GaiaOutput with is_solvable, final_answer, and unsolvable_reason
    """
    provider = get_provider(model)
    
    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": GAIA_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content
        
        if finish_reason == "refusal" or content is None:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                final_answer=""
            )
        return GaiaOutput.model_validate_json(content)


def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive).
    
    Args:
        prediction: Model's predicted answer
        answer: Ground truth answer
    
    Returns:
        True if prediction matches answer (case-insensitive), False otherwise
    """
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    """Evaluate a single problem-model pair and return result.
    
    Args:
        problem: GAIA problem dictionary with "task_id", "Question", and "Final answer"
        model: Model name to evaluate
    
    Returns:
        Dictionary with evaluation results:
        - task_id: Problem task ID
        - model: Model name
        - correct: Whether prediction matches answer
        - is_solvable: Whether model marked problem as solvable
        - prediction: Model's predicted answer
        - answer: Ground truth answer
        - unsolvable_reason: Reason if marked unsolvable
        - error: Error message if exception occurred
    """
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }


async def run_experiment(
    problems: list[dict],
    models: list[str],
) -> Dict[str, List[dict]]:
    """Evaluate all models on all problems.
    
    Args:
        problems: List of GAIA problem dictionaries
        models: List of model names to evaluate
    
    Returns:
        Dictionary mapping model names to lists of evaluation results
    """
    tasks = [
        evaluate_gaia_single(problem, model)
        for problem in problems
        for model in models
    ]
    
    all_results = await tqdm_asyncio.gather(*tasks)
    
    # Group results by model
    results = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)
    
    return results

