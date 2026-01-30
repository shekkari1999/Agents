"""Callback utilities for agent execution."""

import inspect
from typing import Optional, Callable

from .models import ExecutionContext
from .llm import LlmRequest, LlmResponse
from .memory import count_tokens


def create_optimizer_callback(
    apply_optimization: Callable,
    threshold: int = 50000,
    model_id: str = "gpt-4"
) -> Callable:
    """Factory function that creates a callback applying optimization strategy.
    
    Args:
        apply_optimization: Function that modifies the LlmRequest in place
        threshold: Token count threshold to trigger optimization
        model_id: Model identifier for token counting
    
    Returns:
        Callback function that can be used as before_llm_callback
    """
    async def callback(
        context: ExecutionContext,
        request: LlmRequest
    ) -> Optional[LlmResponse]:
        token_count = count_tokens(request, model_id=model_id)
 
        if token_count < threshold:
            return None
 
        # Support both sync and async functions
        result = apply_optimization(context, request)
        if inspect.isawaitable(result):
            await result
        return None

    return callback
 