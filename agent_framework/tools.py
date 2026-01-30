"""Tool system for the agent framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import inspect
from .models import ExecutionContext
from .utils import function_to_input_schema, format_tool_definition


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(
        self, 
        name: str = None, 
        description: str = None, 
        tool_definition: Dict[str, Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
    
    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition
    
    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass
    
    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""
    
    def __init__(
        self, 
        func: Callable, 
        name: str = None, 
        description: str = None,
        tool_definition: Dict[str, Any] = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters
        
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()
        
        super().__init__(
            name=self.name, 
            description=self.description, 
            tool_definition=tool_definition
        )
    
    async def execute(self, context: ExecutionContext = None, **kwargs) -> Any:
        """Execute the wrapped function.
        
        Context is only required if the wrapped function has a 'context' parameter.
        """
        if self.needs_context:
            if context is None:
                raise ValueError(
                    f"Tool '{self.name}' requires a context parameter. "
                    f"Please provide an ExecutionContext instance."
                )
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)
        
        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result
    
    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Dict[str, Any] = None
):
    """Decorator to convert a function into a FunctionTool.
    
    Usage:
        @tool
        def my_function(x: int) -> int:
            return x * 2
        
        # Or with parameters:
        @tool(name="custom_name", description="Custom description")
        def my_function(x: int) -> int:
            return x * 2
    """
    from typing import Union
    
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            tool_definition=tool_definition
        )
    
    if func is not None:
        return decorator(func)
    return decorator