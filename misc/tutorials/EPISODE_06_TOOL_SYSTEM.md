# Episode 6: Building the Tool System

**Duration**: 35 minutes  
**What to Build**: `agent_framework/tools.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Tool system that wraps functions
- Automatic schema generation
- @tool decorator

**Hook Statement**: "Today we'll build the system that lets LLMs use Python functions. This is what makes agents powerful - they can actually do things!"

---

### 2. Problem (3 min)
**Why do we need a tool system?**

**The Challenge:**
- LLMs can't execute code directly
- Need to bridge Python functions to LLM
- Need to describe functions to LLM
- Need to handle execution safely

**The Solution:**
- BaseTool abstract interface
- FunctionTool wrapper
- Automatic schema generation
- @tool decorator for ease

---

### 3. Concept: Tool Architecture (5 min)

**The Flow:**
1. Define Python function
2. Wrap as BaseTool
3. Generate JSON schema
4. Send to LLM
5. LLM calls tool
6. Execute function
7. Return result

**Key Components:**
- BaseTool: Abstract interface
- FunctionTool: Wraps functions
- Schema generation: From type hints
- Decorator: Syntactic sugar

---

### 4. Live Coding: Building the Tool System (25 min)

#### Step 1: BaseTool Abstract Class (5 min)
```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from .models import ExecutionContext

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(
        self, 
        name: str = None, 
        description: str = None, 
        tool_definition: Dict[str, Any] = None
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
```

**Key Points:**
- Abstract base class
- Name and description
- Tool definition (JSON schema)
- Execute method

**Live Coding**: Build BaseTool

---

#### Step 2: Schema Generation Utilities (5 min)
```python
import inspect
from .utils import function_to_input_schema, format_tool_definition

def function_to_input_schema(func) -> dict:
    """Convert function signature to JSON Schema."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    
    signature = inspect.signature(func)
    parameters = {}
    
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}
    
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]
    
    return {
        "type": "object",
        "properties": parameters,
        "required": required,
    }

def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
    """Format tool definition in OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
```

**Key Points:**
- Inspects function signature
- Maps Python types to JSON Schema
- Handles required parameters
- Formats for OpenAI

**Live Coding**: Build schema generation

---

#### Step 3: FunctionTool Class (8 min)
```python
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
        """Execute the wrapped function."""
        if self.needs_context:
            if context is None:
                raise ValueError(f"Tool '{self.name}' requires a context parameter.")
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
```

**Key Points:**
- Wraps any function
- Detects context parameter
- Handles sync/async
- Auto-generates schema

**Live Coding**: Build FunctionTool

---

#### Step 4: @tool Decorator (4 min)
```python
def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Dict[str, Any] = None
):
    """Decorator to convert a function into a FunctionTool."""
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
```

**Usage:**
```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))
```

**Live Coding**: Build @tool decorator

---

#### Step 5: Testing Tools (3 min)
```python
# Simple tool
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Tool with context
@tool
def get_step_count(context: ExecutionContext) -> int:
    """Get current step count."""
    return context.current_step

# Test
tool = add
print(tool.name)  # "add"
print(tool.description)  # "Add two numbers"
print(tool.tool_definition)  # JSON schema

# Execute
result = await tool.execute(context=None, a=2, b=3)
print(result)  # 5
```

---

### 5. Demo: Creating Tools (2 min)

**Show:**
- Simple calculator tool
- Tool with context
- Schema generation
- Tool execution

---

### 6. Next Steps (1 min)

**Preview Episode 7:**
- Integrating tools into agent
- Tool execution in agent loop
- Handling tool results

**What We Built:**
- Complete tool system
- Schema generation
- @tool decorator

---

## Key Takeaways

1. **BaseTool** provides abstract interface
2. **FunctionTool** wraps any function
3. **Schema generation** from type hints
4. **@tool decorator** for ease of use
5. **Context-aware** tools supported

---

## Common Mistakes

**Mistake 1: Forgetting docstring**
```python
# Wrong - no description for LLM
@tool
def add(a: int, b: int) -> int:
    return a + b

# Right - LLM knows what it does
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

**Mistake 2: Not handling async functions**
```python
# Wrong - doesn't await
result = self.func(**kwargs)
return result

# Right - checks if coroutine
result = self.func(**kwargs)
if inspect.iscoroutine(result):
    return await result
return result
```

---

## Exercises

1. Add support for Optional types
2. Implement custom type mappings
3. Add tool validation
4. Create a tool registry

---

**Previous Episode**: [Episode 5: The Basic Agent Loop](./EPISODE_05_AGENT_LOOP.md)  
**Next Episode**: [Episode 7: Tool Execution & Complete Agent](./EPISODE_07_TOOL_EXECUTION.md)

