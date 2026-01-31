# Episode 6: Building the Tool System

**Duration**: 40 minutes  
**What to Build**: `agent_framework/tools.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Tool system that wraps functions
- Automatic schema generation
- @tool decorator
- Tool confirmation for dangerous operations

**Hook Statement**: "Today we'll build the system that lets LLMs use Python functions. This is what makes agents powerful - they can actually do things!"

---

### 2. Problem (3 min)
**Why do we need a tool system?**

**The Challenge:**
- LLMs can't execute code directly
- Need to bridge Python functions to LLM
- Need to describe functions to LLM
- Need to handle execution safely
- Some tools are dangerous (delete file, send email)

**The Solution:**
- BaseTool abstract interface
- FunctionTool wrapper
- Automatic schema generation
- @tool decorator for ease
- Confirmation workflow for safety

---

### 3. Concept: Tool Architecture (5 min)

**The Flow:**
1. Define Python function
2. Wrap as BaseTool
3. Generate JSON schema
4. Send to LLM
5. LLM calls tool
6. Check if confirmation required
7. Execute function (or wait for approval)
8. Return result

**Key Components:**
- BaseTool: Abstract interface
- FunctionTool: Wraps functions
- Schema generation: From type hints
- Decorator: Syntactic sugar
- Confirmation: Safety for dangerous tools

---

### 4. Live Coding: Building the Tool System (30 min)

#### Step 1: BaseTool Abstract Class (7 min)
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
        tool_definition: Dict[str, Any] = None,
        # Confirmation support
        requires_confirmation: bool = False,
        confirmation_message_template: str = None
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = confirmation_message_template or (
            "The agent wants to execute '{name}' with arguments: {arguments}. "
            "Do you approve?"
        )
    
    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition
    
    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass
    
    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)

    def get_confirmation_message(self, arguments: dict[str, Any]) -> str:
        """Generate a confirmation message for this tool call."""
        return self.confirmation_message_template.format(
            name=self.name,
            arguments=arguments
        )
```

**Key Points:**
- Abstract base class
- Name and description
- Tool definition (JSON schema)
- Execute method
- **NEW: `requires_confirmation`** - marks dangerous tools
- **NEW: `confirmation_message_template`** - customizable message
- **NEW: `get_confirmation_message()`** - generates message for user

**Why Confirmation?**
- Some tools are dangerous (delete files, send emails)
- Users should approve before execution
- Allows argument modification

**Live Coding**: Build BaseTool with confirmation

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
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters
        
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()
        
        super().__init__(
            name=self.name, 
            description=self.description, 
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template
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
```

**Key Points:**
- Wraps any function
- Detects context parameter
- Handles sync/async
- Auto-generates schema
- **Passes confirmation params to parent**

**Live Coding**: Build FunctionTool

---

#### Step 4: @tool Decorator with Confirmation (6 min)
```python
def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Dict[str, Any] = None,
    requires_confirmation: bool = False,
    confirmation_message: str = None
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
        
        # With confirmation:
        @tool(requires_confirmation=True, confirmation_message="Delete file?")
        def delete_file(filename: str) -> str:
            ...
    """
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message
        )
    
    if func is not None:
        return decorator(func)
    return decorator
```

**Usage Examples:**

**Simple Tool (no confirmation):**
```python
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))
```

**Dangerous Tool (requires confirmation):**
```python
@tool(
    requires_confirmation=True,
    confirmation_message="Delete file '{arguments[filename]}'? This cannot be undone."
)
def delete_file(filename: str) -> str:
    """Delete a file from the filesystem."""
    import os
    os.remove(filename)
    return f"Deleted {filename}"
```

**Custom Confirmation Message:**
```python
@tool(
    requires_confirmation=True,
    confirmation_message="Send email to {arguments[recipient]}? Subject: {arguments[subject]}"
)
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email."""
    # ... send email ...
    return "Email sent"
```

**Live Coding**: Build @tool decorator with confirmation

---

#### Step 5: Testing Tools (4 min)

**Simple Tool:**
```python
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

print(add.name)  # "add"
print(add.requires_confirmation)  # False
result = await add.execute(context=None, a=2, b=3)
print(result)  # 5
```

**Tool with Context:**
```python
@tool
def get_step_count(context: ExecutionContext) -> int:
    """Get current step count."""
    return context.current_step

print(get_step_count.needs_context)  # True
```

**Tool with Confirmation:**
```python
@tool(
    requires_confirmation=True,
    confirmation_message="Delete '{arguments[filename]}'?"
)
def delete_file(filename: str) -> str:
    """Delete a file."""
    return f"Deleted {filename}"

print(delete_file.requires_confirmation)  # True

# Generate confirmation message
message = delete_file.get_confirmation_message({"filename": "secret.txt"})
print(message)  # "Delete 'secret.txt'?"
```

**Test Schema:**
```python
print(add.tool_definition)
# {
#     "type": "function",
#     "function": {
#         "name": "add",
#         "description": "Add two numbers.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "a": {"type": "integer"},
#                 "b": {"type": "integer"}
#             },
#             "required": ["a", "b"]
#         }
#     }
# }
```

---

### 5. Demo: Creating Tools (2 min)

**Show:**
- Simple calculator tool
- Tool with context
- Tool with confirmation
- Schema generation
- Confirmation message generation

---

### 6. Next Steps (1 min)

**Preview Episode 7:**
- Integrating tools into agent
- Tool execution in agent loop
- **Handling pending tool calls**
- **Processing confirmations**

**What We Built:**
- Complete tool system
- Schema generation
- @tool decorator
- Confirmation workflow

---

## Key Takeaways

1. **BaseTool** provides abstract interface
2. **FunctionTool** wraps any function
3. **Schema generation** from type hints
4. **@tool decorator** for ease of use
5. **Context-aware** tools supported
6. **`requires_confirmation`** marks dangerous tools
7. **`get_confirmation_message()`** generates user-facing message

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

**Mistake 3: Forgetting confirmation for dangerous tools**
```python
# Wrong - dangerous tool without confirmation
@tool
def delete_file(filename: str) -> str:
    os.remove(filename)
    return "Deleted"

# Right - requires user approval
@tool(requires_confirmation=True)
def delete_file(filename: str) -> str:
    os.remove(filename)
    return "Deleted"
```

**Mistake 4: Bad confirmation message**
```python
# Wrong - generic message
@tool(
    requires_confirmation=True,
    confirmation_message="Are you sure?"
)
def delete_file(filename: str) -> str: ...

# Right - specific and informative
@tool(
    requires_confirmation=True,
    confirmation_message="Delete file '{arguments[filename]}'? This cannot be undone."
)
def delete_file(filename: str) -> str: ...
```

---

## Exercises

1. **Add `requires_confirmation` to a tool**: Create a `send_email` tool that requires confirmation
2. **Custom confirmation message**: Create a message template that includes all arguments
3. **Create tool registry**: Build a `ToolRegistry` class that tracks all tools and their confirmation status
4. **Add validation**: Add a method to validate arguments before execution

---

## Complete tools.py File

```python
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
        requires_confirmation: bool = False,
        confirmation_message_template: str = None
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = confirmation_message_template or (
            "The agent wants to execute '{name}' with arguments: {arguments}. "
            "Do you approve?"
        )
    
    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition
    
    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass
    
    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)

    def get_confirmation_message(self, arguments: dict[str, Any]) -> str:
        """Generate a confirmation message for this tool call."""
        return self.confirmation_message_template.format(
            name=self.name,
            arguments=arguments
        )


class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""
    
    def __init__(
        self, 
        func: Callable, 
        name: str = None, 
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters
        
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()
        
        super().__init__(
            name=self.name, 
            description=self.description, 
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template
        )
    
    async def execute(self, context: ExecutionContext = None, **kwargs) -> Any:
        if self.needs_context:
            if context is None:
                raise ValueError(f"Tool '{self.name}' requires a context parameter.")
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)
        
        if inspect.iscoroutine(result):
            return await result
        return result
    
    def _generate_definition(self) -> Dict[str, Any]:
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Dict[str, Any] = None,
    requires_confirmation: bool = False,
    confirmation_message: str = None
):
    """Decorator to convert a function into a FunctionTool."""
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message
        )
    
    if func is not None:
        return decorator(func)
    return decorator
```

---

**Previous Episode**: [Episode 5: The Basic Agent Loop](./EPISODE_05_AGENT_LOOP.md)  
**Next Episode**: [Episode 7: Tool Execution & Complete Agent](./EPISODE_07_TOOL_EXECUTION.md)
