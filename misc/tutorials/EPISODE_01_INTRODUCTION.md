# Episode 1: Introduction & Python Foundations

**Duration**: 30 minutes  
**What to Build**: None (concepts only)  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Quick demo of the final agent framework
- Agent using calculator and web search
- Session persistence demo
- Web application preview

**Hook Statement**: "By the end of this series, you'll have built a complete AI agent framework from scratch that can reason, use tools, remember conversations, and be deployed as a web app."

---

### 2. Problem (3 min)
**Why build an agent framework?**

**The Problem:**
- LLMs are powerful but limited to their training data
- They can't access real-time information
- They can't perform actions (file operations, API calls, etc.)
- They need structure to be reliable

**The Solution:**
- Agent framework that gives LLMs tools
- Multi-step reasoning loop
- State management
- Extensible architecture

**Real-world Use Cases:**
- Customer support bots with database access
- Research assistants that can search the web
- Code assistants that can read/write files
- Data analysis agents that can process files

---

### 3. Concept: Python Patterns We'll Use (20 min)

#### 3.1 Type Hints (5 min)

**Why Type Hints?**
- Self-documenting code
- IDE autocomplete
- Catch errors early
- Better collaboration

**Common Patterns:**
```python
from typing import List, Optional, Literal, Union, Dict, Any

# Basic types
name: str = "Agent"
count: int = 5

# Collections
messages: List[str] = []
config: Dict[str, Any] = {}

# Optional (can be None)
result: Optional[str] = None  # Same as: str | None

# Union (multiple types)
content: Union[str, dict] = "text"  # Same as: str | dict

# Literal (specific values only)
role: Literal["user", "assistant", "system"] = "user"
```

**Live Demo:**
- Show IDE autocomplete with type hints
- Show what happens without type hints
- Explain `Optional` vs required parameters

---

#### 3.2 Pydantic (7 min)

**What is Pydantic?**
- Data validation library
- Runtime type checking
- Automatic serialization

**Basic Example:**
```python
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

# Valid
msg = Message(role="user", content="Hello")
print(msg.role)  # "user"

# Invalid - raises ValidationError
msg = Message(role="user")  # Missing 'content'!
```

**Field Defaults:**
```python
from datetime import datetime
import uuid

class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    content: List[str] = Field(default_factory=list)  # NOT content: List[str] = []
```

**Why Field(default_factory)?**
- Prevents shared mutable defaults
- Each instance gets a new list/dict
- Critical for avoiding bugs

**Serialization:**
```python
# Model to dict
data = msg.model_dump()

# Model to JSON
json_str = msg.model_dump_json()

# Dict to model
msg2 = Message.model_validate({"role": "user", "content": "hi"})
```

**When to Use Pydantic:**
- Data crossing boundaries (API requests/responses)
- User input validation
- Configuration files
- External data

---

#### 3.3 Dataclasses (5 min)

**What are Dataclasses?**
- Lightweight data containers
- Less overhead than Pydantic
- Mutable by default

**Example:**
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ExecutionContext:
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: Event):
        self.events.append(event)
```

**Pydantic vs Dataclass:**
| Feature | Pydantic | Dataclass |
|---------|----------|-----------|
| Validation | Yes | No |
| JSON serialization | Built-in | Manual |
| Performance | Slower | Faster |
| Use Case | External data | Internal state |

**Our Rule:**
- **Pydantic** for data crossing boundaries
- **Dataclass** for internal mutable state

---

#### 3.4 Async/Await (3 min)

**Why Async?**
- LLM API calls take seconds
- Without async, program waits doing nothing
- With async, can do other work while waiting

**Basic Example:**
```python
import asyncio

async def call_llm(prompt: str):
    await asyncio.sleep(2)  # Simulates API call
    return f"Response to: {prompt}"

# Sequential - takes 6 seconds
async def sequential():
    results = []
    for prompt in ["A", "B", "C"]:
        results.append(await call_llm(prompt))
    return results

# Concurrent - takes 2 seconds
async def concurrent():
    results = await asyncio.gather(
        call_llm("A"),
        call_llm("B"),
        call_llm("C")
    )
    return results
```

**Key Rules:**
1. `async` functions must be `await`ed
2. `await` only works inside `async` functions
3. Use `asyncio.run()` at top level

**In Jupyter:**
- Can use `await` directly (event loop already running)

---

### 4. Demo: Working Agent (3 min)

**Show the Final Product:**
```python
from agent_framework import Agent, LlmClient
from agent_tools import calculator, search_web

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator, search_web],
    instructions="You are a helpful assistant.",
    max_steps=10
)

result = await agent.run("What is 123 * 456?")
print(result.output)
```

**What Happens:**
1. Agent receives user input
2. Calls LLM with tools available
3. LLM decides to use calculator
4. Agent executes calculator tool
5. Returns result to user

**Show Trace:**
- Display execution trace
- Show each step
- Highlight tool calls

---

### 5. Next Steps (2 min)

**Preview Episode 2:**
- Making your first LLM API call
- Understanding chat completion format
- Building a simple LLM wrapper

**What to Prepare:**
- Python 3.10+ installed
- OpenAI API key (or other provider)
- Code editor ready

**Repository:**
- GitHub link
- Each episode has a branch
- Follow along with code

---

## Key Takeaways

1. **Type hints** make code self-documenting
2. **Pydantic** validates data at runtime
3. **Dataclasses** are lightweight for internal state
4. **Async/await** enables concurrent operations
5. **Field(default_factory=...)** prevents shared mutable defaults

---

## Common Questions

**Q: Do I need to know advanced Python?**  
A: Intermediate level is enough. Know functions, classes, dictionaries, and basic async.

**Q: Can I use this with local models?**  
A: Yes! LiteLLM supports Ollama and other local providers.

**Q: How is this different from LangChain?**  
A: We're building from scratch to understand every detail. LangChain is great but hides complexity.

**Q: What if I get stuck?**  
A: Each episode builds on the previous. Pause, rewatch, and check the GitHub branches.

---

## Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Async/Await Guide](https://docs.python.org/3/library/asyncio.html)
- [Repository](https://github.com/yourusername/ai-agent-from-scratch)

---

## Exercises (Optional)

1. Create a Pydantic model for a `User` with name, email, and age
2. Create a dataclass for `AppState` with a list of users
3. Write an async function that makes 3 concurrent API calls
4. Experiment with type hints in your IDE

---

**Next Episode**: [Episode 2: Your First LLM Call](./EPISODE_02_LLM_CALL.md)

