# Episode 3: Core Data Models

**Duration**: 30 minutes  
**What to Build**: `agent_framework/models.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Complete data model structure
- Type-safe message handling
- Execution tracking

**Hook Statement**: "Today we'll build the data structures that power our entire agent framework. These models ensure type safety and make our code predictable."

---

### 2. Problem (3 min)
**Why do we need structured data models?**

**The Challenge:**
- Raw dictionaries are error-prone
- No validation
- Hard to understand
- Easy to make mistakes

**The Solution:**
- Pydantic models for validation
- Type hints for clarity
- Structured data flow
- Self-documenting code

---

### 3. Concept: Building Our Models (20 min)

#### 3.1 Message Model (3 min)

**What is a Message?**
- Text content in conversation
- Has a role (system, user, assistant, tool)
- Simple but critical

**Implementation:**
```python
from pydantic import BaseModel
from typing import Literal

class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant", "tool"]
    content: str
```

**Why Literal?**
- Only specific values allowed
- Catches typos at runtime
- Self-documenting

**Live Coding**: Build Message model

---

#### 3.2 ToolCall Model (3 min)

**What is a ToolCall?**
- LLM's request to execute a tool
- Contains tool name and arguments
- Has unique ID for tracking

**Implementation:**
```python
class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict
```

**Key Fields:**
- `tool_call_id`: Links to ToolResult
- `name`: Which tool to call
- `arguments`: Parameters for the tool

**Live Coding**: Build ToolCall model

---

#### 3.3 ToolResult Model (3 min)

**What is a ToolResult?**
- Outcome of tool execution
- Success or error status
- Contains output or error message

**Implementation:**
```python
class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list
```

**Why list for content?**
- Can have multiple outputs
- Flexible for different tools
- Matches API format

**Live Coding**: Build ToolResult model

---

#### 3.4 ContentItem Union (2 min)

**What is ContentItem?**
- Union type for all content types
- Used in events and requests
- Type-safe polymorphism

**Implementation:**
```python
ContentItem = Union[Message, ToolCall, ToolResult]
```

**Why Union?**
- Events can contain any content type
- Type checker understands all possibilities
- Runtime validation via Pydantic

**Live Coding**: Define ContentItem

---

#### 3.5 Event Model (4 min)

**What is an Event?**
- Recorded step in execution
- Contains one or more content items
- Has timestamp and author

**Implementation:**
```python
import uuid
from datetime import datetime

class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)
```

**Key Points:**
- `default_factory` for unique IDs
- Timestamp for ordering
- Author tracks who created it
- Content list can be empty

**Live Coding**: Build Event model

---

#### 3.6 ExecutionContext Dataclass (5 min)

**What is ExecutionContext?**
- Central state container
- Mutable (needs to be dataclass)
- Tracks entire execution

**Implementation:**
```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ExecutionContext:
    """Central storage for all execution state."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    session_id: Optional[str] = None
    
    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)
    
    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1
```

**Why Dataclass?**
- Mutable state needs to be lightweight
- No validation needed (internal use)
- Better performance

**Live Coding**: Build ExecutionContext

---

### 4. Testing Our Models (3 min)

**Test Message:**
```python
msg = Message(role="user", content="Hello")
print(msg.role)  # "user"
print(msg.type)  # "message"
```

**Test ToolCall:**
```python
tool_call = ToolCall(
    tool_call_id="call_123",
    name="calculator",
    arguments={"expression": "2+2"}
)
print(tool_call.name)  # "calculator"
```

**Test Event:**
```python
event = Event(
    execution_id="exec_123",
    author="user",
    content=[Message(role="user", content="Hello")]
)
print(len(event.content))  # 1
```

**Test ExecutionContext:**
```python
context = ExecutionContext()
context.add_event(event)
print(context.current_step)  # 0
context.increment_step()
print(context.current_step)  # 1
```

---

### 5. Why Pydantic vs Dataclass? (2 min)

**Pydantic (Message, ToolCall, ToolResult, Event):**
- Data crossing boundaries
- Needs validation
- Serialization required
- Immutable by default

**Dataclass (ExecutionContext):**
- Internal mutable state
- No validation needed
- Performance critical
- Frequent updates

**Rule of Thumb:**
- External data → Pydantic
- Internal state → Dataclass

---

### 6. Demo: Complete Models (2 min)

**Show:**
- All models working together
- Type safety in action
- Validation catching errors
- Serialization working

---

### 7. Next Steps (1 min)

**Preview Episode 4:**
- Building the LLM client
- Converting our models to API format
- Parsing API responses

**What We Built:**
- Complete data model structure
- Type-safe message handling
- Execution tracking

---

## Key Takeaways

1. **Pydantic** validates data at runtime
2. **Literal types** constrain values
3. **Union types** enable polymorphism
4. **Field(default_factory=...)** prevents shared defaults
5. **Dataclass** for mutable internal state

---

## Common Mistakes

**Mistake 1: Mutable default arguments**
```python
# Wrong
class BadEvent(BaseModel):
    content: List[str] = []  # Shared across instances!

# Right
class GoodEvent(BaseModel):
    content: List[str] = Field(default_factory=list)  # New list each time
```

**Mistake 2: Missing type hints**
```python
# Wrong - no type safety
def process(event):
    return event.content

# Right - type checker helps
def process(event: Event) -> List[ContentItem]:
    return event.content
```

---

## Exercises

1. Add a `metadata` field to Event
2. Create a `UserMessage` model that extends Message
3. Add validation to ensure content is not empty
4. Create a helper function to extract all messages from events

---

**Previous Episode**: [Episode 2: Your First LLM Call](./EPISODE_02_LLM_CALL.md)  
**Next Episode**: [Episode 4: The LLM Client](./EPISODE_04_LLM_CLIENT.md)

