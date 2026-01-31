# Episode 3: Core Data Models

**Duration**: 35 minutes  
**What to Build**: `agent_framework/models.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Complete data model structure
- Type-safe message handling
- Execution tracking
- Tool confirmation workflow

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

### 3. Concept: Building Our Models (25 min)

#### 3.1 Message Model (3 min)

**What is a Message?**
- Text content in conversation
- Has a role (system, user, assistant)
- Simple but critical

**Implementation:**
```python
from pydantic import BaseModel
from typing import Literal

class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
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

#### 3.5 ToolConfirmation Model (3 min)

**What is ToolConfirmation?**
- User's decision on a pending tool call
- Can approve or reject
- Can modify arguments before execution

**Implementation:**
```python
class ToolConfirmation(BaseModel):
    """User's decision on a pending tool call."""
    
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None  # Reason for rejection (if not approved)
```

**Why This Model?**
- Some tools are dangerous (delete file, send email)
- Users should approve before execution
- Allows argument modification (e.g., change file path)
- Captures rejection reasons for debugging

**Use Cases:**
- Delete file confirmation
- API call approval
- Email sending confirmation
- Database modification

**Live Coding**: Build ToolConfirmation model

---

#### 3.6 PendingToolCall Model (2 min)

**What is PendingToolCall?**
- A tool call awaiting user confirmation
- Contains the original ToolCall
- Has a confirmation message to show user

**Implementation:**
```python
class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    
    tool_call: ToolCall
    confirmation_message: str
```

**Key Points:**
- Wraps the original ToolCall
- `confirmation_message` explains what will happen
- Agent pauses until user responds

**Flow:**
1. Agent decides to call dangerous tool
2. Creates PendingToolCall with message
3. Returns to user for approval
4. User submits ToolConfirmation
5. Agent continues or skips based on approval

**Live Coding**: Build PendingToolCall model

---

#### 3.7 Event Model (4 min)

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

#### 3.8 ExecutionContext Dataclass (5 min)

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
    session_id: Optional[str] = None  # Link to session for persistence
    
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

**Key Fields:**
- `state`: Store pending tool calls, confirmations, custom data
- `session_id`: Links to session for persistence

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

**Test ToolConfirmation:**
```python
# User approves with modification
confirmation = ToolConfirmation(
    tool_call_id="call_123",
    approved=True,
    modified_arguments={"expression": "2+3"}  # Changed!
)
print(confirmation.approved)  # True

# User rejects with reason
rejection = ToolConfirmation(
    tool_call_id="call_456",
    approved=False,
    reason="I don't want to delete that file"
)
print(rejection.reason)  # "I don't want to delete that file"
```

**Test PendingToolCall:**
```python
pending = PendingToolCall(
    tool_call=tool_call,
    confirmation_message="The agent wants to calculate '2+2'. Do you approve?"
)
print(pending.confirmation_message)
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

# Store pending tool calls in state
context.state["pending_tool_calls"] = [pending.model_dump()]
```

---

### 5. Why Pydantic vs Dataclass? (2 min)

**Pydantic (Message, ToolCall, ToolResult, Event, ToolConfirmation, PendingToolCall):**
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
- Confirmation workflow

---

### 7. Next Steps (1 min)

**Preview Episode 4:**
- Building the LLM client
- Converting our models to API format
- Parsing API responses

**What We Built:**
- 7 Pydantic models: Message, ToolCall, ToolResult, ToolConfirmation, PendingToolCall, Event
- 1 Dataclass: ExecutionContext
- ContentItem union type
- Complete data model structure matching actual codebase

---

## Key Takeaways

1. **Pydantic** validates data at runtime
2. **Literal types** constrain values
3. **Union types** enable polymorphism
4. **Field(default_factory=...)** prevents shared defaults
5. **Dataclass** for mutable internal state
6. **ToolConfirmation** enables user approval workflow
7. **PendingToolCall** pauses execution for confirmation

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

**Mistake 3: Forgetting optional fields**
```python
# Wrong - reason required even for approval
class BadConfirmation(BaseModel):
    approved: bool
    reason: str  # Always required!

# Right - reason optional
class GoodConfirmation(BaseModel):
    approved: bool
    reason: str | None = None  # Only needed for rejection
```

---

## Exercises

1. **Build ToolConfirmation Validator**: Add validation that `reason` is required when `approved=False`
2. **Create Helper Function**: Write `extract_pending_calls(context: ExecutionContext) -> List[PendingToolCall]`
3. **Add Metadata to Event**: Add an optional `metadata: dict` field to Event for custom data
4. **Create ToolCallWithResult Model**: Combine ToolCall and ToolResult into a single model for reporting

---

## Complete models.py File

```python
"""Core data models for the agent framework."""

from typing import Literal, Union, List, Dict, Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import uuid
from datetime import datetime


class Message(BaseModel):
    """A text message in the conversation."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


ContentItem = Union[Message, ToolCall, ToolResult]


class ToolConfirmation(BaseModel):
    """User's decision on a pending tool call."""
    
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None


class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    
    tool_call: ToolCall
    confirmation_message: str


class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: List[ContentItem] = Field(default_factory=list)


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
        self.events.append(event)
    
    def increment_step(self):
        self.current_step += 1
```

---

**Previous Episode**: [Episode 2: Your First LLM Call](./EPISODE_02_LLM_CALL.md)  
**Next Episode**: [Episode 4: The LLM Client](./EPISODE_04_LLM_CLIENT.md)
