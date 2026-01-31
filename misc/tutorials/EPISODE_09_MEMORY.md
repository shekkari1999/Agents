# Episode 9: Session & Memory Management

**Duration**: 40 minutes  
**What to Build**: `agent_framework/memory.py`, session integration in `agent.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Session persistence
- Memory optimization
- Token management
- Full session integration with agent

**Hook Statement**: "Today we'll add memory to our agent - it will remember conversations across multiple interactions and optimize token usage. This makes agents truly useful for real applications!"

---

### 2. Problem (3 min)
**Why do we need memory management?**

**The Challenge:**
- Conversations get long
- Token costs increase
- Context windows limited
- Need to remember across sessions
- State must persist across agent runs

**The Solution:**
- Session persistence
- Token counting
- Memory optimization strategies
- Full integration with agent loop

---

### 3. Concept: Memory Strategies (5 min)

**Strategies:**
1. **Sliding Window**: Keep recent N messages
2. **Compaction**: Replace tool calls/results with references
3. **Summarization**: Compress old history with LLM

**When to Use:**
- Sliding Window: Simple, fast
- Compaction: Tool-heavy conversations
- Summarization: Very long conversations

---

### 4. Live Coding: Building Memory System (30 min)

#### Step 1: Session Models (3 min)
```python
# In agent_framework/models.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any

class Session(BaseModel):
    """Container for persistent conversation state."""
    session_id: str
    user_id: str | None = None
    events: list[Event] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
```

**Key Points:**
- Stores events from conversation
- Custom state for pending calls, etc.
- Timestamps for tracking

**Live Coding**: Build Session model

---

#### Step 2: Session Manager (5 min)
```python
# In agent_framework/models.py
from abc import ABC, abstractmethod

class BaseSessionManager(ABC):
    """Abstract base class for session management."""
    
    @abstractmethod
    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        """Create a new session."""
        pass
    
    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID. Returns None if not found."""
        pass
    
    @abstractmethod
    async def save(self, session: Session) -> None:
        """Persist session changes to storage."""
        pass
    
    async def get_or_create(self, session_id: str, user_id: str | None = None) -> Session:
        """Get existing session or create new one."""
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id, user_id)
        return session


class InMemorySessionManager(BaseSessionManager):
    """In-memory session storage for development and testing."""
    
    def __init__(self):
        self._sessions: dict[str, Session] = {}
    
    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        """Create a new session."""
        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        session = Session(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session
    
    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)
    
    async def save(self, session: Session) -> None:
        """Save session to storage."""
        self._sessions[session.session_id] = session
```

**Key Points:**
- Abstract interface for flexibility
- In-memory implementation for development
- Easy to extend (database, Redis, etc.)

**Live Coding**: Build session managers

---

#### Step 3: Session Integration in Agent (8 min)

**Update ExecutionContext:**
```python
# In agent_framework/models.py
@dataclass
class ExecutionContext:
    """Central storage for all execution state."""
    
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    session_id: Optional[str] = None  # NEW: Link to session for persistence
```

**Update Agent.__init__:**
```python
class Agent:
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 5,
        session_manager: BaseSessionManager | None = None  # NEW
    ):
        # ... other init ...
        self.session_manager = session_manager or InMemorySessionManager()
```

**Update Agent.run() with Full Session Support:**
```python
async def run(
    self, 
    user_input: str, 
    context: ExecutionContext = None,
    session_id: Optional[str] = None,
    tool_confirmations: Optional[List[ToolConfirmation]] = None
) -> AgentResult:
    """Execute the agent with optional session support.
    
    Args:
        user_input: User's input message
        context: Optional execution context (creates new if None)
        session_id: Optional session ID for persistent conversations
        tool_confirmations: Optional list of tool confirmations for pending calls
    """
    # Load or create session if session_id is provided
    session = None
    if session_id and self.session_manager:
        session = await self.session_manager.get_or_create(session_id)
        
        # Load session data into context if context is new
        if context is None:
            context = ExecutionContext()
            # Restore events and state from session
            context.events = session.events.copy()
            context.state = session.state.copy()
            context.execution_id = session.session_id
        context.session_id = session_id

    if tool_confirmations:
        if context is None:
            context = ExecutionContext()
        context.state["tool_confirmations"] = [
            c.model_dump() for c in tool_confirmations
        ]
    
    # Create or reuse context
    if context is None:
        context = ExecutionContext()
    
    # Add user input as the first event
    user_event = Event(
        execution_id=context.execution_id,
        author="user",
        content=[Message(role="user", content=user_input)]
    )
    context.add_event(user_event)
    
    # Execute steps until completion or max steps reached
    while not context.final_result and context.current_step < self.max_steps:
        await self.step(context)
        
        # Check for pending confirmations after each step
        if context.state.get("pending_tool_calls"):
            pending_calls = [
                PendingToolCall.model_validate(p)
                for p in context.state["pending_tool_calls"]
            ]
            # Save session state before returning
            if session:
                session.events = context.events
                session.state = context.state
                await self.session_manager.save(session)
            return AgentResult(
                status="pending",
                context=context,
                pending_tool_calls=pending_calls,
            )
        
        # Check if the last event is a final response
        last_event = context.events[-1]
        if self._is_final_response(last_event):
            context.final_result = self._extract_final_result(last_event)
    
    # Save session after execution completes
    if session:
        session.events = context.events
        session.state = context.state
        await self.session_manager.save(session)

    return AgentResult(output=context.final_result, context=context)
```

**Key Points:**
- Load session at start of run()
- Restore events and state from session
- Set `context.session_id` for tracking
- Save session before returning pending
- Save session after completion

**Live Coding**: Integrate session with agent

---

#### Step 4: Token Counting (4 min)
```python
# In agent_framework/memory.py
import tiktoken
import json
from .llm import build_messages
from .models import LlmRequest

def count_tokens(request: LlmRequest, model_id: str = "gpt-4") -> int:
    """Calculate total token count of LlmRequest."""
    try:
        encoding = tiktoken.encoding_for_model(model_id)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    
    messages = build_messages(request)
    total_tokens = 0
    
    for message in messages:
        total_tokens += 4  # Per-message overhead
        
        if message.get("content"):
            total_tokens += len(encoding.encode(message["content"]))
        
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                func = tool_call.get("function", {})
                if func.get("name"):
                    total_tokens += len(encoding.encode(func["name"]))
                if func.get("arguments"):
                    total_tokens += len(encoding.encode(func["arguments"]))
    
    if request.tools:
        for tool in request.tools:
            tool_def = tool.tool_definition
            total_tokens += len(encoding.encode(json.dumps(tool_def)))
    
    return total_tokens
```

**Key Points:**
- Uses tiktoken for accurate counting
- Counts messages, tool calls, tools
- Model-specific encoding

**Live Coding**: Build token counting

---

#### Step 5: Sliding Window (4 min)
```python
# In agent_framework/memory.py
from .models import Message

def apply_sliding_window(
    context: ExecutionContext,
    request: LlmRequest,
    window_size: int = 20
) -> None:
    """Keep only the most recent N messages."""
    contents = request.contents
    
    # Find user message position
    user_message_idx = None
    for i, item in enumerate(contents):
        if isinstance(item, Message) and item.role == "user":
            user_message_idx = i
            break
    
    if user_message_idx is None:
        return
    
    # Preserve up to user message
    preserved = contents[:user_message_idx + 1]
    
    # Keep only the most recent N from remaining items
    remaining = contents[user_message_idx + 1:]
    if len(remaining) > window_size:
        remaining = remaining[-window_size:]
    
    request.contents = preserved + remaining
```

**Key Points:**
- Preserves user message
- Keeps recent N items
- Simple and fast

**Live Coding**: Build sliding window

---

#### Step 6: Compaction (4 min)
```python
# In agent_framework/memory.py
from .models import ToolCall, ToolResult

TOOLRESULT_COMPACTION_RULES = {
    "read_file": "File content from {file_path}. Re-read if needed.",
    "search_web": "Search results processed. Query: {query}. Re-search if needed.",
}

def apply_compaction(context: ExecutionContext, request: LlmRequest) -> None:
    """Compress tool calls and results into reference messages."""
    tool_call_args = {}
    compacted = []
    
    for item in request.contents:
        if isinstance(item, ToolCall):
            tool_call_args[item.tool_call_id] = item.arguments
            compacted.append(item)  # Keep tool calls
            
        elif isinstance(item, ToolResult):
            if item.name in TOOLRESULT_COMPACTION_RULES:
                args = tool_call_args.get(item.tool_call_id, {})
                template = TOOLRESULT_COMPACTION_RULES[item.name]
                compressed_content = template.format(
                    file_path=args.get("file_path", "unknown"),
                    query=args.get("query", "unknown")
                )
                compacted.append(ToolResult(
                    tool_call_id=item.tool_call_id,
                    name=item.name,
                    status=item.status,
                    content=[compressed_content]
                ))
            else:
                compacted.append(item)
        else:
            compacted.append(item)
    
    request.contents = compacted
```

**Key Points:**
- Replaces tool results with references
- Configurable rules
- Reduces token count

**Live Coding**: Build compaction

---

#### Step 7: Optimizer Callback (3 min)
```python
# In agent_framework/callbacks.py
import inspect
from typing import Callable, Optional
from .models import ExecutionContext
from .llm import LlmRequest, LlmResponse
from .memory import count_tokens

def create_optimizer_callback(
    apply_optimization: Callable,
    threshold: int = 50000,
    model_id: str = "gpt-4"
) -> Callable:
    """Factory for optimization callbacks."""
    async def callback(
        context: ExecutionContext,
        request: LlmRequest
    ) -> Optional[LlmResponse]:
        token_count = count_tokens(request, model_id=model_id)
        
        if token_count < threshold:
            return None
        
        result = apply_optimization(context, request)
        if inspect.isawaitable(result):
            await result
        return None
    
    return callback
```

**Key Points:**
- Factory function
- Checks threshold
- Supports sync/async

**Live Coding**: Build callback system

---

### 5. Testing Session Integration (3 min)

**Test Session Persistence:**
```python
import asyncio
from agent_framework import Agent, LlmClient, InMemorySessionManager
from agent_tools import calculator

# Create a shared session manager
session_manager = InMemorySessionManager()

# Create agent with session support
agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator],
    instructions="You are a helpful assistant.",
    session_manager=session_manager
)

session_id = "user-123"

# First conversation - introduce yourself
result1 = await agent.run(
    "Hi! My name is Alice and I'm a software engineer.",
    session_id=session_id
)
print(f"Response 1: {result1.output}")
print(f"Events: {len(result1.context.events)}")

# Second conversation - continue
result2 = await agent.run(
    "What's 1234 * 5678?",
    session_id=session_id
)
print(f"Response 2: {result2.output}")
print(f"Events: {len(result2.context.events)}")  # Should include previous events!

# Third conversation - test memory
result3 = await agent.run(
    "What's my name and what do I do for work?",
    session_id=session_id
)
print(f"Response 3: {result3.output}")
# Should remember: "Your name is Alice and you're a software engineer."

# Different session - should NOT remember
result4 = await agent.run(
    "What's my name?",
    session_id="different-user"
)
print(f"Response 4: {result4.output}")
# Should say it doesn't know
```

**Test Session Isolation:**
```python
# Check stored sessions
print("Session Storage Summary:")
for sid, session in session_manager._sessions.items():
    print(f"Session ID: {session.session_id}")
    print(f"  Events: {len(session.events)}")
    print(f"  State keys: {list(session.state.keys())}")
```

---

### 6. Demo: Memory in Action (3 min)

**Show:**
- Session persistence across runs
- Token counting
- Sliding window optimization
- Long conversation handling
- Session isolation between users

---

### 7. Next Steps (1 min)

**Preview Episode 10:**
- Web deployment
- FastAPI backend
- Frontend interface

**What We Built:**
- Session model and managers
- Full session integration with Agent.run()
- Token counting
- Memory optimization strategies

---

## Key Takeaways

1. **Sessions** persist conversations across multiple run() calls
2. **session_id** links context to session
3. **Events and state** are restored from session
4. **Session saves** on pending and completion
5. **Token counting** tracks usage
6. **Optimization strategies** reduce costs

---

## Common Mistakes

**Mistake 1: Not loading session state**
```python
# Wrong - creates empty context
if context is None:
    context = ExecutionContext()

# Right - loads from session
if context is None:
    context = ExecutionContext()
    context.events = session.events.copy()
    context.state = session.state.copy()
```

**Mistake 2: Not saving before pending return**
```python
# Wrong - loses state on pending
if context.state.get("pending_tool_calls"):
    return AgentResult(status="pending", ...)

# Right - saves before returning
if context.state.get("pending_tool_calls"):
    if session:
        session.events = context.events
        session.state = context.state
        await self.session_manager.save(session)
    return AgentResult(status="pending", ...)
```

**Mistake 3: Mutating session directly**
```python
# Wrong - might not persist
session.events.append(new_event)

# Right - copy and save
session.events = context.events  # Full replacement
await self.session_manager.save(session)
```

---

## Exercises

1. **Implement Database Session Manager**: Create a PostgreSQL or SQLite session manager
2. **Add Session Expiry**: Auto-delete sessions after 24 hours of inactivity
3. **Build Token Dashboard**: Track token usage per session
4. **Add Summarization Strategy**: Use LLM to summarize old history

---

## Complete Session Flow

```
User Request with session_id="user-123"
    |
    v
Agent.run(session_id="user-123")
    |
    v
session_manager.get_or_create("user-123")
    |
    v
[New Session?] --Yes--> Create empty Session
    |                        |
    No                       |
    |                        v
    v                   context = ExecutionContext()
Load existing Session       |
    |                        v
    v               context.events = []
context = ExecutionContext()
context.events = session.events.copy()
context.state = session.state.copy()
    |
    v
Execute agent loop (step, step, ...)
    |
    +--[Pending?]--Yes--> Save session, return pending
    |
    No
    |
    v
Complete execution
    |
    v
session.events = context.events
session.state = context.state
await session_manager.save(session)
    |
    v
Return AgentResult(status="complete")
```

---

**Previous Episode**: [Episode 8: MCP Integration](./EPISODE_08_MCP.md)  
**Next Episode**: [Episode 10: Web Deployment](./EPISODE_10_WEB_DEPLOYMENT.md)
