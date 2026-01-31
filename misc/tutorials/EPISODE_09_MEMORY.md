# Episode 9: Session & Memory Management

**Duration**: 35 minutes  
**What to Build**: `agent_framework/memory.py`, `agent_framework/callbacks.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Session persistence
- Memory optimization
- Token management

**Hook Statement**: "Today we'll add memory to our agent - it will remember conversations and optimize token usage. This makes agents truly useful for real applications!"

---

### 2. Problem (3 min)
**Why do we need memory management?**

**The Challenge:**
- Conversations get long
- Token costs increase
- Context windows limited
- Need to remember across sessions

**The Solution:**
- Session persistence
- Token counting
- Memory optimization strategies
- Callback system

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

### 4. Live Coding: Building Memory System (25 min)

#### Step 1: Session Models (3 min)
```python
from pydantic import BaseModel, Field
from datetime import datetime
from .models import Event

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
- Stores events
- Custom state
- Timestamps

**Live Coding**: Build Session model

---

#### Step 2: Session Manager (5 min)
```python
from abc import ABC, abstractmethod

class BaseSessionManager(ABC):
    """Abstract base class for session management."""
    
    @abstractmethod
    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        pass
    
    @abstractmethod
    async def get(self, session_id: str) -> Session | None:
        pass
    
    @abstractmethod
    async def save(self, session: Session) -> None:
        pass
    
    async def get_or_create(self, session_id: str, user_id: str | None = None) -> Session:
        session = await self.get(session_id)
        if session is None:
            session = await self.create(session_id, user_id)
        return session

class InMemorySessionManager(BaseSessionManager):
    """In-memory session storage."""
    
    def __init__(self):
        self._sessions: dict[str, Session] = {}
    
    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        session = Session(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session
    
    async def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)
    
    async def save(self, session: Session) -> None:
        session.updated_at = datetime.now()
        self._sessions[session.session_id] = session
```

**Key Points:**
- Abstract interface
- In-memory implementation
- Easy to extend (database, Redis, etc.)

**Live Coding**: Build session managers

---

#### Step 3: Token Counting (4 min)
```python
import tiktoken
from .llm import build_messages

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
- Uses tiktoken
- Counts messages, tool calls, tools
- Model-specific encoding

**Live Coding**: Build token counting

---

#### Step 4: Sliding Window (4 min)
```python
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

#### Step 5: Compaction (4 min)
```python
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

#### Step 6: Callback System (3 min)
```python
from .callbacks import create_optimizer_callback

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

#### Step 7: Integrating with Agent (2 min)
```python
from agent_framework.memory import apply_sliding_window, create_optimizer_callback

optimizer = create_optimizer_callback(
    apply_optimization=apply_sliding_window,
    threshold=30000
)

agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[calculator],
    before_llm_callback=optimizer  # Add callback
)
```

**Key Points:**
- Add to agent
- Automatic optimization
- Configurable threshold

**Live Coding**: Integrate with agent

---

### 5. Demo: Memory in Action (3 min)

**Show:**
- Session persistence
- Token counting
- Sliding window
- Long conversation handling

---

### 6. Next Steps (1 min)

**Preview Episode 10:**
- Web deployment
- FastAPI backend
- Frontend interface

**What We Built:**
- Session management
- Memory optimization
- Token management

---

## Key Takeaways

1. **Sessions** persist conversations
2. **Token counting** tracks usage
3. **Optimization strategies** reduce costs
4. **Callbacks** enable extensibility
5. **Configurable** thresholds

---

## Common Mistakes

**Mistake 1: Not saving sessions**
```python
# Wrong - loses state
result = await agent.run("Hello", session_id="user-123")
# Missing: session.save()

# Right - persists state
result = await agent.run("Hello", session_id="user-123")
# Agent automatically saves
```

**Mistake 2: Too aggressive optimization**
```python
# Wrong - loses important context
window_size = 5  # Too small!

# Right - balanced
window_size = 20  # Keeps enough context
```

---

## Exercises

1. Implement database session manager
2. Add summarization strategy
3. Create token usage dashboard
4. Add memory analytics

---

**Previous Episode**: [Episode 8: MCP Integration](./EPISODE_08_MCP.md)  
**Next Episode**: [Episode 10: Web Deployment](./EPISODE_10_WEB_DEPLOYMENT.md)

