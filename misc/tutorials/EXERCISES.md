# Exercises and Challenges

This document contains exercises for each episode to reinforce learning. Exercises are designed to build incrementally toward the actual codebase implementation.

---

## Episode 1: Python Foundations

### Exercise 1: Pydantic Model
Create a `User` model with:
- `name`: string (required)
- `email`: string (required, must contain "@")
- `age`: integer (optional, must be >= 0)
- `is_active`: boolean (default: True)

**Solution:**
```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    email: str
    age: int | None = None
    is_active: bool = True
    
    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Email must contain @')
        return v
    
    @field_validator('age')
    def validate_age(cls, v):
        if v is not None and v < 0:
            raise ValueError('Age must be >= 0')
        return v
```

### Exercise 2: Dataclass with Methods
Create a `ShoppingCart` dataclass with:
- `items`: list of strings (default: empty list)
- `total`: float (default: 0.0)
- Methods: `add_item(name, price)`, `get_total()`

**Solution:**
```python
from dataclasses import dataclass, field

@dataclass
class ShoppingCart:
    items: list[dict] = field(default_factory=list)
    total: float = 0.0
    
    def add_item(self, name: str, price: float):
        self.items.append({"name": name, "price": price})
        self.total += price
    
    def get_total(self) -> float:
        return self.total
```

### Exercise 3: Async Parallel Calls
Write a function that makes 5 concurrent API calls and returns all results.

**Solution:**
```python
import asyncio

async def call_api(id: int) -> str:
    await asyncio.sleep(1)  # Simulate API call
    return f"Result {id}"

async def parallel_calls():
    results = await asyncio.gather(
        call_api(1),
        call_api(2),
        call_api(3),
        call_api(4),
        call_api(5)
    )
    return results

# Run it
results = asyncio.run(parallel_calls())
print(results)
```

---

## Episode 2: Your First LLM Call

### Exercise 1: Retry with Backoff
Implement exponential backoff for rate limit errors.

**Solution:**
```python
import asyncio
from litellm import acompletion
from litellm.exceptions import RateLimitError

async def call_with_retry(messages: list, max_retries: int = 3):
    delay = 1
    for attempt in range(max_retries):
        try:
            response = await acompletion(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
```

### Exercise 2: Streaming Responses
Implement streaming for real-time output.

**Solution:**
```python
from litellm import acompletion

async def stream_response(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    
    response = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    
    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage
async def main():
    async for chunk in stream_response("Tell me a story"):
        print(chunk, end='', flush=True)

asyncio.run(main())
```

### Exercise 3: Temperature Experiment
Call the LLM 5 times with temperature=0 and 5 times with temperature=1. Compare outputs.

**Solution:**
```python
async def temperature_experiment(prompt: str):
    results_temp0 = []
    results_temp1 = []
    
    for _ in range(5):
        response = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        results_temp0.append(response.choices[0].message.content)
    
    for _ in range(5):
        response = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1
        )
        results_temp1.append(response.choices[0].message.content)
    
    print("Temperature 0 (deterministic):")
    for r in results_temp0:
        print(f"  {r}")
    
    print("\nTemperature 1 (creative):")
    for r in results_temp1:
        print(f"  {r}")
```

---

## Episode 3: Core Data Models

### Exercise 1: Build ToolConfirmation Model
Create the `ToolConfirmation` model that captures a user's decision on a pending tool call. It should have:
- `tool_call_id`: string (required) - links to the pending tool call
- `approved`: boolean (required) - whether user approved
- `modified_arguments`: optional dict - if user wants to change arguments
- `reason`: optional string - reason for rejection

**Solution:**
```python
from pydantic import BaseModel

class ToolConfirmation(BaseModel):
    """User's decision on a pending tool call."""
    
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None  # Reason for rejection
```

### Exercise 2: Build PendingToolCall Model
Create the `PendingToolCall` model that wraps a ToolCall awaiting confirmation:
- `tool_call`: ToolCall (required) - the original tool call
- `confirmation_message`: string (required) - message to show user

**Solution:**
```python
class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    
    tool_call: ToolCall  # Assumes ToolCall is already defined
    confirmation_message: str
```

### Exercise 3: Add Validation to ToolConfirmation
Add a validator that requires `reason` when `approved=False`.

**Solution:**
```python
from pydantic import BaseModel, model_validator

class ToolConfirmation(BaseModel):
    """User's decision on a pending tool call."""
    
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None
    
    @model_validator(mode='after')
    def validate_reason_on_rejection(self):
        if not self.approved and not self.reason:
            raise ValueError('reason is required when approved=False')
        return self
```

### Exercise 4: Extract Pending Calls Helper
Create a helper function to extract pending tool calls from ExecutionContext state.

**Solution:**
```python
from typing import List

def extract_pending_calls(context: ExecutionContext) -> List[PendingToolCall]:
    """Extract all pending tool calls from context state."""
    raw_pending = context.state.get("pending_tool_calls", [])
    return [PendingToolCall.model_validate(p) for p in raw_pending]
```

---

## Episode 4: The LLM Client

### Exercise 1: Add Streaming Support
Add streaming support to `LlmClient`.

**Solution:**
```python
async def generate_streaming(self, request: LlmRequest):
    """Generate streaming response from LLM."""
    messages = self._build_messages(request)
    
    response = await acompletion(
        model=self.model,
        messages=messages,
        stream=True
    )
    
    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### Exercise 2: Response Caching
Implement response caching based on request hash.

**Solution:**
```python
import hashlib
import json

class LlmClient:
    def __init__(self, model: str, cache: dict = None, **config):
        self.model = model
        self.config = config
        self.cache = cache or {}
    
    def _get_cache_key(self, request: LlmRequest) -> str:
        """Generate cache key from request."""
        data = {
            "model": self.model,
            "instructions": request.instructions,
            "contents": [c.model_dump() for c in request.contents]
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    async def generate(self, request: LlmRequest) -> LlmResponse:
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        response = await self._generate_uncached(request)
        self.cache[cache_key] = response
        return response
```

---

## Episode 5: The Basic Agent Loop

### Exercise 1: Add Verbose Logging
Add verbose logging to show agent thinking process.

**Solution:**
```python
import logging

class Agent:
    def __init__(self, ..., verbose: bool = False):
        # ... existing code ...
        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)
    
    async def step(self, context: ExecutionContext):
        if self.verbose:
            logging.info(f"Step {context.current_step + 1}: Thinking...")
        
        llm_request = self._prepare_llm_request(context)
        llm_response = await self.think(llm_request)
        
        if self.verbose:
            logging.info(f"Step {context.current_step + 1}: Got response")
            for item in llm_response.content:
                if isinstance(item, Message):
                    logging.info(f"  Message: {item.content[:100]}")
```

### Exercise 2: Step-by-Step Trace
Implement a method to display step-by-step trace.

**Solution:**
```python
def display_step_trace(self, context: ExecutionContext, step: int):
    """Display trace for a specific step."""
    if step >= len(context.events):
        print(f"Step {step} does not exist")
        return
    
    event = context.events[step]
    print(f"\n{'='*60}")
    print(f"Step {step + 1} - {event.author.upper()}")
    print(f"{'='*60}")
    
    for item in event.content:
        if isinstance(item, Message):
            print(f"[Message] {item.role}: {item.content}")
        elif isinstance(item, ToolCall):
            print(f"[Tool Call] {item.name}({item.arguments})")
        elif isinstance(item, ToolResult):
            print(f"[Tool Result] {item.name}: {item.status}")
```

---

## Episode 6: Building the Tool System

### Exercise 1: Add requires_confirmation to a Tool
Create a `delete_file` tool that requires confirmation before execution.

**Solution:**
```python
from agent_framework import tool

@tool(
    requires_confirmation=True,
    confirmation_message="Delete file '{arguments[filename]}'? This cannot be undone."
)
def delete_file(filename: str) -> str:
    """Delete a file from the filesystem."""
    import os
    os.remove(filename)
    return f"Deleted {filename}"

# Test it
print(delete_file.requires_confirmation)  # True
print(delete_file.get_confirmation_message({"filename": "test.txt"}))
# "Delete file 'test.txt'? This cannot be undone."
```

### Exercise 2: Create Custom Confirmation Message Template
Create a `send_email` tool with a detailed confirmation message.

**Solution:**
```python
@tool(
    requires_confirmation=True,
    confirmation_message=(
        "Send email?\n"
        "  To: {arguments[recipient]}\n"
        "  Subject: {arguments[subject]}\n"
        "  Body preview: {arguments[body][:50]}..."
    )
)
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    # ... email sending logic ...
    return f"Email sent to {recipient}"

# Test confirmation message
msg = send_email.get_confirmation_message({
    "recipient": "user@example.com",
    "subject": "Hello",
    "body": "This is a test email with some content that is quite long..."
})
print(msg)
```

### Exercise 3: Tool Registry with Confirmation Status
Create a tool registry that tracks confirmation requirements.

**Solution:**
```python
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_all(self) -> List[BaseTool]:
        """List all registered tools."""
        return list(self._tools.values())
    
    def list_dangerous(self) -> List[BaseTool]:
        """List tools requiring confirmation."""
        return [t for t in self._tools.values() if t.requires_confirmation]
    
    def list_safe(self) -> List[BaseTool]:
        """List tools not requiring confirmation."""
        return [t for t in self._tools.values() if not t.requires_confirmation]

# Usage
registry = ToolRegistry()
registry.register(calculator)
registry.register(delete_file)
print(f"Safe tools: {[t.name for t in registry.list_safe()]}")
print(f"Dangerous tools: {[t.name for t in registry.list_dangerous()]}")
```

---

## Episode 7: Tool Execution with Confirmation

### Exercise 1: Implement Pending Tool Call Detection
Write the logic to detect when a tool requires confirmation and create a PendingToolCall.

**Solution:**
```python
async def act(
    self, 
    context: ExecutionContext, 
    tool_calls: List[ToolCall]
) -> List[ToolResult]:
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []
    pending_calls = []

    for tool_call in tool_calls:
        tool = tools_dict[tool_call.name]
        
        # Check if confirmation is required
        if tool.requires_confirmation:
            pending = PendingToolCall(
                tool_call=tool_call,
                confirmation_message=tool.get_confirmation_message(
                    tool_call.arguments
                )
            )
            pending_calls.append(pending)
            continue  # Skip execution
        
        # Execute tool normally
        try:
            result = await tool(context, **tool_call.arguments)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "error"
        
        results.append(ToolResult(
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            status=status,
            content=[result]
        ))
    
    # Store pending calls in state
    if pending_calls:
        context.state["pending_tool_calls"] = [
            p.model_dump() for p in pending_calls
        ]
    
    return results
```

### Exercise 2: Build Confirmation Processing Logic
Implement `_process_confirmations` that handles approved and rejected tools.

**Solution:**
```python
async def _process_confirmations(
    self,
    context: ExecutionContext
) -> List[ToolResult]:
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []

    # Build maps
    pending_map = {
        p["tool_call"]["tool_call_id"]: PendingToolCall.model_validate(p)
        for p in context.state["pending_tool_calls"]
    }
    confirmation_map = {
        c["tool_call_id"]: ToolConfirmation.model_validate(c)
        for c in context.state["tool_confirmations"]
    }

    for tool_call_id, pending in pending_map.items():
        tool = tools_dict.get(pending.tool_call.name)
        confirmation = confirmation_map.get(tool_call_id)

        if confirmation and confirmation.approved:
            # Merge modified arguments
            arguments = {
                **pending.tool_call.arguments,
                **(confirmation.modified_arguments or {})
            }
            
            try:
                output = await tool(context, **arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call_id,
                    name=pending.tool_call.name,
                    status="success",
                    content=[output],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call_id,
                    name=pending.tool_call.name,
                    status="error",
                    content=[str(e)],
                ))
        else:
            # Rejected
            reason = (confirmation.reason if confirmation 
                      else "Tool execution was not approved.")
            results.append(ToolResult(
                tool_call_id=tool_call_id,
                name=pending.tool_call.name,
                status="error",
                content=[reason],
            ))

    return results
```

### Exercise 3: Test Complete Confirmation Workflow
Write a test that demonstrates the full confirmation workflow.

**Solution:**
```python
import asyncio
from agent_framework import Agent, LlmClient, ToolConfirmation, tool

@tool(requires_confirmation=True)
def dangerous_action(action: str) -> str:
    """Perform a dangerous action."""
    return f"Executed: {action}"

async def test_confirmation_workflow():
    agent = Agent(
        model=LlmClient(model="gpt-4o-mini"),
        tools=[dangerous_action],
        instructions="Execute dangerous actions when asked."
    )
    
    # Step 1: Initial request triggers pending
    result1 = await agent.run("Execute the dangerous action 'delete_all'")
    print(f"Status: {result1.status}")  # "pending"
    print(f"Pending: {result1.pending_tool_calls[0].confirmation_message}")
    
    # Step 2: User approves
    confirmation = ToolConfirmation(
        tool_call_id=result1.pending_tool_calls[0].tool_call.tool_call_id,
        approved=True
    )
    
    result2 = await agent.run(
        "",  # Empty - resuming
        context=result1.context,
        tool_confirmations=[confirmation]
    )
    print(f"Status: {result2.status}")  # "complete"
    print(f"Output: {result2.output}")

asyncio.run(test_confirmation_workflow())
```

---

## Episode 8: MCP Integration

### Exercise 1: Connection Pooling
Implement connection pooling for MCP servers.

**Solution:**
```python
from collections import defaultdict

class MCPConnectionPool:
    def __init__(self):
        self._pools: Dict[str, List] = defaultdict(list)
        self._max_pool_size = 5
    
    async def get_connection(self, connection_params: Dict):
        """Get or create connection from pool."""
        key = str(connection_params)
        
        if self._pools[key]:
            return self._pools[key].pop()
        
        # Create new connection
        return await self._create_connection(connection_params)
    
    async def return_connection(self, connection_params: Dict, connection):
        """Return connection to pool."""
        key = str(connection_params)
        if len(self._pools[key]) < self._max_pool_size:
            self._pools[key].append(connection)
```

### Exercise 2: MCP Server Health Check
Add health check for MCP servers.

**Solution:**
```python
async def check_mcp_health(connection: Dict) -> bool:
    """Check if MCP server is healthy."""
    try:
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()
                return True
    except Exception:
        return False
```

---

## Episode 9: Session & Memory Management

### Exercise 1: Build Session Restoration Logic
Implement the logic to restore session data into ExecutionContext.

**Solution:**
```python
async def run(
    self, 
    user_input: str, 
    session_id: Optional[str] = None,
    context: ExecutionContext = None
) -> AgentResult:
    # Load session if provided
    session = None
    if session_id and self.session_manager:
        session = await self.session_manager.get_or_create(session_id)
        
        # Restore into context
        if context is None:
            context = ExecutionContext()
            context.events = session.events.copy()
            context.state = session.state.copy()
            context.execution_id = session.session_id
        context.session_id = session_id
    
    if context is None:
        context = ExecutionContext()
    
    # ... rest of run logic ...
```

### Exercise 2: Implement Database Session Manager
Create a SQLite-backed session manager.

**Solution:**
```python
import sqlite3
import json
from datetime import datetime

class DatabaseSessionManager(BaseSessionManager):
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                events TEXT,
                state TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.close()
    
    async def create(self, session_id: str, user_id: str | None = None) -> Session:
        session = Session(session_id=session_id, user_id=user_id)
        await self.save(session)
        return session
    
    async def get(self, session_id: str) -> Session | None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return Session(
            session_id=row[0],
            user_id=row[1],
            events=[Event.model_validate(e) for e in json.loads(row[2])],
            state=json.loads(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5])
        )
    
    async def save(self, session: Session) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, user_id, events, state, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.user_id,
            json.dumps([e.model_dump() for e in session.events]),
            json.dumps(session.state),
            session.created_at.isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
```

### Exercise 3: Session with Pending Tool Calls
Test that pending tool calls persist across session saves.

**Solution:**
```python
async def test_session_with_pending():
    session_manager = InMemorySessionManager()
    
    agent = Agent(
        model=LlmClient(model="gpt-4o-mini"),
        tools=[delete_file],  # requires confirmation
        session_manager=session_manager
    )
    
    session_id = "test-session"
    
    # First call - should return pending
    result1 = await agent.run("Delete test.txt", session_id=session_id)
    assert result1.status == "pending"
    
    # Check session was saved with pending state
    session = await session_manager.get(session_id)
    assert "pending_tool_calls" in session.state
    print(f"Session has {len(session.state['pending_tool_calls'])} pending calls")
    
    # Resume with confirmation
    confirmation = ToolConfirmation(
        tool_call_id=result1.pending_tool_calls[0].tool_call.tool_call_id,
        approved=True
    )
    
    result2 = await agent.run("", session_id=session_id, 
                               tool_confirmations=[confirmation])
    assert result2.status == "complete"
    
    # Check pending was cleared
    session = await session_manager.get(session_id)
    assert "pending_tool_calls" not in session.state
    print("Session pending calls cleared after completion")

asyncio.run(test_session_with_pending())
```

---

## Episode 10: Web Deployment

### Exercise 1: WebSocket Support
Add WebSocket support for streaming responses.

**Solution:**
```python
from fastapi import WebSocket

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    agent = create_agent()
    session_id = None
    
    while True:
        data = await websocket.receive_json()
        message = data.get("message")
        
        if message:
            result = await agent.run(message, session_id=session_id)
            session_id = result.context.session_id
            
            await websocket.send_json({
                "type": "response",
                "content": result.output
            })
```

### Exercise 2: Confirmation UI
Add a UI component for handling tool confirmations.

**Solution:**
```javascript
// In index.html
async function handlePendingToolCalls(pendingCalls) {
    const modal = document.getElementById('confirmationModal');
    const content = document.getElementById('confirmationContent');
    
    content.innerHTML = pendingCalls.map(pending => `
        <div class="pending-call" data-id="${pending.tool_call.tool_call_id}">
            <p>${pending.confirmation_message}</p>
            <button onclick="approveToolCall('${pending.tool_call.tool_call_id}')">
                Approve
            </button>
            <button onclick="rejectToolCall('${pending.tool_call.tool_call_id}')">
                Reject
            </button>
        </div>
    `).join('');
    
    modal.style.display = 'block';
}

async function approveToolCall(toolCallId) {
    const confirmation = {
        tool_call_id: toolCallId,
        approved: true
    };
    
    await resumeWithConfirmation([confirmation]);
}

async function rejectToolCall(toolCallId) {
    const reason = prompt('Reason for rejection:');
    const confirmation = {
        tool_call_id: toolCallId,
        approved: false,
        reason: reason
    };
    
    await resumeWithConfirmation([confirmation]);
}
```

---

## Final Integration Challenge

### Build Complete Agent with All Features
Create an agent that:
1. Uses multiple tools (calculator, search, file operations)
2. Has dangerous tools requiring confirmation
3. Persists sessions across requests
4. Displays execution trace

**Solution:**
```python
import asyncio
from agent_framework import (
    Agent, LlmClient, InMemorySessionManager, 
    ToolConfirmation, format_trace, tool
)
from agent_tools import calculator, search_web

@tool(
    requires_confirmation=True,
    confirmation_message="Delete '{arguments[filename]}'?"
)
def delete_file(filename: str) -> str:
    """Delete a file."""
    return f"Deleted {filename}"

async def main():
    session_manager = InMemorySessionManager()
    
    agent = Agent(
        model=LlmClient(model="gpt-4o-mini"),
        tools=[calculator, search_web, delete_file],
        instructions="You are a helpful assistant with file management capabilities.",
        session_manager=session_manager
    )
    
    session_id = "demo-session"
    
    # Conversation 1: Simple calculation
    result1 = await agent.run("What is 25 * 17?", session_id=session_id)
    print(f"Response: {result1.output}\n")
    
    # Conversation 2: Try dangerous action
    result2 = await agent.run("Delete old_data.txt", session_id=session_id)
    
    if result2.status == "pending":
        print("Agent wants to delete a file!")
        print(f"Message: {result2.pending_tool_calls[0].confirmation_message}")
        
        # Approve the deletion
        confirmation = ToolConfirmation(
            tool_call_id=result2.pending_tool_calls[0].tool_call.tool_call_id,
            approved=True
        )
        
        result2 = await agent.run(
            "", 
            session_id=session_id,
            tool_confirmations=[confirmation]
        )
    
    print(f"Response: {result2.output}\n")
    
    # Conversation 3: Test memory
    result3 = await agent.run(
        "What calculations did I ask about earlier?", 
        session_id=session_id
    )
    print(f"Response: {result3.output}\n")
    
    # Display trace
    print(format_trace(result3.context))

asyncio.run(main())
```

---

## Solutions Location

Solutions can be found in:
- `misc/tutorials/exercises/solutions/`
- Each episode has its own solution file
- Solutions include explanations

---

## Contributing Exercises

Feel free to contribute additional exercises:
1. Fork the repository
2. Add exercise to appropriate episode section
3. Include solution
4. Submit pull request
