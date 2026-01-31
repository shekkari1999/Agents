# Exercises and Challenges

This document contains exercises for each episode to reinforce learning.

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

### Exercise 1: Add Metadata Field
Add a `metadata` field to the `Event` model that stores arbitrary key-value pairs.

**Solution:**
```python
class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: List[ContentItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Added
```

### Exercise 2: Validation
Add validation to ensure Message content is not empty.

**Solution:**
```python
from pydantic import field_validator

class Message(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v
```

### Exercise 3: Extract Messages Helper
Create a helper function to extract all messages from a list of events.

**Solution:**
```python
def extract_messages(events: List[Event]) -> List[Message]:
    """Extract all messages from events."""
    messages = []
    for event in events:
        for item in event.content:
            if isinstance(item, Message):
                messages.append(item)
    return messages
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

### Exercise 1: Optional Type Support
Add support for `Optional` types in schema generation.

**Solution:**
```python
from typing import get_origin, get_args

def function_to_input_schema(func) -> dict:
    # ... existing code ...
    
    for param in signature.parameters.values():
        # Handle Optional types
        if get_origin(param.annotation) is Union:
            args = get_args(param.annotation)
            if type(None) in args:
                # It's Optional, use the non-None type
                param_type = type_map.get(args[0], "string")
            else:
                param_type = type_map.get(param.annotation, "string")
        else:
            param_type = type_map.get(param.annotation, "string")
        
        parameters[param.name] = {"type": param_type}
        
        # Don't require if has default or is Optional
        if param.default != inspect._empty or get_origin(param.annotation) is Union:
            # Don't add to required
            pass
        else:
            required.append(param.name)
```

### Exercise 2: Tool Registry
Create a tool registry to manage all available tools.

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

# Usage
registry = ToolRegistry()
registry.register(calculator)
registry.register(search_web)
```

---

## Episode 7: Tool Execution

### Exercise 1: Tool Execution Timeout
Add timeout support for tool execution.

**Solution:**
```python
import asyncio

async def act(self, context: ExecutionContext, tool_calls: List[ToolCall], timeout: int = 30):
    """Execute tool calls with timeout."""
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []
    
    for tool_call in tool_calls:
        tool = tools_dict[tool_call.name]
        
        try:
            tool_response = await asyncio.wait_for(
                tool(context, **tool_call.arguments),
                timeout=timeout
            )
            status = "success"
        except asyncio.TimeoutError:
            tool_response = f"Tool execution timed out after {timeout} seconds"
            status = "error"
        except Exception as e:
            tool_response = str(e)
            status = "error"
        
        # ... create ToolResult ...
```

### Exercise 2: Tool Usage Statistics
Track which tools are used and how often.

**Solution:**
```python
class Agent:
    def __init__(self, ...):
        # ... existing code ...
        self.tool_stats: Dict[str, int] = {}
    
    async def act(self, context: ExecutionContext, tool_calls: List[ToolCall]):
        # ... existing code ...
        
        for tool_call in tool_calls:
            # Track usage
            self.tool_stats[tool_call.name] = self.tool_stats.get(tool_call.name, 0) + 1
            
            # ... execute tool ...
    
    def get_tool_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return self.tool_stats.copy()
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
                # Try to list tools as health check
                await session.list_tools()
                return True
    except Exception:
        return False
```

---

## Episode 9: Session & Memory Management

### Exercise 1: Database Session Manager
Implement a database-backed session manager.

**Solution:**
```python
import sqlite3
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

### Exercise 2: Token Usage Dashboard
Create a function to display token usage statistics.

**Solution:**
```python
def display_token_usage(context: ExecutionContext, model_id: str = "gpt-4"):
    """Display token usage statistics."""
    from agent_framework.memory import count_tokens
    from agent_framework.llm import LlmRequest
    
    total_tokens = 0
    request_count = 0
    
    for event in context.events:
        if event.author != "user":
            # Estimate tokens for this step
            request = LlmRequest(contents=[item for item in event.content])
            tokens = count_tokens(request, model_id)
            total_tokens += tokens
            request_count += 1
    
    print(f"Total Requests: {request_count}")
    print(f"Estimated Total Tokens: {total_tokens}")
    print(f"Average per Request: {total_tokens / request_count if request_count else 0}")
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
            
            # Stream response
            await websocket.send_json({
                "type": "response",
                "content": result.output
            })
```

### Exercise 2: User Authentication
Add basic user authentication.

**Solution:**
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    # Validate token
    # Return user info
    return {"user_id": "user123"}

@app.post("/api/chat")
async def chat(request: ChatRequest, user = Depends(get_current_user)):
    # Use user["user_id"] for session management
    session_id = f"{user['user_id']}-{request.session_id}"
    # ... rest of code ...
```

---

## General Challenges

### Challenge 1: Multi-Agent System
Create a system where multiple agents can collaborate.

### Challenge 2: Tool Marketplace
Build a system to discover and share tools.

### Challenge 3: Advanced Memory
Implement vector-based memory retrieval.

### Challenge 4: Cost Tracking
Track and display API usage costs.

### Challenge 5: Monitoring Dashboard
Create a dashboard to monitor agent performance.

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

