# Episode 4: The LLM Client

**Duration**: 30 minutes  
**What to Build**: `agent_framework/llm.py`  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Complete LLM client abstraction
- Request/response models
- Message format conversion

**Hook Statement**: "Today we'll build the bridge between our data models and the LLM API. This client will handle all the complexity of API communication."

---

### 2. Problem (3 min)
**Why do we need an LLM client abstraction?**

**The Challenge:**
- API format is different from our models
- Need to convert back and forth
- Error handling is complex
- Multiple providers have different formats

**The Solution:**
- Request/response models
- Conversion functions
- Unified interface
- Error handling built-in

---

### 3. Concept: Request/Response Pattern (5 min)

**The Pattern:**
1. Create `LlmRequest` from our models
2. Convert to API format
3. Call API
4. Parse response into `LlmResponse`
5. Return our models

**Benefits:**
- Type safety
- Easy to test
- Provider agnostic
- Clear data flow

---

### 4. Live Coding: Building the Client (20 min)

#### Step 1: Request Model (3 min)
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from .models import ContentItem
from .tools import BaseTool

class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[str] = 'auto'
```

**Key Fields:**
- `instructions`: System messages
- `contents`: Conversation history
- `tools`: Available tools
- `tool_choice`: Force tool usage or not

**Live Coding**: Build LlmRequest

---

#### Step 2: Response Model (2 min)
```python
class LlmResponse(BaseModel):
    """Response object from LLM calls."""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Key Fields:**
- `content`: Messages and tool calls
- `error_message`: Error if any
- `usage_metadata`: Token usage

**Live Coding**: Build LlmResponse

---

#### Step 3: build_messages() Function (5 min)

**Purpose**: Convert our models to API format

**Implementation:**
```python
import json
from .models import Message, ToolCall, ToolResult

def build_messages(request: LlmRequest) -> List[dict]:
    """Convert LlmRequest to API message format."""
    messages = []
    
    # Add system instructions
    for instruction in request.instructions:
        messages.append({"role": "system", "content": instruction})
    
    # Convert content items
    for item in request.contents:
        if isinstance(item, Message):
            messages.append({"role": item.role, "content": item.content})
            
        elif isinstance(item, ToolCall):
            tool_call_dict = {
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": json.dumps(item.arguments)
                }
            }
            # Append to previous assistant message if exists
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call_dict]
                })
                
        elif isinstance(item, ToolResult):
            messages.append({
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "content": str(item.content[0]) if item.content else ""
            })
    
    return messages
```

**Key Points:**
- System messages first
- Tool calls attach to assistant messages
- Tool results use "tool" role
- JSON stringify arguments

**Live Coding**: Build build_messages()

---

#### Step 4: LlmClient Class (5 min)
```python
from litellm import acompletion
from typing import Dict, Any

class LlmClient:
    """Client for LLM API calls using LiteLLM."""
    
    def __init__(self, model: str, **config):
        self.model = model
        self.config = config
    
    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = self._build_messages(request)
            tools = [t.tool_definition for t in request.tools] if request.tools else None
           
            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice} 
                   if request.tool_choice else {}),
                **self.config
            )
            
            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))
    
    def _build_messages(self, request: LlmRequest) -> List[dict]:
        """Convert LlmRequest to API message format."""
        return build_messages(request)
```

**Key Points:**
- Wraps LiteLLM
- Handles tools
- Error handling
- Configurable

**Live Coding**: Build LlmClient

---

#### Step 5: _parse_response() Method (5 min)
```python
def _parse_response(self, response) -> LlmResponse:
    """Convert API response to LlmResponse."""
    choice = response.choices[0]
    content_items = []
    
    # Parse message content
    if choice.message.content:
        content_items.append(Message(
            role="assistant",
            content=choice.message.content
        ))

    # Parse tool calls
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            content_items.append(ToolCall(
                tool_call_id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments)
            ))
    
    return LlmResponse(
        content=content_items,
        usage_metadata={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    )
```

**Key Points:**
- Extract content and tool calls
- Parse JSON arguments
- Track token usage
- Return our models

**Live Coding**: Build _parse_response()

---

### 5. Testing the Client (3 min)

**Test Basic Call:**
```python
from agent_framework.llm import LlmClient, LlmRequest
from agent_framework.models import Message

client = LlmClient(model="gpt-4o-mini")

request = LlmRequest(
    instructions=["You are helpful."],
    contents=[Message(role="user", content="Hello!")]
)

response = await client.generate(request)
print(response.content[0].content)  # "Hello! How can I help?"
```

**Test with Tools:**
```python
from agent_framework.tools import FunctionTool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

request = LlmRequest(
    instructions=["Use tools when needed."],
    contents=[Message(role="user", content="What is 2+2?")],
    tools=[calculator]
)

response = await client.generate(request)
# Should contain ToolCall for calculator
```

---

### 6. Error Handling (2 min)

**Built-in Error Handling:**
```python
# Invalid API key
response = await client.generate(request)
if response.error_message:
    print(f"Error: {response.error_message}")
```

**Common Errors:**
- Authentication errors
- Rate limits
- Invalid tool definitions
- Network timeouts

---

### 7. Demo: Complete Client (2 min)

**Show:**
- Basic message
- Multi-turn conversation
- Tool calls
- Error handling

---

### 8. Next Steps (1 min)

**Preview Episode 5:**
- Building the agent loop
- Think-Act-Observe cycle
- Execution context management

**What We Built:**
- Complete LLM client
- Request/response models
- Message conversion

---

## Key Takeaways

1. **Request/Response pattern** provides clean abstraction
2. **build_messages()** converts our models to API format
3. **Error handling** is built into the response
4. **Tool support** is integrated
5. **Provider agnostic** via LiteLLM

---

## Common Mistakes

**Mistake 1: Forgetting to JSON stringify arguments**
```python
# Wrong
"arguments": item.arguments  # Dict, not string!

# Right
"arguments": json.dumps(item.arguments)  # JSON string
```

**Mistake 2: Not handling tool calls in response**
```python
# Wrong - misses tool calls
if choice.message.content:
    return choice.message.content

# Right - check both
if choice.message.content:
    # Add message
if choice.message.tool_calls:
    # Add tool calls
```

---

## Exercises

1. Add streaming support
2. Implement retry logic
3. Add response caching
4. Support multiple response formats

---

**Previous Episode**: [Episode 3: Core Data Models](./EPISODE_03_DATA_MODELS.md)  
**Next Episode**: [Episode 5: The Basic Agent Loop](./EPISODE_05_AGENT_LOOP.md)

