# Episode 2: Your First LLM Call

**Duration**: 25 minutes  
**What to Build**: Simple LLM wrapper script  
**Target Audience**: Intermediate Python developers

---

## Episode Structure

### 1. Hook (2 min)
**Show what we'll build:**
- Simple LLM client that can chat
- Multi-turn conversation
- Error handling

**Hook Statement**: "Today we'll build the foundation that lets us talk to LLMs. This is the bridge between our Python code and the AI."

---

### 2. Problem (3 min)
**Why do we need an LLM client?**

**The Challenge:**
- Different providers have different APIs
- Message format is complex
- Error handling is crucial
- We need a unified interface

**The Solution:**
- Abstract the API complexity
- Standardize message format
- Handle errors gracefully
- Support multiple providers

---

### 3. Concept: Understanding Chat Completion APIs (5 min)

#### 3.1 Message Format

**The Standard Format:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "And 3+3?"},
]
```

**Roles Explained:**
- `system`: Sets behavior/personality (optional but recommended)
- `user`: Human's messages
- `assistant`: AI's previous responses
- `tool`: Tool execution results (we'll cover this later)

**Key Point**: LLMs are **stateless**. You must send the full conversation history each time.

---

#### 3.2 Making API Calls

**Direct OpenAI:**
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**LiteLLM (Multi-Provider):**
```python
from litellm import completion

# OpenAI
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Anthropic (same interface!)
response = completion(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Local models
response = completion(
    model="ollama/llama2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Why LiteLLM?**
- Unified interface for all providers
- Easy to switch models
- Handles provider differences

---

#### 3.3 Async Calls

**Why Async?**
- API calls take 1-5 seconds
- Don't block the program
- Can make multiple calls concurrently

```python
from litellm import acompletion

async def get_response(prompt: str) -> str:
    response = await acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Run it
result = asyncio.run(get_response("What is Python?"))
```

---

### 4. Live Coding: Building SimpleLlmClient (15 min)

#### Step 1: Setup (2 min)
```python
# simple_llm_client.py
import os
from typing import Optional
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()
```

#### Step 2: Response Model (3 min)
```python
from pydantic import BaseModel

class LlmResponse(BaseModel):
    """Standardized response from LLM."""
    content: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error_message is None
```

**Why This Model?**
- Standardized interface
- Error handling built-in
- Easy to check success

#### Step 3: Client Class (5 min)
```python
class SimpleLlmClient:
    """A simple wrapper around LLM API calls."""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
    
    async def generate(self, messages: list[dict]) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            
            return LlmResponse(
                content=response.choices[0].message.content
            )
            
        except Exception as e:
            return LlmResponse(
                error_message=str(e)
            )
```

**Key Points:**
- Wraps API complexity
- Handles errors gracefully
- Returns standardized response

#### Step 4: Testing (3 min)
```python
async def main():
    client = SimpleLlmClient()
    
    response = await client.generate([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"}
    ])
    
    if response.success:
        print(response.content)
    else:
        print(f"Error: {response.error_message}")

import asyncio
asyncio.run(main())
```

#### Step 5: Multi-Turn Conversation (2 min)
```python
async def chat():
    client = SimpleLlmClient()
    
    # Maintain conversation history
    messages = [
        {"role": "system", "content": "You are a math tutor."}
    ]
    
    # Turn 1
    messages.append({"role": "user", "content": "What is 5 x 3?"})
    response = await client.generate(messages)
    print(f"AI: {response.content}")
    
    # Add AI response to history
    messages.append({"role": "assistant", "content": response.content})
    
    # Turn 2 (AI remembers because we sent full history)
    messages.append({"role": "user", "content": "Now divide that by 3"})
    response = await client.generate(messages)
    print(f"AI: {response.content}")  # Should say 5
```

**Key Point**: Must maintain full conversation history.

---

### 5. Understanding the Response (2 min)

**Response Structure:**
```python
response = await acompletion(...)

# The response object
print(response.model)           # "gpt-4o-mini"
print(response.choices)         # List of completions

# The main content
choice = response.choices[0]
print(choice.message.role)      # "assistant"
print(choice.message.content)   # "Hello! How can I help you?"
print(choice.finish_reason)     # "stop" or "tool_calls" or "length"

# Token usage (for cost tracking)
print(response.usage.prompt_tokens)      # Tokens in input
print(response.usage.completion_tokens)  # Tokens in output
print(response.usage.total_tokens)       # Total
```

**Finish Reasons:**
- `stop`: Normal completion
- `tool_calls`: Model wants to use a tool (we'll cover this)
- `length`: Hit max token limit
- `content_filter`: Content was filtered

---

### 6. Error Handling (3 min)

**Common Errors:**
```python
from litellm.exceptions import (
    RateLimitError,
    APIError,
    AuthenticationError,
    Timeout
)

async def safe_completion(messages: list) -> str | None:
    try:
        response = await acompletion(
            model="gpt-4o-mini",
            messages=messages,
            timeout=30
        )
        return response.choices[0].message.content
        
    except AuthenticationError:
        print("Invalid API key!")
        return None
        
    except RateLimitError:
        print("Rate limited - wait and retry")
        await asyncio.sleep(60)
        return await safe_completion(messages)  # Retry
        
    except Timeout:
        print("Request timed out")
        return None
        
    except APIError as e:
        print(f"API error: {e}")
        return None
```

**Best Practices:**
- Always handle errors
- Provide meaningful error messages
- Implement retry logic for rate limits
- Set timeouts

---

### 7. Demo: Working Client (2 min)

**Show:**
- Single message
- Multi-turn conversation
- Error handling
- Different models

---

### 8. Next Steps (1 min)

**Preview Episode 3:**
- Building data models (Message, ToolCall, etc.)
- Why we need structured models
- Pydantic validation

**What We Built:**
- Simple LLM client
- Error handling
- Multi-turn conversations

---

## Key Takeaways

1. LLM APIs are **stateless** - send full history each time
2. **LiteLLM** provides unified interface for multiple providers
3. **Async** is essential for non-blocking operations
4. **Error handling** is crucial for production
5. **Message format** is standardized across providers

---

## Common Mistakes

**Mistake 1: Forgetting conversation history**
```python
# Wrong - AI doesn't remember
response1 = await client.generate([{"role": "user", "content": "My name is Alice"}])
response2 = await client.generate([{"role": "user", "content": "What's my name?"}])  # Doesn't know!

# Right - Maintain history
messages = [{"role": "user", "content": "My name is Alice"}]
response1 = await client.generate(messages)
messages.append({"role": "assistant", "content": response1.content})
messages.append({"role": "user", "content": "What's my name?"})
response2 = await client.generate(messages)  # Knows!
```

**Mistake 2: Not handling errors**
```python
# Wrong - crashes on error
response = await acompletion(...)
print(response.choices[0].message.content)  # Might crash!

# Right - handle errors
try:
    response = await acompletion(...)
    if response.choices[0].message.content:
        print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
```

---

## Exercises

1. Add retry logic with exponential backoff
2. Implement streaming responses
3. Add token usage tracking
4. Support temperature and other parameters

---

**Previous Episode**: [Episode 1: Introduction](./EPISODE_01_INTRODUCTION.md)  
**Next Episode**: [Episode 3: Core Data Models](./EPISODE_03_DATA_MODELS.md)

